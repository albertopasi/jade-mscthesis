# JADE Approach: Joint Training with Supervised Contrastive Learning

## 1. Overview

JADE extends the Fine-Tuning (FT) pipeline by introducing a **Supervised Contrastive Loss** (SupCon) as an auxiliary training objective alongside the standard Cross-Entropy (CE) loss. The total loss is a weighted sum:

$$\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{CE}} + (1 - \alpha) \cdot \mathcal{L}_{\text{SupCon}}$$

where $\alpha \in [0, 1]$ controls the trade-off between classification accuracy (CE) and representation structure (SupCon).

The motivation is that CE alone optimises for decision boundaries without explicitly structuring the latent space. SupCon encourages same-class representations to cluster tightly while pushing different-class representations apart, yielding more discriminative and generalisable features. This is particularly valuable for EEG classification where within-class variability across subjects is high.

---

## 2. Pipeline Architecture

JADE inherits the two-stage training pipeline from FT:

### Stage 1: LP Warmup (CE only)
- **Encoder**: Frozen (pretrained REVE weights)
- **Classifier head**: Trainable (RMSNorm -> Dropout -> Linear)
- **cls_query_token**: Trainable
- **Projection head**: Frozen (not used)
- **Loss**: CE only
- **Mixup**: Disabled (see Section 7)
- **Purpose**: Warm up the classifier head and query token before introducing LoRA/SupCon

### Stage 2: LoRA Fine-Tuning + SupCon (joint CE + SupCon)
- **Encoder**: LoRA adapters injected (or full fine-tuning)
- **Classifier head**: Trainable
- **cls_query_token**: Trainable
- **Projection head**: Trainable
- **Loss**: $\alpha \cdot \mathcal{L}_{\text{CE}} + (1 - \alpha) \cdot \mathcal{L}_{\text{SupCon}}$
- **Mixup**: Disabled

### Justification for freezing the projection head in Stage 1

The projection head exists solely to provide representations for the SupCon loss. During Stage 1, only CE is active, so the projection head receives no meaningful gradient signal. Training it with random/uninformative gradients during LP warmup would produce a poorly initialised projection space that Stage 2 would need to overcome. Keeping it frozen until the SupCon loss activates ensures clean initialisation.

---

## 3. Model Architecture

### 3.1 Base: ReveClassifierFT (unchanged)

The classifier path is identical to the FT approach:

```
EEG (B, C, T)
    |
    v
REVE Encoder -> (B, C, H, E)     [E=512, H=n_patches]
    |
    v
Rearrange -> (B, C*H, E)
    |
    v
Query Attention:
    query = cls_query_token        [trainable, (1, 1, 512)]
    attn_scores = query @ x^T / sqrt(E)
    attn_weights = softmax(attn_scores)
    context = attn_weights @ x     [(B, 1, 512)]
    |
    +---> [context; x] -> flatten -> (B, (1 + C*H)*E) -> classifier head -> logits
    |                                                      (pooling="no")
    |
    +---> context (B, 512) ---------> projection head -> z (B, 128) -> SupCon loss
```

Key points:
- The classifier head receives the **full flattened representation** (180,736-d for FACED 10s window with `pooling="no"`), preserving the performance advantage of the "no pooling" mode.
- The projection head receives only the **512-d context vector**, a compact semantic summary produced by the query-attention mechanism.

### 3.2 Projection Head

```
Linear(input_dim, 512) -> ReLU -> Linear(512, 128) -> L2-normalize
```

| Component | Value | Justification |
|---|---|---|
| Architecture | 2-layer MLP | SimCLR and Khosla et al. both demonstrate that nonlinear projections outperform linear (+3%) and no projection (+10%). A 2-layer MLP is the standard (Section 5.1). |
| Hidden dim | 512 | Matches the input dimension (REVE embed_dim). Convention is hidden_dim >= input_dim to avoid information bottleneck before the final projection. |
| Output dim | 128 | Standard across SimCLR and Khosla. SimCLR ablations show diminishing returns above 128 and minimal degradation down to 64. With 9 classes, 128 provides ample capacity. |
| Activation | ReLU | Standard nonlinearity. No BatchNorm is used because BN can leak information across batch samples in contrastive settings. |
| L2 normalisation | Applied at output | Required for the SupCon loss to function correctly (Section 5.2). |

### 3.3 Why the Projection Head Uses the Context Vector, Not the Full Representation

The classifier uses the full `(1 + C*H) * E` flattened vector (180,736-d) because the linear head can learn per-patch, per-channel weights, which experiments show yields the best classification accuracy.

However, using this full vector as input to the projection head would require a first linear layer of `Linear(180736, 512)` = **92.5M parameters** — more than the entire REVE encoder (~69M params). This would:
- Dominate training and overfit easily
- Slow down computation significantly
- Defeat the purpose of LoRA (parameter-efficient fine-tuning)

Instead, the projection head taps off the **context vector** (512-d), which is the query-attention output — a compact semantic summary of the full patch sequence. This is architecturally clean:

- **No information loss for classification**: The classifier still sees the full 180k-d vector, unchanged from FT.
- **Effective contrastive learning**: The SupCon loss shapes the encoder's representations through backpropagation. Gradients flow from the projection head through the context vector, through the query-attention mechanism, and into the encoder. This indirectly improves the quality of *all* patch embeddings — including the ones the classifier uses.
- **Parameter efficiency**: The projection head adds only ~330K parameters (512 * 512 + 512 * 128 + biases), negligible relative to the encoder.

The `--supcon-repr` CLI argument allows selecting the input to the projection head:
- `context` (default, 512-d): Query-attention output. Most semantically compressed.
- `mean` (512-d): Mean pool of all C*H patch tokens. Less task-specific but captures global statistics.
- `both` (1024-d): Concatenation of context and mean. Most informative compact representation, doubles projection head input size.

---

## 4. Supervised Contrastive Loss (Khosla et al. 2020, $\mathcal{L}_{\text{sup,out}}$)

### 4.1 Formulation

For a batch of $B$ samples with labels $\{y_1, \ldots, y_B\}$ and L2-normalised projections $\{z_1, \ldots, z_B\}$:

$$\mathcal{L}_{\text{SupCon}} = \frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} \mathcal{L}_i$$

where $\mathcal{I}$ is the set of anchors that have at least one positive pair, and:

$$\mathcal{L}_i = \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}$$

with:
- $P(i) = \{p \in \{1, \ldots, B\} : y_p = y_i, \; p \neq i\}$ — positives (same class, excluding self)
- $A(i) = \{a \in \{1, \ldots, B\} : a \neq i\}$ — all samples except self (denominator)
- $\tau > 0$ — temperature parameter

### 4.2 Choice of SupCon Variant

Khosla et al. define two variants:

- **$\mathcal{L}_{\text{sup,out}}$** (used here): The $1/|P(i)|$ normalisation is **outside** the log. This is the recommended variant — it directly optimises for all positive pairs equally.
- **$\mathcal{L}_{\text{sup,in}}$**: The normalisation is inside the log. Khosla et al. show this is more susceptible to easy positives dominating the gradient.

We use $\mathcal{L}_{\text{sup,out}}$ as recommended by the original paper.

### 4.3 Temperature $\tau$

The temperature controls the sharpness of the similarity distribution in the softmax:

- **Low $\tau$ (e.g., 0.05)**: Sharp distribution. The loss focuses heavily on hard negatives (closest different-class samples). More discriminative but can be unstable — gradients concentrate on a few pairs.
- **High $\tau$ (e.g., 0.5)**: Flat distribution. All pairs contribute more equally. Smoother optimisation but less discriminative.

**Default: $\tau = 0.07$** — the value used by SimCLR and widely adopted. Khosla et al. use 0.07 in their main experiments. This is a reasonable starting point for EEG with 9 classes. The range $\tau \in [0.05, 0.1]$ is worth sweeping.

### 4.4 Singleton Handling

When a class has only one sample in a batch, $|P(i)| = 0$ and the loss for that anchor is undefined. These samples are excluded from the mean ($|\mathcal{I}|$ counts only anchors with at least one positive).

With `batch_size=64` and 9 classes, the expected samples per class is ~7.1. The probability of a singleton class is low but nonzero, especially if class frequencies are imbalanced across stimulus splits. The implementation must handle this gracefully.

### 4.5 Numerical Stability

The `exp(z_i \cdot z_a / \tau)` term can overflow for large similarities or small $\tau$. The standard trick is to subtract the maximum logit before exponentiation:

$$\log \frac{\exp(s_{ip} / \tau)}{\sum_a \exp(s_{ia} / \tau)} = \frac{s_{ip}}{\tau} - \log \sum_a \exp\left(\frac{s_{ia}}{\tau}\right)$$

where the log-sum-exp is computed with the max-subtraction trick:

$$\log \sum_a \exp(x_a) = m + \log \sum_a \exp(x_a - m), \quad m = \max_a x_a$$

With L2-normalised vectors, all similarities lie in $[-1, 1]$, so $s/\tau \in [-1/\tau, 1/\tau]$. At $\tau = 0.07$ this gives a range of $[-14.3, 14.3]$, which is manageable with the max-subtraction trick in float32 (and safe in float16 AMP with careful implementation).

---

## 5. Justification of Key Design Decisions

### 5.1 Why a Nonlinear Projection Head (vs. Linear or None)

SimCLR (Chen et al., 2020) demonstrated empirically that:
- Nonlinear MLP projection: best performance
- Linear projection: -3% accuracy
- No projection (contrastive loss directly on encoder output): -10% accuracy

The theoretical explanation (formalised by "Projection Head is Secretly an Information Bottleneck", Ouyang et al.) is that the contrastive loss **discards information** that is not useful for distinguishing positive from negative pairs. Some of this discarded information may be useful for downstream classification. The projection head acts as an **information buffer**: it absorbs the information destruction caused by the contrastive loss, protecting the encoder representation.

In our setting, the encoder output (context vector) feeds both the classifier and the projection head. Without a projection head, the SupCon loss would directly shape the context vector to discard non-contrastive information — potentially harming classification. With the projection head, the SupCon loss shapes the *projected* space, and only the gradients that improve class separability propagate back to the encoder.

### 5.2 Why L2 Normalisation on Projection Output

L2 normalisation maps all projections onto the unit hypersphere ($\|z\| = 1$). This is required for three reasons:

1. **Cosine similarity via dot product.** The SupCon loss computes pairwise similarity as $z_i \cdot z_p$. For this to equal cosine similarity, vectors must be unit-norm:
   $$\cos(\theta_{ip}) = \frac{z_i \cdot z_p}{\|z_i\| \cdot \|z_p\|} \xrightarrow{\|z\|=1} z_i \cdot z_p$$
   Without normalisation, the loss is dominated by vector magnitudes rather than angular separation — a sample with large norm would appear similar to everything regardless of direction.

2. **Bounded, well-conditioned loss.** After L2-norm, all pairwise similarities lie in $[-1, 1]$, making $\exp(z_i \cdot z_p / \tau)$ bounded. Without normalisation, unbounded dot products would cause exponential overflow/underflow, making training numerically unstable.

3. **Direction-only optimisation.** Contrastive learning should cluster representations by semantic direction (encoding class identity), not by magnitude (encoding "confidence"). L2-norm forces the loss to optimise purely over the angular structure of the embedding space, yielding a cleaner geometric interpretation.

This is universal in contrastive learning: SimCLR, Khosla, MoCo, BYOL, and DINO all L2-normalise projection outputs. Khosla et al. explicitly state: "we normalise the projection to lie on the unit hypersphere" (Section 3).

### 5.3 Why 2-Layer MLP (Not Deeper)

SimCLR v2 (Chen et al., 2020) found that 3-layer MLPs improve performance with very large encoders (ResNet-152x3, ~795M params) in semi-supervised settings. Our encoder (REVE, ~85M params) is much smaller, and we are doing fully supervised training with SupCon as an auxiliary loss. The marginal gain from a 3rd layer is negligible in this regime.

The "Deep Fusion" approach (Li et al., 2025) proposes transformer-based projection heads that capture cross-sample dependencies via attention. While theoretically interesting, this adds substantial complexity and compute for an auxiliary loss — the primary training signal comes from CE.

**Principle: the projection head is not the model.** It is a lens through which the contrastive objective views the representation. Overengineering it provides diminishing returns when SupCon is auxiliary rather than the sole objective.

### 5.4 Why Not Convolutional Projection Heads (CLISA, CL-CS)

CLISA (Shen et al., 2023) and CL-CS (Hu et al., 2025) use convolutional projection heads with depthwise spatial and temporal convolutions. These are designed for encoders that output spatiotemporal feature maps from raw EEG.

Our encoder (REVE) is a transformer that outputs abstract 512-d patch embeddings. By the time we extract the context vector, there is no spatial or temporal structure remaining to exploit. A convolutional projection head would be architecturally mismatched — convolutions require grid-structured inputs, not single vectors.

---

## 6. Gradient Flow

### 6.1 Stage 2 Gradient Paths

```
                                  CE loss
                                    |
EEG -> Encoder(LoRA) -> patches -> query attention -> [context; patches] -> flatten -> classifier head
                            |                |
                            |                v
                            |          context (512-d) -> projection head -> z (128-d)
                            |                                                    |
                            |                                                SupCon loss
                            |
                      (gradients from both losses flow back through encoder)
```

- **Classifier head** receives gradients from CE only.
- **Projection head** receives gradients from SupCon only.
- **Encoder (LoRA params)**, **cls_query_token**, and the **query-attention mechanism** receive gradients from **both** losses via the shared computation path.

This is the desired behaviour: CE optimises for classification accuracy while SupCon structures the representation space. Both objectives jointly shape the encoder's learned features.

### 6.2 No Gradient Routing Required

Because the classifier and projection heads are separate modules with separate inputs to their respective losses, gradient routing is automatic via PyTorch's autograd. No `detach()` or gradient scaling is needed. The `alpha` weighting in the total loss naturally controls the relative gradient magnitude from each objective.

---

## 7. Mixup Incompatibility

Mixup augmentation blends inputs: $\tilde{x} = \lambda x_i + (1 - \lambda) x_j$ with $\lambda \sim \text{Beta}(\alpha, \alpha)$, producing soft labels.

SupCon requires **discrete class labels** to identify positive pairs ($P(i) = \{p : y_p = y_i\}$). A mixup'd sample has ambiguous class membership — it belongs partially to two classes. There is no principled way to define positive pairs for such samples.

Possible workarounds:
- Assign the majority class (whichever has higher $\lambda$) — introduces noise in pair assignment.
- Use mixup for the CE branch only and clean samples for SupCon — requires two forward passes or careful bookkeeping, with marginal benefit.
- Disable mixup entirely — simplest and cleanest.

Additionally, FT experiments on this project show that mixup does not consistently improve results. **Mixup is therefore disabled by default in both stages.**

---

## 8. Batch Size Considerations

SupCon benefits from larger batch sizes because more samples per batch means more positive pairs per anchor. With $B$ samples and $K$ classes (uniform distribution):

| Batch size | Expected samples/class | Expected positives/anchor |
|---|---|---|
| 64 | 7.1 | 6.1 |
| 128 | 14.2 | 13.2 |
| 256 | 28.4 | 27.4 |

With `batch_size=64` and 9 classes, ~6 positive pairs per anchor is workable but on the lower end. Increasing to 128 or 256 is recommended if GPU memory permits.

**Impact on other hyperparameters**: With Adam-family optimisers (StableAdamW), learning rate scaling is less critical than with SGD. The per-parameter adaptive learning rate in Adam/StableAdamW largely absorbs the effect of batch size changes. Recommendation: keep LR unchanged when going from 64 to 128. If going to 256+, consider a modest LR increase (1.5x) and monitor training stability.

Batch size is exposed as the `--batch-size` CLI argument.

---

## 9. Hyperparameters

### 9.1 SupCon-Specific Hyperparameters

| Parameter | CLI flag | Default | Range to sweep | Notes |
|---|---|---|---|---|
| $\alpha$ (CE weight) | `--alpha` | 0.5 | [0.3, 0.5, 0.7, 0.8] | Most impactful hyperparameter |
| $\tau$ (temperature) | `--temperature` | 0.07 | [0.05, 0.07, 0.1] | Lower = harder contrasts |
| Projection output dim | `--proj-dim` | 128 | [64, 128, 256] | 128 is likely optimal |
| Projection hidden dim | `--proj-hidden` | 512 | [256, 512] | Match embed_dim |
| Projection input repr | `--supcon-repr` | `context` | [context, mean, both] | Ablation recommended |

### 9.2 Inherited from FT (unchanged defaults)

| Parameter | LP warmup | FT + SupCon stage |
|---|---|---|
| Optimiser | StableAdamW (betas=0.92/0.999, wd=0.01) | Same |
| LR | 5e-3 | 1e-4 |
| LR schedule | Exp warmup (3 ep) + ReduceLROnPlateau | Exp warmup (5 ep) + ReduceLROnPlateau |
| Epochs | 20 max | 200 max |
| Early stopping | patience=10 (val_acc) | patience=20 |
| Scheduler patience | 6 | 6 |
| Grad clip | max_norm=2.0 | max_norm=2.0 |
| Dropout | 0.05 | 0.1 |
| Batch size | 64 | 64 |
| Precision | AMP float16 | AMP float16 |
| Mixup | Disabled | Disabled |

### 9.3 Priority for Hyperparameter Tuning

1. **$\alpha$** — Controls the balance between objectives. Start with 0.5, sweep [0.3, 0.8].
2. **$\tau$** — Temperature. Start with 0.07, sweep [0.05, 0.1].
3. **Batch size** — More positive pairs. Try 128, 256 if memory allows.
4. **`--supcon-repr`** — Quick ablation: context vs mean vs both.
5. **Projection dims** — Low priority, defaults are well-justified.

---

## 10. Checkpoint Strategy

### 10.1 What to Save

Same as FT, plus the projection head:

- **LoRA adapter**: `lora_adapter/` directory (peft convention)
- **Head weights**: `head_weights.pt` (cls_query_token + classifier head)
- **Projection head**: `projection_head.pt` (saved separately)
- **Fold metadata**: `fold_meta.json`

For full fine-tuning (no LoRA): single `full_model.pt` + `projection_head.pt`.

### 10.2 Why Save the Projection Head Separately

The projection head is **not needed for inference** — classification uses the classifier head only. However, saving it enables:
- Analysis of the learned contrastive embedding space (t-SNE/UMAP visualisations)
- Comparison of projection spaces across $\alpha$, $\tau$ settings
- Potential transfer to other tasks or zero-shot evaluation via nearest-neighbour in projection space

---

## 11. W&B Logging

- **Project**: `eeg-sc-v2`, **Entity**: `zl-tudelft-thesis`
- **Additional metrics** (beyond FT):
  - `train/loss_ce` — CE component of the total loss
  - `train/loss_sc` — SupCon component of the total loss
  - `train/loss_total` — Weighted sum ($\alpha \cdot \text{CE} + (1-\alpha) \cdot \text{SupCon}$)
- **Standard metrics** (inherited): `val/acc`, `val/bal_acc`, `val/auroc`, `val/f1`, `val/loss`
- **Hyperparameters logged**: all FT params + $\alpha$, $\tau$, `proj_dim`, `proj_hidden`, `supcon_repr`

Monitoring both component losses is important for diagnosing training dynamics. If `loss_sc` dominates early training, $\alpha$ may need to be increased. If `loss_sc` plateaus while `loss_ce` keeps dropping, the contrastive objective may not be receiving enough gradient — consider decreasing $\alpha$.

---

## 12. Evaluation Modes

Identical to FT (inherited, no changes):

1. **10-fold cross-subject CV** (default): KFold(n_splits=10, shuffle=True, seed=42). Saves `summary_*.json`.
2. **Static REVE split** (`--revesplit`, FACED only): train 0-79, val 80-99, test 100-122. Single run.
3. **Stimulus generalisation** (`--generalization`): 2/3 stimuli per emotion for train, 1/3 held-out.

---

## 13. CLI Interface

```bash
# All folds, FACED, 9-class (LoRA + SupCon, default)
uv run python -m src.approaches.supcon.train_sc \
    --dataset faced --task 9-class

# Custom SupCon hyperparameters
uv run python -m src.approaches.supcon.train_sc \
    --dataset faced --alpha 0.7 --temperature 0.1

# Different projection head input representation
uv run python -m src.approaches.supcon.train_sc \
    --dataset faced --supcon-repr mean

# Full fine-tuning (no LoRA)
uv run python -m src.approaches.supcon.train_sc --dataset faced --fullft

# Official REVE static split
uv run python -m src.approaches.supcon.train_sc --dataset faced --revesplit

# Generalization evaluation
uv run python -m src.approaches.supcon.train_sc --dataset faced --generalization

# Smoke test
uv run python -m src.approaches.supcon.train_sc --dataset faced --fold 1 --lp-epochs 2 --ft-epochs 3

# Larger batch size
uv run python -m src.approaches.supcon.train_sc --dataset faced --batch-size 128
```

---

## 14. References

1. **Khosla, P. et al.** (2020). "Supervised Contrastive Learning." *NeurIPS 2020.* — SupCon loss formulation ($\mathcal{L}_{\text{sup,out}}$), projection head design, temperature analysis.
2. **Chen, T. et al.** (2020). "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR). *ICML 2020.* — Demonstrated necessity of nonlinear projection head, established $\tau=0.07$ and output dim 128 as defaults.
3. **Chen, T. et al.** (2020). "Big Self-Supervised Models are Strong Semi-Supervised Learners" (SimCLR v2). *NeurIPS 2020.* — 3-layer projection head analysis (marginal gains with large encoders).
4. **Shen, X. et al.** (2023). "Contrastive Learning of Subject-Invariant EEG Representations for Cross-Subject Emotion Recognition" (CLISA). *IEEE TAFFC.* — Convolutional projection head for raw EEG contrastive learning.
5. **Hu, J. et al.** (2025). "CL-CS: Cross-Subject Emotion Recognition Using Contrastive Learning with Cost-Sensitive Learning." — Multi-domain convolutional projection head for EEG.
6. **Ouyang, Z. et al.** "Projection Head is Secretly an Information Bottleneck." — Theoretical justification for projection heads as information buffers.
7. **Li, Y. et al.** (2025). "Deep Fusion: Capturing Dependencies in Contrastive Learning via Transformer Projection Heads." *IEEE.* — Transformer-based projection heads (not applicable to our setting).
