"""
model.py — REVE models for Linear Probing.

Provides:
  - RMSNorm: Root Mean Square Layer Normalization.
  - ReveClassifierLP: LP module with frozen encoder + trainable cls_query_token + trainable linear head.
  - EmbeddingExtractor: utility for pre-computing fixed REVE embeddings (fast mode).
  - LinearProber: Lightning Module training a linear head on pre-computed embeddings.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,

    f1_score,
    roc_auc_score,
    average_precision_score,
)
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel

from src.approaches.linear_probing.config import LPConfig, REVE_MODEL_PATH, REVE_POS_PATH


# ── RMSNorm ──────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (matching REVE backbone's RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


# ── Helper: load REVE model and position bank ────────────────────────────────

def load_reve_and_positions(
    channel_names: list[str],
    device: str = "cuda",
    reve_model_path: Path = REVE_MODEL_PATH,
    reve_pos_path: Path = REVE_POS_PATH,
) -> Tuple[nn.Module, Tensor]:
    """Load frozen REVE encoder and compute electrode position tensor.

    Returns:
        reve_model: The pretrained REVE model (eval mode, all params frozen).
        pos_tensor: Electrode positions, shape (n_channels, 3).
    """
    print("Loading REVE model from local path …")
    reve_model = AutoModel.from_pretrained(
        str(reve_model_path), trust_remote_code=True, torch_dtype="auto",
    )
    reve_model.eval()
    reve_model.to(device)
    for param in reve_model.parameters():
        param.requires_grad_(False)

    print("Loading REVE position bank from local path …")
    pos_bank = AutoModel.from_pretrained(
        str(reve_pos_path), trust_remote_code=True, torch_dtype="auto",
    )
    pos_bank.eval()
    pos_bank.to(device)

    with torch.no_grad():
        pos_tensor = pos_bank(channel_names)  # (n_channels, 3)
    print(f"  Electrode positions cached for {len(channel_names)} channels.")

    # Free position bank memory
    del pos_bank
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return reve_model, pos_tensor


def compute_n_patches(window_size: int, patch_size: int = 200, overlap: int = 20) -> int:
    """Compute number of time patches from REVE's unfold parameters."""
    step = patch_size - overlap
    return (window_size - patch_size) // step + 1


# ── ReveClassifierLP ────────────────────────────────

class ReveClassifierLP(nn.Module):
    """
    Official-faithful LP module: frozen REVE encoder + trainable cls_query_token
    + trainable linear head.

    Reproduces the exact architecture from reve_official/src/models/classifier.py
    (ReveClassifier) used during linear probing.

    During LP, the encoder is frozen but cls_query_token and linear_head are trainable.

    Pooling modes:
      - "no":       query attention → concat [context, patches] → flatten → RMSNorm → Dropout → Linear
      - "last":     query attention → squeeze → RMSNorm → Dropout → Linear
      - "last_avg": mean pool → RMSNorm → Dropout → Linear
    """

    def __init__(
        self,
        reve_model: nn.Module,
        pos_tensor: Tensor,
        n_classes: int,
        n_channels: int,
        window_size: int,
        pooling: str = "no",
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        assert pooling in ("no", "last", "last_avg"), f"Pooling '{pooling}' not supported"

        self.pooling = pooling
        self.reve = reve_model
        self.embed_dim = reve_model.config.embed_dim  # 512 for reve-base

        # Register electrode positions as buffer (not a parameter)
        self.register_buffer("pos_tensor", pos_tensor)  # (n_channels, 3)

        # Trainable query token — fresh random init
        self.cls_query_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.dropout = nn.Dropout(dropout)

        # Compute dimensions
        n_patches = compute_n_patches(window_size)
        n_tokens = n_channels * n_patches  # C * H

        if pooling == "no":
            out_shape = (1 + n_tokens) * self.embed_dim
            self.linear_head = nn.Sequential(
                RMSNorm(out_shape),
                self.dropout,
                nn.Linear(out_shape, n_classes),
            )
            self._out_shape = out_shape
        else:
            self.linear_head = nn.Sequential(
                RMSNorm(self.embed_dim),
                self.dropout,
                nn.Linear(self.embed_dim, n_classes),
            )
            self._out_shape = self.embed_dim

        self.n_tokens = n_tokens
        self.n_channels = n_channels
        self.n_patches = n_patches

    def forward(self, eeg: Tensor, pos: Tensor | None = None) -> Tensor:
        """
        Args:
            eeg: (B, C, T) raw EEG.
            pos: (B, C, 3) electrode positions. If None, uses registered buffer.

        Returns:
            logits: (B, n_classes)
        """
        B = eeg.shape[0]

        if pos is None:
            pos = self.pos_tensor.unsqueeze(0).expand(B, -1, -1)

        # Frozen encoder forward pass
        with torch.no_grad():
            # HF model returns (B, C, H, E) when return_output=False
            out_4d = self.reve(eeg, pos, return_output=False)

        # Rearrange to (B, C*H, E) = (B, n_tokens, embed_dim)
        x = rearrange(out_4d, "b c h e -> b (c h) e")

        if self.pooling == "last_avg":
            x = x.mean(dim=1)  # (B, E)
            return self.linear_head(x)

        elif self.pooling == "no":
            # Query attention
            query = self.cls_query_token.expand(B, -1, -1)  # (B, 1, E)
            attn_scores = torch.matmul(query, x.transpose(-1, -2)) / (self.embed_dim ** 0.5)
            attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, 1, n_tokens)
            context = torch.matmul(attn_weights, x)  # (B, 1, E)

            # Concat [context, all_patches] then flatten
            x = torch.cat([context, x], dim=1)  # (B, 1+n_tokens, E)
            x = x.reshape(B, -1)  # (B, (1+n_tokens)*E)
            return self.linear_head(x)

        else:  # "last"
            query = self.cls_query_token.expand(B, -1, -1)
            attn_scores = torch.matmul(query, x.transpose(-1, -2)) / (self.embed_dim ** 0.5)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            context = torch.matmul(attn_weights, x).squeeze(1)  # (B, E)
            return self.linear_head(context)

    def trainable_parameters(self):
        """Yield only trainable parameters (cls_query_token + linear_head)."""
        yield self.cls_query_token
        yield from self.linear_head.parameters()

    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())


# ── Evaluation helper ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    n_classes: int = 9,
    use_amp: bool = True,
) -> dict:
    """Evaluate model on a dataloader, returning official REVE metrics.

    Returns dict with keys: accuracy, balanced_acc, f1_weighted,
    auroc (binary only), auc_pr (binary only).
    """
    model.eval()
    device_type = "cuda" if "cuda" in device else "cpu"
    binary = n_classes == 2

    all_preds, all_targets, all_probs = [], [], []
    score, count = 0, 0
    total_loss = 0.0

    for batch in loader:
        eeg, target = batch[0].to(device), batch[1].long().to(device)

        with torch.autocast(device_type=device_type, enabled=use_amp, dtype=torch.float16):
            with torch.inference_mode():
                output = model(eeg)

        loss = F.cross_entropy(output.float(), target)
        total_loss += loss.item() * target.shape[0]

        decisions = torch.argmax(output, dim=1)
        score += (decisions == target).int().sum().item()
        count += target.shape[0]

        all_preds.append(decisions.cpu())
        all_targets.append(target.cpu())
        all_probs.append(output.float().cpu())

    gt = torch.cat(all_targets).numpy()
    pr = torch.cat(all_preds).numpy()
    pr_probs = torch.cat(all_probs).numpy()

    acc = score / count
    avg_loss = total_loss / max(count, 1)
    bal_acc = balanced_accuracy_score(gt, pr)
    f1_w = f1_score(gt, pr, average="weighted")

    metrics = {
        "accuracy": acc,
        "val_loss": avg_loss,
        "balanced_acc": bal_acc,
        "f1_weighted": f1_w,
        "auroc": 0.0,
        "auc_pr": 0.0,
    }

    try:
        probs_softmax = torch.softmax(torch.from_numpy(pr_probs), dim=-1).numpy()
        if binary:
            metrics["auroc"] = roc_auc_score(gt, probs_softmax[:, 1])
            metrics["auc_pr"] = average_precision_score(gt, probs_softmax[:, 1])
        else:
            metrics["auroc"] = roc_auc_score(gt, probs_softmax, multi_class="ovr", average="macro")
    except ValueError:
        pass  # too few classes in fold (e.g. single-class val split)

    return metrics


# ── EmbeddingExtractor (fast mode) ──────────────────────────────────────────

class EmbeddingExtractor:
    """
    Pre-computation utility: runs the frozen REVE encoder once over an entire
    dataset and saves the resulting embeddings to a .pt file.
    """

    def __init__(
        self,
        reve_model: nn.Module,
        pos_tensor: Tensor,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.reve = reve_model
        self._pos_1d = pos_tensor  # (n_channels, 3)

    @torch.no_grad()
    def extract_embeddings(
        self,
        dataset: Dataset,
        batch_size: int = 64,
        use_pooling: bool = True,
        no_pool_mode: str = "mean",
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Pass every window through the frozen REVE encoder.

        Returns:
            embeddings: (N, D) float32 on CPU.
            labels:     (N,) int64 on CPU.
            stimulus_indices: (N,) int64 on CPU.
        """
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        all_embeddings: list[Tensor] = []
        all_labels: list[Tensor] = []

        print(f"Extracting REVE embeddings for {len(dataset):,} windows …")
        for eeg_batch, label_batch in tqdm(loader, unit="batch"):
            B = eeg_batch.shape[0]
            eeg_batch = eeg_batch.to(self.device)
            pos = self._pos_1d.unsqueeze(0).expand(B, -1, -1)

            out_4d: Tensor = self.reve(eeg_batch, pos)  # (B, C, H, 512)

            if use_pooling:
                emb = self.reve.attention_pooling(out_4d)   # (B, 512)
            elif no_pool_mode == "mean":
                emb = out_4d.mean(dim=1).reshape(B, -1)     # (B, H*512)
            else:  # "flat"
                emb = out_4d.reshape(B, -1)                  # (B, C*H*512)

            all_embeddings.append(emb.cpu())
            all_labels.append(label_batch.long())

        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Recover stimulus index from dataset's flat index
        stimulus_indices = torch.tensor(
            [dataset.index[i][1] for i in range(len(dataset))],
            dtype=torch.long,
        )

        print(f"  Done. Embeddings shape: {embeddings.shape}")
        return embeddings, labels, stimulus_indices

    @staticmethod
    def save_embeddings(
        embeddings: Tensor,
        labels: Tensor,
        save_path: Path,
        stimulus_indices: Tensor | None = None,
    ) -> None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Tensor] = {"embeddings": embeddings, "labels": labels}
        if stimulus_indices is not None:
            payload["stimulus_indices"] = stimulus_indices
        torch.save(payload, save_path)
        print(f"  Saved {embeddings.shape[0]:,} embeddings → {save_path}")


# ── LinearProber (fast mode, Lightning) ─────────────────────────────────────

import lightning as L
import torchmetrics


class LinearProber(L.LightningModule):
    """
    Classification head trained on pre-computed REVE embeddings (fast mode).

    Architecture: RMSNorm → Dropout → Linear
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 512,
        lr: float = 5e-3,
        dropout: float = 0.05,
        warmup_epochs: int = 3,
        scheduler_patience: int = 6,
        normalize_features: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.scheduler_patience = scheduler_patience
        self.normalize_features = normalize_features

        self.classifier = nn.Sequential(
            RMSNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        task = "binary" if num_classes == 2 else "multiclass"
        metric_kwargs = dict(task=task, num_classes=num_classes, average="macro")

        self.train_acc = torchmetrics.Accuracy(**metric_kwargs)
        self.val_acc   = torchmetrics.Accuracy(**metric_kwargs)

        auroc_kwargs = dict(task=task, num_classes=num_classes, average="macro")
        self.train_auroc = torchmetrics.AUROC(**auroc_kwargs)
        self.val_auroc   = torchmetrics.AUROC(**auroc_kwargs)

        self.train_f1 = torchmetrics.F1Score(**metric_kwargs)
        self.val_f1   = torchmetrics.F1Score(**metric_kwargs)

    def forward(self, x: Tensor) -> Tensor:
        if self.normalize_features:
            x = F.normalize(x, dim=-1)
        return self.classifier(x)

    def _shared_step(self, batch: Tuple[Tensor, Tensor], prefix: str) -> Tensor:
        embeddings, labels = batch
        labels = labels.long()

        logits = self(embeddings)
        loss = F.cross_entropy(logits, labels)

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        auroc_preds = probs[:, 1] if self.num_classes == 2 else probs

        acc   = getattr(self, f"{prefix}_acc")
        auroc = getattr(self, f"{prefix}_auroc")
        f1    = getattr(self, f"{prefix}_f1")

        acc(preds, labels)
        auroc(auroc_preds, labels)
        f1(preds, labels)

        self.log(f"{prefix}/loss",  loss,        on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/acc",   acc,          on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/auroc", auroc,        on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{prefix}/f1",    f1,           on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, prefix="train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        self._shared_step(batch, prefix="val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=self.scheduler_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/acc",
                "interval": "epoch",
            },
        }
