"""
training.py — Generic training loop for JADE two-stage pipeline.

Contains the shared train_stage_jade() function used by both LP warmup
(CE only) and LoRA/full fine-tuning (joint CE + SupCon) stages.
"""

from __future__ import annotations

import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from src.approaches.jade.loss import SupConLoss
from src.approaches.shared.metrics import evaluate_model
from src.approaches.shared.stable_adamw import StableAdamW
from src.approaches.shared.training_utils import COL_W, _PatienceMonitor, _get_exponential_warmup_lambda, fmt_dur


def train_stage_jade(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    stage_name: str,
    lr: float,
    max_epochs: int,
    warmup_epochs: int,
    scheduler_patience: int,
    early_stop_patience: int,
    grad_clip: float,
    weight_decay: float,
    use_mixup: bool,
    use_amp: bool,
    n_classes: int,
    device: str,
    wandb_epoch_offset: int = 0,
    save_trainable_only: bool = False,
    # SupCon-specific
    use_supcon: bool = False,
    supcon_alpha: float = 0.5,
    supcon_temperature: float = 0.07,
) -> dict:
    """
    Shared training loop for both LP warmup and FT stages.

    When use_supcon=True, computes joint loss:
        L = alpha * CE + (1 - alpha) * SupCon
    Mixup is disabled in this path (labels must be clean for SupCon).

    When use_supcon=False, behaves identically to FT's train_stage()
    (optional mixup, CE-only loss).

    Returns dict with best metrics, best state dict, and training stats.
    """
    device_type = "cuda" if "cuda" in device else "cpu"

    optimizer = StableAdamW(
        model.trainable_parameters(),
        lr=lr,
        betas=(0.92, 0.999),
        weight_decay=weight_decay,
    )

    warmup_steps = warmup_epochs * len(train_loader)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_get_exponential_warmup_lambda(warmup_steps),
    )

    reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=scheduler_patience,
    )

    scaler = torch.amp.GradScaler()
    patience_monitor = _PatienceMonitor(early_stop_patience)

    supcon_criterion = SupConLoss(temperature=supcon_temperature) if use_supcon else None

    best_metrics = {}
    best_acc = 0.0
    best_state = None

    print(f"\n{'─' * COL_W}")
    supcon_label = (
        f"  |  SupCon alpha={supcon_alpha} tau={supcon_temperature}" if use_supcon else ""
    )
    print(
        f"  Stage: {stage_name}  |  lr={lr}  |  max_epochs={max_epochs}  |  grad_clip={grad_clip}{supcon_label}"
    )
    header = (
        f"{'Epoch':>6}  {'EpTime':>7}  {'Elapsed':>8}  "
        f"{'TrLoss':>8}  {'TrAcc':>7}  {'VaAcc':>7}  {'VaBalAcc':>9}  "
        f"{'VaAUROC':>8}  {'VaF1w':>7}  {'LR':>10}"
    )
    print(header)
    print(f"{'─' * COL_W}")

    fit_start = time.time()
    last_epoch = 0

    for epoch in range(max_epochs):
        last_epoch = epoch
        epoch_start = time.time()
        warmup_active = epoch <= warmup_epochs

        # Train
        model.train()
        epoch_loss = 0.0
        epoch_loss_ce = 0.0
        epoch_loss_sc = 0.0
        n_batches = 0
        n_correct = 0
        n_samples = 0

        for batch in train_loader:
            eeg, target = batch[0].to(device), batch[1].long().to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, enabled=use_amp, dtype=torch.float16):
                if use_supcon:
                    logits, projections = model(eeg, return_projections=True)
                    loss_ce = F.cross_entropy(logits, target)
                    loss_sc = supcon_criterion(projections, target)
                    loss = supcon_alpha * loss_ce + (1 - supcon_alpha) * loss_sc
                elif use_mixup:
                    mm = random.random()
                    perm = torch.randperm(eeg.size(0), device=device)
                    output = model(mm * eeg + (1 - mm) * eeg[perm])
                    loss = mm * F.cross_entropy(output, target) + (1 - mm) * F.cross_entropy(
                        output, target[perm]
                    )
                else:
                    output = model(eeg)
                    loss = F.cross_entropy(output, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()

            if warmup_active and scale_before == scaler.get_scale():
                warmup_scheduler.step()

            epoch_loss += loss.item()
            if use_supcon:
                epoch_loss_ce += loss_ce.item()
                epoch_loss_sc += loss_sc.item()
            n_batches += 1

            with torch.no_grad():
                preds = (logits if use_supcon else output).argmax(dim=1)
            n_correct += (preds == target).sum().item()
            n_samples += target.size(0)

        avg_loss = epoch_loss / max(n_batches, 1)
        train_acc = n_correct / max(n_samples, 1)

        # Validate
        metrics = evaluate_model(
            model,
            val_loader,
            device=device,
            n_classes=n_classes,
            use_amp=use_amp,
        )

        val_acc = metrics["accuracy"]

        # Scheduler on val_acc, gated behind warmup (matches official)
        if epoch > warmup_epochs:
            reduce_scheduler.step(val_acc)

        # Track best by accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_metrics = {**metrics, "epoch": epoch + 1, "train_loss": avg_loss}
            if save_trainable_only:
                trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items() if k in trainable_keys
                }
            else:
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Print epoch summary
        ep_time = time.time() - epoch_start
        elapsed = time.time() - fit_start
        current_lr = optimizer.param_groups[0]["lr"]
        avg_ep = elapsed / (epoch + 1)
        remaining = avg_ep * (max_epochs - epoch - 1)
        eta = f"ETA {fmt_dur(remaining)}" if epoch + 1 < max_epochs else "done"

        print(
            f"{epoch + 1:>6}  {fmt_dur(ep_time):>7}  {fmt_dur(elapsed):>8}  "
            f"{avg_loss:>8.4f}  {train_acc:>7.4f}  {val_acc:>7.4f}  {metrics['balanced_acc']:>9.4f}  "
            f"{metrics['auroc']:>8.4f}  {metrics['f1_weighted']:>7.4f}  "
            f"{current_lr:>10.2e}  ({eta})"
        )

        # W&B per-epoch logging
        if wandb.run is not None:
            log_dict = {
                "train/loss": avg_loss,
                "train/acc": train_acc,
                "val/loss": metrics.get("val_loss"),
                "val/acc": val_acc,
                "val/bal_acc": metrics["balanced_acc"],
                "val/auroc": metrics["auroc"],
                "val/f1": metrics["f1_weighted"],
                "lr": current_lr,
                "stage": 0 if stage_name == "lp" else 1,
            }
            if use_supcon:
                avg_ce = epoch_loss_ce / max(n_batches, 1)
                avg_sc = epoch_loss_sc / max(n_batches, 1)
                log_dict["train/loss_ce"] = avg_ce
                log_dict["train/loss_sc"] = avg_sc
            wandb.log(log_dict, step=wandb_epoch_offset + epoch + 1)

        # Early stopping on accuracy
        if patience_monitor(val_acc):
            print(f"Early stopping at epoch {epoch + 1} (patience={early_stop_patience})")
            break

    total_time = time.time() - fit_start
    print(f"{'─' * COL_W}")
    print(
        f"{stage_name} complete — {last_epoch + 1} epochs | total: {fmt_dur(total_time)} | "
        f"best epoch={best_metrics.get('epoch', 'n/a')} val_acc={best_acc:.4f}"
    )
    print(f"{'─' * COL_W}")

    return {
        "val_acc": best_metrics.get("accuracy"),
        "val_bal_acc": best_metrics.get("balanced_acc"),
        "val_auroc": best_metrics.get("auroc"),
        "val_f1": best_metrics.get("f1_weighted"),
        "train_loss": best_metrics.get("train_loss"),
        "val_loss": best_metrics.get("val_loss"),
        "best_epoch": best_metrics.get("epoch"),
        "epochs_trained": last_epoch + 1,
        "best_state": best_state,
    }
