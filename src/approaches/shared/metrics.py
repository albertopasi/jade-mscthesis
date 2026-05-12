"""
metrics.py — Shared evaluation metrics for REVE-based classifiers.

Used by both LP and FT pipelines.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    n_classes: int = 9,
    use_amp: bool = True,
    return_preds: bool = False,
) -> dict:
    """Evaluate model on a dataloader, returning official REVE metrics.

    Returns dict with keys: accuracy, balanced_acc, f1_weighted,
    auroc (binary only), auc_pr (binary only). When return_preds=True,
    also returns y_true and y_pred (lists of ints) for the entire loader.
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

    if return_preds:
        metrics["y_true"] = gt.astype(int).tolist()
        metrics["y_pred"] = pr.astype(int).tolist()

    return metrics
