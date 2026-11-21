"""Confusion-matrix based segmentation metrics shared by training and evaluation."""

from __future__ import annotations

import torch

from . import NUM_CLASSES


def update_confusion(confusion: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor) -> None:
    """Accumulate a (true class, predicted class) pixel-count matrix in place."""
    encoded = labels.reshape(-1) * NUM_CLASSES + predictions.reshape(-1)
    confusion += torch.bincount(encoded, minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_confusion(confusion: torch.Tensor) -> dict[str, object]:
    """Pixel accuracy, per-class IoU/Dice, and their means (classes absent from both GT and
    predictions are reported as None and excluded from the mean)."""
    confusion = confusion.double().cpu()
    true_positive = confusion.diag()
    predicted = confusion.sum(0)
    actual = confusion.sum(1)
    union = predicted + actual - true_positive
    denominator = predicted + actual
    iou = torch.where(union > 0, true_positive / union.clamp_min(1), torch.nan)
    dice = torch.where(denominator > 0, 2 * true_positive / denominator.clamp_min(1), torch.nan)
    return {
        "pixel_accuracy": float(true_positive.sum() / confusion.sum().clamp_min(1)),
        "miou": float(torch.nanmean(iou)),
        "mean_dice": float(torch.nanmean(dice)),
        "class_iou": [None if torch.isnan(v) else float(v) for v in iou],
        "class_dice": [None if torch.isnan(v) else float(v) for v in dice],
    }
