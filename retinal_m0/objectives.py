import torch
from torch.nn import functional as F


def pixel_loss(logits, target, epsilon=1e-6):
    if logits.shape != target.shape or logits.ndim != 4 or epsilon <= 0:
        raise ValueError("Expected matching NCHW tensors and positive Dice smoothing")
    # Both terms are explicitly FP32, including under the training autocast context.
    logits, target = logits.float(), target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    probability = logits.sigmoid()
    axes = (1, 2, 3)
    dice_loss = (1 - (2 * (probability * target).sum(axes) + epsilon) /
                 (probability.sum(axes) + target.sum(axes) + epsilon)).mean()
    return bce + dice_loss, {"bce": bce, "soft_dice_loss": dice_loss}


def binary_metrics(probability, target, threshold=0.5):
    prediction, truth = probability >= threshold, target > 0.5
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    return {"dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "tp": tp, "fp": fp, "fn": fn,
            "predicted_positive": tp + fp, "true_positive_pixels": tp + fn}
