"""
Loss functions and evaluation metrics for Semantic Segmentation (U-Net).
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    """
    Multi-class Soft Dice Loss.
    """

    def __init__(self, smooth: float = 1e-5, ignore_background: bool = False):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C, H, W)
        targets: (B, H, W) with class values in [0, C-1]
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        start_idx = 1 if self.ignore_background else 0
        dice_losses = []

        for c in range(start_idx, num_classes):
            prob_c = probs[:, c, :, :]
            target_c = targets_one_hot[:, c, :, :]

            intersection = (prob_c * target_c).sum(dim=(1, 2))
            union = prob_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_losses.append(1.0 - dice.mean())

        return torch.stack(dice_losses).mean()


class CombinedDiceCELoss(nn.Module):
    """
    Weighted combination of CrossEntropyLoss and SoftDiceLoss.
    """

    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5, class_weights: torch.Tensor = None):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = SoftDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def compute_segmentation_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3,
    class_names: Tuple[str, ...] = ("Background", "Healthy Leaf", "Diseased Tissue")
) -> Dict[str, float]:
    """
    Computes per-class and mean Intersection over Union (mIoU) and Dice coefficients.
    preds: (B, H, W) or (H, W)
    targets: (B, H, W) or (H, W)
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    metrics = {}
    ious = []
    dices = []
    precisions = []
    recalls = []

    for c in range(num_classes):
        p_c = (preds == c)
        t_c = (targets == c)

        tp = (p_c & t_c).sum().float().item()
        fp = (p_c & ~t_c).sum().float().item()
        fn = (~p_c & t_c).sum().float().item()
        union = (p_c | t_c).sum().float().item()
        total = p_c.sum().float().item() + t_c.sum().float().item()

        iou = (tp + 1e-6) / (union + 1e-6) if union > 0 else 1.0
        dice = (2.0 * tp + 1e-6) / (total + 1e-6) if total > 0 else 1.0
        prec = (tp + 1e-6) / (tp + fp + 1e-6) if (tp + fp) > 0 else 1.0
        rec = (tp + 1e-6) / (tp + fn + 1e-6) if (tp + fn) > 0 else 1.0

        ious.append(iou)
        dices.append(dice)
        precisions.append(prec)
        recalls.append(rec)
        name = class_names[c] if c < len(class_names) else f"Class_{c}"
        metrics[f"iou_{name}"] = round(iou, 4)
        metrics[f"dice_{name}"] = round(dice, 4)
        metrics[f"precision_{name}"] = round(prec, 4)
        metrics[f"recall_{name}"] = round(rec, 4)

    pixel_accuracy = float((preds == targets).float().mean().item()) if targets.numel() > 0 else 0.0
    metrics["pixel_accuracy"] = round(pixel_accuracy, 4)
    metrics["mean_precision"] = round(float(sum(precisions) / len(precisions)), 4)
    metrics["mean_recall"] = round(float(sum(recalls) / len(recalls)), 4)
    metrics["mean_iou"] = round(float(sum(ious) / len(ious)), 4)
    metrics["mean_dice"] = round(float(sum(dices) / len(dices)), 4)
    return metrics
