"""
Loss functions and evaluation metrics (Precision, Recall, mAP50, mAP50-95)
for Anchor-Free YOLO Object Detection in Cotton Leaf AI.
"""

import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_ciou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes Complete IoU (CIoU) between aligned box pairs [x1, y1, x2, y2].
    Returns tensor of shape (N,) in [-1.0, 1.0].
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.unbind(dim=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.unbind(dim=-1)

    inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(min=0)
    inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(min=0)
    inter_area = inter_w * inter_h

    w1 = (b1_x2 - b1_x1).clamp(min=eps)
    h1 = (b1_y2 - b1_y1).clamp(min=eps)
    w2 = (b2_x2 - b2_x1).clamp(min=eps)
    h2 = (b2_y2 - b2_y1).clamp(min=eps)
    union = w1 * h1 + w2 * h2 - inter_area + eps
    iou = inter_area / union

    # Enclosing box diagonal squared
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps

    # Center distance squared
    cx1 = (b1_x1 + b1_x2) * 0.5
    cy1 = (b1_y1 + b1_y2) * 0.5
    cx2 = (b2_x1 + b2_x2) * 0.5
    cy2 = (b2_y1 + b2_y2) * 0.5
    rho2 = (cx1 - cx2).pow(2) + (cy1 - cy2).pow(2)

    # Aspect ratio consistency
    v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    return iou - (rho2 / c2 + alpha * v)


class YOLODetectionLoss(nn.Module):
    """
    Anchor-free center-region target assignment loss:
      L = lambda_box * L_CIoU + L_obj + L_cls
    """

    def __init__(self, num_classes: int = 6, box_weight: float = 5.0):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(
        self,
        scale_outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
        targets: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        scale_outputs: [(ltrb (B,4,H,W), obj (B,1,H,W), cls (B,C,H,W), stride), ...]
        targets: list length B with {'boxes': (M,4) pixel xyxy, 'labels': (M,) long}
        """
        device = scale_outputs[0][0].device
        batch_size = scale_outputs[0][0].shape[0]

        total_box_loss = torch.tensor(0.0, device=device)
        total_obj_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        num_pos = 0

        for ltrb, obj_logit, cls_logits, stride in scale_outputs:
            _, _, gh, gw = ltrb.shape
            yv, xv = torch.meshgrid(
                torch.arange(gh, device=device, dtype=ltrb.dtype),
                torch.arange(gw, device=device, dtype=ltrb.dtype),
                indexing="ij",
            )
            cx = (xv + 0.5) * stride
            cy = (yv + 0.5) * stride

            obj_target = torch.zeros_like(obj_logit)

            for b in range(batch_size):
                gt_boxes = targets[b]["boxes"].to(device)
                gt_labels = targets[b]["labels"].to(device)
                if gt_boxes.numel() == 0:
                    continue

                # Scale suitability check (P3 small, P4 medium, P5 large)
                widths = gt_boxes[:, 2] - gt_boxes[:, 0]
                heights = gt_boxes[:, 3] - gt_boxes[:, 1]
                max_side = torch.maximum(widths, heights)

                for m in range(gt_boxes.shape[0]):
                    x1, y1, x2, y2 = gt_boxes[m]
                    if stride == 8 and max_side[m] > 128:
                        continue
                    if stride == 16 and (max_side[m] <= 64 or max_side[m] > 280):
                        continue
                    if stride == 32 and max_side[m] <= 192:
                        continue

                    gx = (x1 + x2) * 0.5
                    gy = (y1 + y2) * 0.5
                    radius = 1.5 * stride
                    in_center = (
                        (cx >= max(x1, gx - radius))
                        & (cx <= min(x2, gx + radius))
                        & (cy >= max(y1, gy - radius))
                        & (cy <= min(y2, gy + radius))
                    )
                    pos_y, pos_x = torch.where(in_center)
                    if pos_y.numel() == 0:
                        # Fallback to closest center grid cell
                        iy = int(torch.clamp(gy / stride, 0, gh - 1).item())
                        ix = int(torch.clamp(gx / stride, 0, gw - 1).item())
                        pos_y = torch.tensor([iy], device=device)
                        pos_x = torch.tensor([ix], device=device)

                    obj_target[b, 0, pos_y, pos_x] = 1.0

                    pred_dist = ltrb[b, :, pos_y, pos_x].permute(1, 0) * stride
                    px = cx[pos_y, pos_x]
                    py = cy[pos_y, pos_x]
                    pred_boxes = torch.stack(
                        [px - pred_dist[:, 0], py - pred_dist[:, 1], px + pred_dist[:, 2], py + pred_dist[:, 3]],
                        dim=-1,
                    )
                    gt_rep = gt_boxes[m].unsqueeze(0).expand_as(pred_boxes)
                    ciou = compute_ciou(pred_boxes, gt_rep)
                    total_box_loss = total_box_loss + (1.0 - ciou).sum()

                    cls_target = torch.zeros((pos_y.numel(), self.num_classes), device=device)
                    cls_target[:, int(gt_labels[m].item())] = 1.0
                    pred_cls = cls_logits[b, :, pos_y, pos_x].permute(1, 0)
                    total_cls_loss = total_cls_loss + self.bce(pred_cls, cls_target)

                    num_pos += int(pos_y.numel())

            total_obj_loss = total_obj_loss + self.bce(obj_logit, obj_target)

        denom = float(max(1, num_pos))
        return (self.box_weight * total_box_loss + total_obj_loss + total_cls_loss) / denom


def _box_iou_np(box_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Computes IoU between one [4] box and [M, 4] boxes in xyxy coordinates."""
    if boxes_b.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box_a[0], boxes_b[:, 0])
    y1 = np.maximum(box_a[1], boxes_b[:, 1])
    x2 = np.minimum(box_a[2], boxes_b[:, 2])
    y2 = np.minimum(box_a[3], boxes_b[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, box_a[2] - box_a[0]) * np.maximum(0.0, box_a[3] - box_a[1])
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-7)


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Computes exact area under the monotonically interpolated precision-recall envelope."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_detection_metrics(
    predictions_per_image: List[List[Dict[str, Any]]],
    ground_truths_per_image: List[Dict[str, Any]],
    num_classes: int = 6,
) -> Dict[str, float]:
    """
    Computes Precision, Recall, mAP@0.50, and mAP@0.50:0.95 across images.
    """
    iou_thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]
    aps_by_thresh: Dict[float, List[float]] = {t: [] for t in iou_thresholds}
    per_class_metrics: Dict[int, Dict[str, float]] = {}

    tp_50_total = 0
    fp_50_total = 0
    gt_total = 0

    for c in range(num_classes):
        class_preds = []
        class_gts: Dict[int, np.ndarray] = {}
        total_class_gt = 0

        for img_idx, (preds, gt) in enumerate(zip(predictions_per_image, ground_truths_per_image)):
            gt_boxes = np.asarray(gt.get("boxes", []), dtype=np.float32)
            gt_labels = np.asarray(gt.get("labels", []), dtype=np.int64)
            mask = gt_labels == c if gt_labels.size > 0 else np.zeros((0,), dtype=bool)
            c_boxes = gt_boxes[mask] if gt_boxes.size > 0 else np.zeros((0, 4), dtype=np.float32)
            class_gts[img_idx] = c_boxes
            total_class_gt += int(c_boxes.shape[0])

            for p in preds:
                if int(p["class_id"]) == c:
                    class_preds.append((float(p["confidence"]), img_idx, np.asarray(p["bbox"], dtype=np.float32)))

        if total_class_gt == 0:
            continue

        gt_total += total_class_gt
        class_preds.sort(key=lambda item: item[0], reverse=True)
        class_threshold_aps: List[float] = []
        c_ap50: float = 0.0

        for thresh in iou_thresholds:
            matched = {img_idx: np.zeros(len(boxes), dtype=bool) for img_idx, boxes in class_gts.items()}
            tp = np.zeros(len(class_preds), dtype=np.float32)
            fp = np.zeros(len(class_preds), dtype=np.float32)

            for i, (_, img_idx, pred_box) in enumerate(class_preds):
                gt_b = class_gts[img_idx]
                if gt_b.shape[0] == 0:
                    fp[i] = 1.0
                    continue
                ious = _box_iou_np(pred_box, gt_b)
                best_j = int(np.argmax(ious))
                if ious[best_j] >= thresh and not matched[img_idx][best_j]:
                    matched[img_idx][best_j] = True
                    tp[i] = 1.0
                else:
                    fp[i] = 1.0

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            rec = tp_cum / max(1, total_class_gt)
            prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-7)
            ap_val = _compute_ap(rec, prec)
            aps_by_thresh[thresh].append(ap_val)
            class_threshold_aps.append(ap_val)

            if thresh == 0.50:
                tp_50_total += int(tp.sum())
                fp_50_total += int(fp.sum())
                c_ap50 = ap_val

        per_class_metrics[c] = {
            "ap50": round(float(c_ap50), 4),
            "ap50_95": round(float(np.mean(class_threshold_aps)), 4) if class_threshold_aps else 0.0,
            "gt_boxes": total_class_gt,
        }

    map50 = float(np.mean(aps_by_thresh[0.50])) if aps_by_thresh[0.50] else 0.0
    all_maps = [float(np.mean(aps_by_thresh[t])) for t in iou_thresholds if aps_by_thresh[t]]
    map50_95 = float(np.mean(all_maps)) if all_maps else 0.0
    precision = float(tp_50_total / max(1, tp_50_total + fp_50_total)) if (tp_50_total + fp_50_total) > 0 else 0.0
    recall = float(tp_50_total / max(1, gt_total)) if gt_total > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "evaluated_gt_boxes": gt_total,
        "per_class": per_class_metrics,
    }
