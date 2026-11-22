"""
Ignore-Region Handling for Faster R-CNN (Phase 9.5.2).

Prevents Faster R-CNN's Region Proposal Network (RPN) and RoI box heads
from penalizing diffuse/morphological disease regions (Tier 2 symptoms such as
anthocyanin blush in Leaf Redding, marginal chlorosis in Jassids, and laminar curling)
as negative background anchors and proposals.

Biological Philosophy (Dual-Tier Annotation Standard V3):
- Tier 1 (Focal Lesions): Distinct, localized necrotic/chlorotic spots with sharp boundaries.
  Trained as positive detection targets with tight bounding boxes and class IDs 1..6.
- Tier 2 (Diffuse/Systemic Regions): Broad regional discoloration or morphological distortion.
  Segmented for U-Net pixel-level severity estimation, but treated as ignore regions for
  object detection to prevent false negative supervision and anchor training collapse.
- Golden Healthy Controls: Strictly 0 positive detection boxes and 0 ignore regions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN,
    FastRCNNPredictor,
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss


def compute_box_intersection(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise intersection area between boxes1 [N, 4] and boxes2 [M, 4].
    Format: [x1, y1, x2, y2].
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=boxes1.dtype, device=boxes1.device)
    
    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    return inter_w * inter_h


def compute_box_overlap_metrics(
    boxes: torch.Tensor, ignore_boxes: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Intersection-over-Union (IoU) and Intersection-over-Box-Area (IoA).
    
    IoA measures what fraction of the anchor/proposal is submerged inside the ignore region.
    This is mathematically essential because small anchors/proposals submerged inside large
    diffuse patches have very low traditional IoU, but an IoA of 1.0.

    Args:
        boxes: Tensor of shape [N, 4] representing candidate anchors or proposals.
        ignore_boxes: Tensor of shape [M, 4] representing Tier 2 ignore regions.

    Returns:
        (iou, ioa) tuple of Tensors of shape [N, M].
    """
    if boxes.numel() == 0 or ignore_boxes.numel() == 0:
        device = boxes.device
        return (
            torch.zeros((boxes.shape[0], ignore_boxes.shape[0]), device=device, dtype=boxes.dtype),
            torch.zeros((boxes.shape[0], ignore_boxes.shape[0]), device=device, dtype=boxes.dtype),
        )

    inter = compute_box_intersection(boxes, ignore_boxes)
    box_area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clamp(min=1e-6)
    ign_area = ((ignore_boxes[:, 2] - ignore_boxes[:, 0]) * (ignore_boxes[:, 3] - ignore_boxes[:, 1])).clamp(min=1e-6)
    union = (box_area[:, None] + ign_area[None, :] - inter).clamp(min=1e-6)

    iou = inter / union
    ioa = inter / box_area[:, None]
    return iou, ioa


def substantially_overlaps_ignore(
    boxes: torch.Tensor, ignore_boxes: torch.Tensor, threshold: float = 0.30
) -> torch.Tensor:
    """
    Determines whether each box in `boxes` substantially overlaps any ignore box.
    Returns a boolean mask of shape [N].
    
    A box is defined as substantially overlapping if:
    max_j(IoA(box_i, ignore_box_j)) >= threshold OR max_j(IoU(box_i, ignore_box_j)) >= threshold.
    """
    if boxes.numel() == 0 or ignore_boxes.numel() == 0:
        return torch.zeros((boxes.shape[0],), dtype=torch.bool, device=boxes.device)

    iou, ioa = compute_box_overlap_metrics(boxes, ignore_boxes)
    max_ioa = ioa.max(dim=1).values
    max_iou = iou.max(dim=1).values
    return (max_ioa >= threshold) | (max_iou >= threshold)


class IgnoreAwareRPN(RegionProposalNetwork):
    """
    Custom Region Proposal Network with Tier 2 Ignore-Region Masking.

    Key Invariants:
    1. Positive Anchor Preservation: Valid focal lesions (label == 1.0) remain positive.
    2. Negative Background Exclusion: Anchors overlapping Tier 2 ignore regions that are NOT
       positive focal lesions are assigned label = -1.0 (ignored by BalancedPositiveNegativeSampler).
    3. Normal Background Preservation: Anchors outside ignore regions retain label = 0.0
       and are sampled as normal negative background candidates.
    4. Loss Isolation: Ignored anchors are never sampled into sampled_pos_inds or sampled_neg_inds,
       guaranteeing zero contribution to RPN objectness or box regression losses.
    5. Gradient Stability: If an edge-case image has 0 sampled anchors, loss returns 0.0 with valid autograd.
    """

    def __init__(self, base_rpn: RegionProposalNetwork, ignore_threshold: float = 0.30):
        super().__init__(
            anchor_generator=base_rpn.anchor_generator,
            head=base_rpn.head,
            fg_iou_thresh=base_rpn.proposal_matcher.high_threshold,
            bg_iou_thresh=base_rpn.proposal_matcher.low_threshold,
            batch_size_per_image=base_rpn.fg_bg_sampler.batch_size_per_image,
            positive_fraction=base_rpn.fg_bg_sampler.positive_fraction,
            pre_nms_top_n=base_rpn._pre_nms_top_n,
            post_nms_top_n=base_rpn._post_nms_top_n,
            nms_thresh=base_rpn.nms_thresh,
            score_thresh=base_rpn.score_thresh,
        )
        self.rpn_ignore_threshold = ignore_threshold

    def assign_targets_to_anchors(
        self, anchors: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Assigns ground truth targets to anchors, masking anchors overlapping Tier 2 ignore regions.
        """
        labels, matched_gt_boxes = super().assign_targets_to_anchors(anchors, targets)

        for i, (anchors_i, target_i) in enumerate(zip(anchors, targets)):
            ignore_boxes = target_i.get("ignore_boxes", None)
            if ignore_boxes is None or ignore_boxes.numel() == 0:
                continue

            overlaps_ignore = substantially_overlaps_ignore(
                anchors_i, ignore_boxes, threshold=self.rpn_ignore_threshold
            )

            # Preserve valid positive anchor assignments (label == 1.0)
            is_positive = labels[i] == 1.0
            mask_to_ignore = overlaps_ignore & (~is_positive)

            # Mask overlapping non-positive anchors to -1.0 (ignored by sampler)
            labels[i][mask_to_ignore] = -1.0

        return labels, matched_gt_boxes

    def compute_loss(
        self,
        objectness: torch.Tensor,
        pred_bbox_deltas: torch.Tensor,
        labels: List[torch.Tensor],
        regression_targets: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes RPN objectness and box regression loss with zero-anchor safety guards.
        """
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        if sampled_inds.numel() == 0:
            # Safe finite zero loss maintaining gradient connectivity
            return objectness.sum() * 0.0, pred_bbox_deltas.sum() * 0.0

        return super().compute_loss(objectness, pred_bbox_deltas, labels, regression_targets)


class IgnoreAwareRoIHeads(RoIHeads):
    """
    Custom RoIHeads with Tier 2 Ignore-Region Masking.

    Key Invariants:
    1. Positive Focal Match Preservation: Proposals matching valid focal lesions (label > 0)
       are strictly preserved with their ground truth class IDs and regression targets.
    2. Negative Background Exclusion: Proposals overlapping Tier 2 ignore regions that are NOT
       focal lesion matches are assigned label = -1.
    3. Sampler Masking: BalancedPositiveNegativeSampler ignores proposals with label == -1.
       They are excluded from sampled_inds and never passed to FastRCNNPredictor during training.
    4. Loss Isolation: Excluded proposals contribute 0 to loss_classifier (no false background penalty)
       and 0 to loss_box_reg.
    5. Normal Background Preservation: Proposals outside ignore regions retain label = 0
       and provide normal negative background supervision.
    6. Gradient Stability: If an edge-case image has 0 sampled proposals, loss returns 0.0 with valid autograd.
    """

    def __init__(self, base_roi_heads: RoIHeads, ignore_threshold: float = 0.30):
        super().__init__(
            box_roi_pool=base_roi_heads.box_roi_pool,
            box_head=base_roi_heads.box_head,
            box_predictor=base_roi_heads.box_predictor,
            fg_iou_thresh=base_roi_heads.proposal_matcher.high_threshold,
            bg_iou_thresh=base_roi_heads.proposal_matcher.low_threshold,
            batch_size_per_image=base_roi_heads.fg_bg_sampler.batch_size_per_image,
            positive_fraction=base_roi_heads.fg_bg_sampler.positive_fraction,
            bbox_reg_weights=base_roi_heads.box_coder.weights,
            score_thresh=base_roi_heads.score_thresh,
            nms_thresh=base_roi_heads.nms_thresh,
            detections_per_img=base_roi_heads.detections_per_img,
            mask_roi_pool=base_roi_heads.mask_roi_pool,
            mask_head=base_roi_heads.mask_head,
            mask_predictor=base_roi_heads.mask_predictor,
            keypoint_roi_pool=base_roi_heads.keypoint_roi_pool,
            keypoint_head=base_roi_heads.keypoint_head,
            keypoint_predictor=base_roi_heads.keypoint_predictor,
        )
        self.roi_ignore_threshold = ignore_threshold

    def select_training_samples(
        self,
        proposals: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # Append ground truth bboxes to proposals
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # Base proposal matching
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)

        # Apply Tier 2 Ignore-Region Masking
        for i, (prop_i, target_i) in enumerate(zip(proposals, targets)):
            ignore_boxes = target_i.get("ignore_boxes", None)
            if ignore_boxes is None or ignore_boxes.numel() == 0:
                continue

            overlaps_ignore = substantially_overlaps_ignore(
                prop_i, ignore_boxes, threshold=self.roi_ignore_threshold
            )

            # Preserve valid focal matches: only mask non-positive proposals
            is_positive = labels[i] > 0
            mask_to_ignore = overlaps_ignore & (~is_positive)

            # Mask to -1 (ignored by BalancedPositiveNegativeSampler)
            labels[i][mask_to_ignore] = -1

        # Subsample proposals (samples label > 0 for pos and label == 0 for neg; label == -1 ignored)
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)

            # Edge-case guard: if 0 proposals sampled across entire batch
            total_sampled = sum(l.numel() for l in labels)
            if total_sampled == 0:
                # Run empty forward through head to maintain autograd graph
                box_features = self.box_roi_pool(features, proposals, image_shapes)
                box_features = self.box_head(box_features)
                class_logits, box_regression = self.box_predictor(box_features)
                return [], {
                    "loss_classifier": class_logits.sum() * 0.0,
                    "loss_box_reg": box_regression.sum() * 0.0,
                }

            box_features = self.box_roi_pool(features, proposals, image_shapes)
            box_features = self.box_head(box_features)
            class_logits, box_regression = self.box_predictor(box_features)

            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            return [], {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
            }

        return super().forward(features, proposals, image_shapes, targets)


def equip_faster_rcnn_with_ignore_regions(
    model: FasterRCNN,
    rpn_ignore_threshold: float = 0.30,
    roi_ignore_threshold: float = 0.30,
) -> FasterRCNN:
    """
    Equips an existing FasterRCNN model with IgnoreAwareRPN and IgnoreAwareRoIHeads.
    Preserves all weights, feature extractors, anchor generators, and box predictors.
    """
    model.rpn = IgnoreAwareRPN(model.rpn, ignore_threshold=rpn_ignore_threshold)
    model.roi_heads = IgnoreAwareRoIHeads(model.roi_heads, ignore_threshold=roi_ignore_threshold)
    return model


def build_ignore_aware_faster_rcnn(
    num_classes: int = 7,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    rpn_ignore_threshold: float = 0.30,
    roi_ignore_threshold: float = 0.30,
) -> FasterRCNN:
    """
    Constructs a Faster R-CNN ResNet-50 FPN V2 model equipped with Tier 2 ignore-region handling.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained and checkpoint_path is None else None

    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    # 7 classes (0=background, 1..6=cotton diseases)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Equip with Ignore-Region Modules
    model = equip_faster_rcnn_with_ignore_regions(
        model,
        rpn_ignore_threshold=rpn_ignore_threshold,
        roi_ignore_threshold=roi_ignore_threshold,
    )

    if checkpoint_path is not None:
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")
        state = torch.load(ckpt_p, map_location=device)
        state_dict = state.get("model_state_dict", state)
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state_dict)

    model.to(device)
    return model
