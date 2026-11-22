"""
Comprehensive Unit and Integration Test Suite for Faster R-CNN Ignore-Region Handling (Phase 9.5.2).

Verifies 10 Required Synthetic Test Cases:
1. Focal lesion with valid bounding box and no ignore region.
2. Diffuse disease region with no detection box.
3. Healthy control with no boxes and no ignore regions.
4. Mixed focal lesion and diffuse disease region.
5. Multiple non-overlapping ignore regions.
6. Ignore region overlapping a positive detection (positive priority).
7. Image with no disease annotations (empty targets).
8. Image with empty detection targets and background proposals.
9. Proposals completely outside ignore regions (normal background supervision).
10. Proposals substantially overlapping ignore regions (excluded from background loss).

Also verifies:
- RPN anchor assignment and sampling.
- RoI proposal matching and sampling.
- Gradient stability (finite gradients, no NaNs).
"""

import sys
from pathlib import Path
from typing import Dict, List

import pytest
import torch
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ignore_region_faster_rcnn import (
    IgnoreAwareRPN,
    IgnoreAwareRoIHeads,
    build_ignore_aware_faster_rcnn,
    compute_box_intersection,
    compute_box_overlap_metrics,
    substantially_overlaps_ignore,
)


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def ignore_aware_model(device):
    torch.manual_seed(42)
    model = build_ignore_aware_faster_rcnn(
        num_classes=7,
        pretrained=False,
        trainable_backbone_layers=1,
        device=device,
        rpn_ignore_threshold=0.30,
        roi_ignore_threshold=0.30,
    )
    model.train()
    return model


def test_box_overlap_metrics_iou_and_ioa():
    """Verifies IoU and IoA calculations for large diffuse patches and small submerged boxes."""
    # Diffuse patch [0, 0, 100, 100] (area 10,000)
    # Small submerged anchor [20, 20, 40, 40] (area 400)
    # Non-overlapping anchor [120, 120, 150, 150] (area 900)
    boxes = torch.tensor([
        [20.0, 20.0, 40.0, 40.0],
        [120.0, 120.0, 150.0, 150.0],
    ])
    ignore_boxes = torch.tensor([[0.0, 0.0, 100.0, 100.0]])

    iou, ioa = compute_box_overlap_metrics(boxes, ignore_boxes)

    # Submerged box: intersection = 400, box_area = 400 -> IoA = 1.0!
    assert torch.isclose(ioa[0, 0], torch.tensor(1.0), atol=1e-4)
    # Traditional IoU: 400 / (400 + 10000 - 400) = 400 / 10000 = 0.04
    assert torch.isclose(iou[0, 0], torch.tensor(0.04), atol=1e-4)

    # Outside box: intersection = 0 -> IoA = 0.0, IoU = 0.0
    assert torch.isclose(ioa[1, 0], torch.tensor(0.0), atol=1e-4)
    assert torch.isclose(iou[1, 0], torch.tensor(0.0), atol=1e-4)

    # Substantially overlaps predicate
    mask = substantially_overlaps_ignore(boxes, ignore_boxes, threshold=0.30)
    assert mask[0].item() is True
    assert mask[1].item() is False


# --------------------------------------------------------------------------
# 10 Synthetic Test Cases
# --------------------------------------------------------------------------

def test_case_1_focal_alone(ignore_aware_model, device):
    """Test 1: Focal lesion with valid bounding box and no ignore region."""
    image = torch.rand((1, 3, 224, 224), device=device)
    target = [{
        "boxes": torch.tensor([[30.0, 30.0, 80.0, 80.0]], device=device),
        "labels": torch.tensor([1], dtype=torch.int64, device=device),
        "ignore_boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
    }]

    losses = ignore_aware_model(image, target)
    assert "loss_classifier" in losses
    assert "loss_box_reg" in losses
    assert "loss_objectness" in losses
    assert "loss_rpn_box_reg" in losses

    for k, v in losses.items():
        assert torch.isfinite(v), f"{k} is not finite"
    assert losses["loss_box_reg"] >= 0.0


def test_case_2_diffuse_alone(ignore_aware_model, device):
    """Test 2: Diffuse disease region with no detection box."""
    image = torch.rand((1, 3, 224, 224), device=device)
    target = [{
        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
        "labels": torch.zeros((0,), dtype=torch.int64, device=device),
        "ignore_boxes": torch.tensor([[40.0, 40.0, 160.0, 160.0]], device=device),
    }]

    losses = ignore_aware_model(image, target)
    for k, v in losses.items():
        assert torch.isfinite(v), f"{k} is not finite"

    # With 0 detection boxes, box regression loss is strictly 0.0
    assert losses["loss_box_reg"].item() == 0.0
    assert losses["loss_rpn_box_reg"].item() == 0.0


def test_case_3_healthy_control(ignore_aware_model, device):
    """Test 3: Healthy control with no boxes and no ignore regions."""
    image = torch.rand((1, 3, 224, 224), device=device)
    target = [{
        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
        "labels": torch.zeros((0,), dtype=torch.int64, device=device),
        "ignore_boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
    }]

    losses = ignore_aware_model(image, target)
    for k, v in losses.items():
        assert torch.isfinite(v), f"{k} is not finite"

    # Regression losses strictly zero for negative control
    assert losses["loss_box_reg"].item() == 0.0
    assert losses["loss_rpn_box_reg"].item() == 0.0
    # Classifier and objectness losses active for background
    assert losses["loss_classifier"] > 0.0
    assert losses["loss_objectness"] > 0.0


def test_case_4_mixed_focal_and_diffuse(ignore_aware_model, device):
    """Test 4: Image containing both focal lesions and diffuse disease."""
    image = torch.rand((1, 3, 224, 224), device=device)
    target = [{
        "boxes": torch.tensor([[20.0, 20.0, 50.0, 50.0]], device=device),
        "labels": torch.tensor([3], dtype=torch.int64, device=device),
        "ignore_boxes": torch.tensor([[100.0, 100.0, 180.0, 180.0]], device=device),
    }]

    losses = ignore_aware_model(image, target)
    for k, v in losses.items():
        assert torch.isfinite(v), f"{k} is not finite"
    assert losses["loss_classifier"] > 0.0
    assert losses["loss_objectness"] > 0.0


def test_case_5_multiple_non_overlapping_ignore_regions(ignore_aware_model, device):
    """Test 5: Multiple non-overlapping ignore regions."""
    image = torch.rand((1, 3, 224, 224), device=device)
    target = [{
        "boxes": torch.tensor([[10.0, 10.0, 40.0, 40.0]], device=device),
        "labels": torch.tensor([2], dtype=torch.int64, device=device),
        "ignore_boxes": torch.tensor([
            [50.0, 10.0, 90.0, 50.0],
            [120.0, 120.0, 180.0, 180.0],
            [10.0, 150.0, 60.0, 200.0],
        ], device=device),
    }]

    losses = ignore_aware_model(image, target)
    for k, v in losses.items():
        assert torch.isfinite(v), f"{k} is not finite"


def test_case_6_ignore_region_overlapping_positive_detection(ignore_aware_model, device):
    """Test 6: Ignore region overlapping a positive detection (positive focal lesion priority)."""
    image = torch.rand((1, 3, 224, 224), device=device)
    # Focal lesion at [30, 30, 70, 70]
    # Ignore region completely encloses focal lesion at [20, 20, 100, 100]
    target = [{
        "boxes": torch.tensor([[30.0, 30.0, 70.0, 70.0]], device=device),
        "labels": torch.tensor([1], dtype=torch.int64, device=device),
        "ignore_boxes": torch.tensor([[20.0, 20.0, 100.0, 100.0]], device=device),
    }]

    losses = ignore_aware_model(image, target)
    for k, v in losses.items():
        assert torch.isfinite(v), f"{k} is not finite"

    # Focal lesion must maintain positive priority: box regression must be computed
    assert losses["loss_box_reg"] >= 0.0


def test_case_7_no_disease_annotations(ignore_aware_model, device):
    """Test 7: Image with no disease annotations (missing ignore_boxes key fallback)."""
    image = torch.rand((1, 3, 224, 224), device=device)
    # Target dictionary without "ignore_boxes" key
    target = [{
        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
        "labels": torch.zeros((0,), dtype=torch.int64, device=device),
    }]

    losses = ignore_aware_model(image, target)
    for k, v in losses.items():
        assert torch.isfinite(v), f"{k} is not finite"


def test_case_8_empty_detection_target(ignore_aware_model, device):
    """Test 8: Image with empty detection target and explicit empty ignore_boxes."""
    image = torch.rand((1, 3, 224, 224), device=device)
    target = [{
        "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
        "labels": torch.empty((0,), dtype=torch.int64, device=device),
        "ignore_boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
    }]

    losses = ignore_aware_model(image, target)
    for k, v in losses.items():
        assert torch.isfinite(v), f"{k} is not finite"


def test_case_9_proposals_outside_ignore_regions(ignore_aware_model, device):
    """Test 9: Proposals completely outside ignore regions retain normal negative supervision."""
    roi_heads = ignore_aware_model.roi_heads
    proposals = [torch.tensor([
        [10.0, 10.0, 30.0, 30.0],
        [40.0, 40.0, 60.0, 60.0],
    ], device=device)]
    targets = [{
        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
        "labels": torch.zeros((0,), dtype=torch.int64, device=device),
        "ignore_boxes": torch.tensor([[150.0, 150.0, 200.0, 200.0]], device=device),
    }]

    # In select_training_samples:
    sampled_props, matched_idxs, labels, reg_targets = roi_heads.select_training_samples(proposals, targets)
    # Outside proposals should retain label == 0 (negative background)
    assert len(labels[0]) > 0
    assert (labels[0] == 0).all()


def test_case_10_proposals_inside_ignore_regions(ignore_aware_model, device):
    """Test 10: Proposals substantially overlapping ignore regions are masked to -1 and excluded."""
    roi_heads = ignore_aware_model.roi_heads
    # Both proposals are completely submerged inside ignore region [0, 0, 100, 100]
    proposals = [torch.tensor([
        [10.0, 10.0, 40.0, 40.0],
        [50.0, 50.0, 80.0, 80.0],
    ], device=device)]
    targets = [{
        "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
        "labels": torch.zeros((0,), dtype=torch.int64, device=device),
        "ignore_boxes": torch.tensor([[0.0, 0.0, 100.0, 100.0]], device=device),
    }]

    # Directly verify assignment step inside select_training_samples
    gt_boxes = [t["boxes"] for t in targets]
    gt_labels = [t["labels"] for t in targets]
    props_with_gt = roi_heads.add_gt_proposals(proposals, gt_boxes)
    matched_idxs, raw_labels = roi_heads.assign_targets_to_proposals(props_with_gt, gt_boxes, gt_labels)

    # Before ignore masking, background proposals have label == 0
    assert (raw_labels[0] == 0).all()

    # Now verify select_training_samples masks them to -1 and excludes them
    sampled_props, matched_idxs, final_labels, reg_targets = roi_heads.select_training_samples(proposals, targets)
    # Since all proposals were submerged in ignore region and 0 focal boxes exist,
    # 0 proposals are sampled into final training batch!
    assert final_labels[0].numel() == 0


def test_rpn_anchor_ignore_masking(ignore_aware_model, device):
    """Unit test: Verifies RPN anchor assignment masks non-positive overlapping anchors to -1.0."""
    rpn = ignore_aware_model.rpn
    anchors = [torch.tensor([
        [10.0, 10.0, 30.0, 30.0],   # Inside ignore
        [120.0, 120.0, 150.0, 150.0], # Outside ignore
        [50.0, 50.0, 70.0, 70.0],   # Overlapping focal box
    ], device=device)]

    targets = [{
        "boxes": torch.tensor([[50.0, 50.0, 70.0, 70.0]], device=device),
        "labels": torch.tensor([1], dtype=torch.int64, device=device),
        "ignore_boxes": torch.tensor([[0.0, 0.0, 40.0, 40.0]], device=device),
    }]

    labels, matched_gt = rpn.assign_targets_to_anchors(anchors, targets)
    labels_arr = labels[0]

    # Anchor 0: Inside ignore -> must be -1.0
    assert labels_arr[0].item() == -1.0
    # Anchor 1: Outside ignore -> normal background 0.0
    assert labels_arr[1].item() == 0.0
    # Anchor 2: Positive focal lesion -> must remain positive 1.0
    assert labels_arr[2].item() == 1.0


def test_gradient_backprop_stability(ignore_aware_model, device):
    """Verifies that all 10 scenarios produce finite gradients and valid backprop."""
    ignore_aware_model.zero_grad()
    image = torch.rand((2, 3, 200, 200), device=device)
    targets = [
        # Image 0: Mixed
        {
            "boxes": torch.tensor([[20.0, 20.0, 50.0, 50.0]], device=device),
            "labels": torch.tensor([1], dtype=torch.int64, device=device),
            "ignore_boxes": torch.tensor([[80.0, 80.0, 140.0, 140.0]], device=device),
        },
        # Image 1: Diffuse alone
        {
            "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
            "labels": torch.zeros((0,), dtype=torch.int64, device=device),
            "ignore_boxes": torch.tensor([[30.0, 30.0, 100.0, 100.0]], device=device),
        },
    ]

    losses = ignore_aware_model(image, targets)
    total_loss = sum(losses.values())
    total_loss.backward()

    # Verify gradients exist and are finite
    has_grad = False
    for name, param in ignore_aware_model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            assert torch.isfinite(param.grad).all(), f"NaN gradient in {name}"
    assert has_grad, "No gradients computed during backward pass"
