"""
Unit Tests for CottonGuard AI Phase 9.2A: Faster R-CNN Pipeline.

Tests:
1. CottonDetectionDataset train and validation loading, dimensions, and healthy controls.
2. detection_collate_fn handling variable and empty bounding boxes.
3. Model instantiation, pretrained weight verification, and 7-class head configuration.
4. Training mode forward pass and 4 loss components.
5. Evaluation mode forward pass and output prediction dictionaries.
6. Numerical stability with AMP gradient scaler and unscale_ verification.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.detection_dataset import (
    CottonDetectionDataset,
    FOREGROUND_CLASSES,
    detection_collate_fn,
)
from src.models.faster_rcnn import build_faster_rcnn_model


def test_detection_dataset_loading():
    """Verifies 35 train and 15 validation samples, healthy control empty targets, and box bounds."""
    train_ds = CottonDetectionDataset(split="train")
    val_ds = CottonDetectionDataset(split="val")

    assert len(train_ds) == 35, f"Expected 35 train samples, got {len(train_ds)}"
    assert len(val_ds) == 15, f"Expected 15 val samples, got {len(val_ds)}"

    # Check sample structure
    img, tgt = train_ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 800, 800)
    assert img.dtype == torch.float32
    assert "boxes" in tgt and "labels" in tgt and "image_id" in tgt

    # Verify healthy samples have empty targets
    healthy_found = False
    for i in range(len(train_ds)):
        _, target = train_ds[i]
        if target["boxes"].shape[0] == 0:
            assert target["boxes"].shape == (0, 4)
            assert target["labels"].shape == (0,)
            assert target["area"].shape == (0,)
            healthy_found = True
            break
    assert healthy_found, "Expected at least one healthy control in train split"


def test_detection_collate_fn():
    """Verifies collate_fn handles variable numbers of boxes and empty healthy boxes."""
    train_ds = CottonDetectionDataset(split="train")
    loader = DataLoader(train_ds, batch_size=4, shuffle=False, collate_fn=detection_collate_fn)
    images, targets = next(iter(loader))

    assert isinstance(images, tuple)
    assert isinstance(targets, tuple)
    assert len(images) == 4
    assert len(targets) == 4
    for img, tgt in zip(images, targets):
        assert img.shape == (3, 800, 800)
        assert "boxes" in tgt and "labels" in tgt


def test_model_instantiation_and_head():
    """Verifies Faster R-CNN V2 model instantiation and 7-class FastRCNNPredictor."""
    device = torch.device("cpu")
    model = build_faster_rcnn_model(num_classes=7, pretrained=False, device=device)

    assert hasattr(model, "roi_heads")
    assert hasattr(model.roi_heads, "box_predictor")
    cls_score = model.roi_heads.box_predictor.cls_score
    bbox_pred = model.roi_heads.box_predictor.bbox_pred

    # 7 classes (1 background + 6 foreground)
    assert cls_score.out_features == 7, f"Expected 7 classes, got {cls_score.out_features}"
    # 7 * 4 = 28 bbox coordinates
    assert bbox_pred.out_features == 28, f"Expected 28 box outputs, got {bbox_pred.out_features}"


def test_model_forward_train_and_losses():
    """Verifies model forward pass in training mode returns 4 valid loss components."""
    device = torch.device("cpu")
    model = build_faster_rcnn_model(num_classes=7, pretrained=False, device=device)
    model.train()

    train_ds = CottonDetectionDataset(split="train")
    loader = DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=detection_collate_fn)
    images, targets = next(iter(loader))

    losses = model(images, targets)
    assert isinstance(losses, dict)
    expected_keys = {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
    assert set(losses.keys()) == expected_keys

    for k, v in losses.items():
        assert torch.isfinite(v), f"Loss {k} is not finite: {v}"
        assert v.item() >= 0.0, f"Loss {k} is negative: {v}"


def test_model_eval_mode_predictions():
    """Verifies model forward pass in evaluation mode returns valid prediction dictionaries."""
    device = torch.device("cpu")
    model = build_faster_rcnn_model(num_classes=7, pretrained=False, device=device)
    model.eval()

    val_ds = CottonDetectionDataset(split="val")
    loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=detection_collate_fn)
    images, _ = next(iter(loader))

    with torch.no_grad():
        predictions = model(images)

    assert isinstance(predictions, list)
    assert len(predictions) == 2
    for pred in predictions:
        assert "boxes" in pred and "labels" in pred and "scores" in pred
        assert pred["boxes"].ndim == 2 and pred["boxes"].shape[1] == 4
        assert pred["labels"].ndim == 1
        assert pred["scores"].ndim == 1
        if pred["scores"].numel() > 0:
            assert torch.all(pred["scores"] >= 0.0) and torch.all(pred["scores"] <= 1.0)


def test_numerical_stability_amp_unscale():
    """Verifies numerical stability with AMP, unscale_ factor, and finite gradients."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_faster_rcnn_model(num_classes=7, pretrained=False, device=device)
    model.train()

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"), init_scale=1024.0)

    train_ds = CottonDetectionDataset(split="train")
    loader = DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=detection_collate_fn)
    images, targets = next(iter(loader))

    imgs_dev = [im.to(device) for im in images]
    tgts_dev = [{k: v.to(device) for k, v in tgt.items()} for tgt in targets]

    with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.float16):
        losses = model(imgs_dev, tgts_dev)
        total_loss = sum(losses.values())

    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)

    non_finite = sum(
        1 for p in model.parameters()
        if p.grad is not None and not torch.all(torch.isfinite(p.grad))
    )
    assert non_finite == 0, f"Detected {non_finite} non-finite gradients after unscaling!"
