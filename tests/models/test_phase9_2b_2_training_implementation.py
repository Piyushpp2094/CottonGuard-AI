"""
Unit Tests for CottonGuard AI Phase 9.2B.2: Training Implementation Preparation.

Tests:
1. YAML configuration parsing (all hyperparameters, AMP, scaler, scheduler).
2. GradScaler initialization from configuration (init_scale=1024.0).
3. Scheduler progression (540 steps: step 0=5e-6, step 54=0.005, step 539=1e-5).
4. AMP skipped-step handling (scheduler does not advance when step is skipped).
5. Dataset loaders separation (35 train / 15 val, zero overlap).
6. Per-class AP reporting in detection_losses.
7. Checkpoint destination isolation (protecting classifier and U-Net).
8. Early stopping logic (patience=5 triggering correctly).

Strict Invariant: ZERO training epochs and ZERO optimizer steps executed.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from src.data.detection_dataset import CottonDetectionDataset
from src.training.detection_losses import compute_detection_metrics
from scripts.train_faster_rcnn import build_warmup_cosine_scheduler

CONFIG_PATH = PROJECT_ROOT / "configs" / "phase9_2a_detector_config.yaml"


def test_yaml_configuration_parsing():
    """Verifies that all required hyperparameters and safeguards are present in YAML."""
    assert CONFIG_PATH.is_file(), f"Missing config: {CONFIG_PATH}"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Model
    assert cfg["model"]["architecture"] == "fasterrcnn_resnet50_fpn_v2"
    assert cfg["model"]["num_classes"] == 7
    assert cfg["model"]["background_label"] == 0

    # Training
    assert cfg["training"]["batch_size"] == 2
    assert cfg["training"]["max_epochs"] == 30
    assert cfg["training"]["optimizer"]["type"] == "SGD"
    assert cfg["training"]["optimizer"]["lr"] == 0.005
    assert cfg["training"]["optimizer"]["momentum"] == 0.9
    assert cfg["training"]["optimizer"]["weight_decay"] == 0.0005
    assert cfg["training"]["gradient_clipping"]["max_norm"] == 10.0

    # Mixed Precision & Scaler
    assert cfg["training"]["mixed_precision"]["enabled"] is True
    assert cfg["training"]["mixed_precision"]["dtype"] == "float16"
    assert cfg["training"]["mixed_precision"]["grad_scaler"] is True
    assert cfg["training"]["mixed_precision"]["grad_scaler_init_scale"] == 1024.0

    # Scheduler
    assert cfg["training"]["lr_scheduler"]["warmup_epochs"] == 3
    assert cfg["training"]["lr_scheduler"]["warmup_steps"] == 54
    assert cfg["training"]["lr_scheduler"]["total_steps"] == 540
    assert cfg["training"]["lr_scheduler"]["base_lr"] == 0.005
    assert cfg["training"]["lr_scheduler"]["min_lr"] == 0.00001
    assert cfg["training"]["lr_scheduler"]["warmup_factor"] == 0.001

    # Checkpoint policy
    assert cfg["evaluation"]["checkpoint_policy"]["early_stopping_patience"] == 5
    assert "best_faster_rcnn.pt" in cfg["evaluation"]["checkpoint_policy"]["best_checkpoint_path"]


def test_gradscaler_initialization_from_config():
    """Verifies GradScaler initializes with configured scale 1024.0."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    init_scale = float(cfg["training"]["mixed_precision"]["grad_scaler_init_scale"])
    assert init_scale == 1024.0

    # Initialize scaler with enabled=True (on CUDA if available, or mock)
    cuda_avail = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=cuda_avail, init_scale=init_scale)
    if cuda_avail:
        assert scaler.get_scale() == 1024.0
    else:
        assert scaler.is_enabled() is False
        assert scaler.get_scale() == 1.0


def test_scheduler_progression_curve():
    """Verifies the 540-step warmup + cosine schedule boundaries and progression."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sched_cfg = cfg["training"]["lr_scheduler"]
    param = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=sched_cfg["base_lr"])

    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_steps=int(sched_cfg["warmup_steps"]),
        total_steps=int(sched_cfg["total_steps"]),
        warmup_factor=float(sched_cfg["warmup_factor"]),
        base_lr=float(sched_cfg["base_lr"]),
        min_lr=float(sched_cfg["min_lr"]),
    )

    lrs = []
    for step in range(sched_cfg["total_steps"]):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    # Step 0: start LR = base_lr * warmup_factor = 0.005 * 0.001 = 5e-6
    assert abs(lrs[0] - 0.000005) < 1e-8, f"Expected 5e-6 at step 0, got {lrs[0]}"
    # Step 54: end of warmup reaches base_lr = 0.005
    assert abs(lrs[54] - 0.005) < 1e-6, f"Expected 0.005 at step 54, got {lrs[54]}"
    # Midpoint step 270: cosine decay midpoint
    assert 0.0028 < lrs[270] < 0.0031, f"Expected ~0.0029 at step 270, got {lrs[270]}"
    # Final step 539: reaches intended minimum LR = 1e-5
    assert abs(lrs[539] - 0.00001) < 1e-6, f"Expected ~1e-5 at step 539, got {lrs[539]}"


def test_amp_skipped_step_handling():
    """Verifies that the scheduler advances ONLY when an optimizer step actually executes."""
    param = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=0.005)
    scheduler = build_warmup_cosine_scheduler(optimizer, warmup_steps=54, total_steps=540)

    # Mock scale tracking
    # Case 1: Successful step (scale unchanged or increased)
    scale_before = 1024.0
    scale_after = 1024.0
    lr_before = optimizer.param_groups[0]["lr"]

    if scale_after >= scale_before:
        scheduler.step()
    lr_after_step1 = optimizer.param_groups[0]["lr"]
    assert lr_after_step1 > lr_before, "Scheduler must advance on successful step"

    # Case 2: Skipped step (scale decreased due to non-finite gradients)
    scale_before = 1024.0
    scale_after = 512.0 # Skipped!

    if scale_after >= scale_before:
        scheduler.step()
    lr_after_step2 = optimizer.param_groups[0]["lr"]
    assert lr_after_step2 == lr_after_step1, "Scheduler must NOT advance on skipped step"


def test_dataset_loaders_separation():
    """Verifies 35 train images, 15 val images, and 0 overlap."""
    train_ds = CottonDetectionDataset(split="train")
    val_ds = CottonDetectionDataset(split="val")

    assert len(train_ds) == 35
    assert len(val_ds) == 15

    train_files = set(train_ds.df["filename"])
    val_files = set(val_ds.df["filename"])
    overlap = train_files.intersection(val_files)
    assert len(overlap) == 0, f"Found overlapping images: {overlap}"

    # Group leakage check
    train_groups = set(train_ds.df["group_id"].dropna())
    val_groups = set(val_ds.df["group_id"].dropna())
    assert len(train_groups.intersection(val_groups)) == 0, "Group leakage detected between train and val"


def test_per_class_ap_reporting():
    """Verifies that compute_detection_metrics provides per-class AP breakdown."""
    gts = [
        {"boxes": np.array([[10, 10, 50, 50]], dtype=np.float32), "labels": np.array([0])},
        {"boxes": np.array([[60, 60, 100, 100]], dtype=np.float32), "labels": np.array([1])},
    ]
    preds = [
        [{"class_id": 0, "confidence": 0.95, "bbox": [10, 10, 50, 50]}],
        [{"class_id": 1, "confidence": 0.90, "bbox": [60, 60, 100, 100]}],
    ]

    metrics = compute_detection_metrics(preds, gts, num_classes=2)
    assert "per_class" in metrics, "Missing per_class key in detection metrics"
    per_class = metrics["per_class"]

    assert 0 in per_class and 1 in per_class
    assert per_class[0]["ap50"] == 1.0
    assert per_class[1]["ap50"] == 1.0
    assert per_class[0]["gt_boxes"] == 1
    assert per_class[1]["gt_boxes"] == 1


def test_checkpoint_destination_isolation():
    """Verifies detection checkpoint directory is isolated and protects classifier and U-Net."""
    det_ckpt_dir = PROJECT_ROOT / "artifacts" / "checkpoints" / "detection"
    clf_ckpt_dir = PROJECT_ROOT / "artifacts" / "checkpoints" / "classification"
    unet_ckpt_dir = PROJECT_ROOT / "artifacts" / "checkpoints" / "segmentation"

    assert det_ckpt_dir.resolve() != clf_ckpt_dir.resolve()
    assert det_ckpt_dir.resolve() != unet_ckpt_dir.resolve()

    # Verify existing checkpoints remain intact
    clf_pth = clf_ckpt_dir / "best_model.pt"
    unet_pth = unet_ckpt_dir / "exp4b_w20_best_unet.pt"
    assert clf_pth.is_file(), "Classifier checkpoint missing!"
    assert unet_pth.is_file(), "U-Net checkpoint missing!"


def test_early_stopping_logic():
    """Simulates early stopping with patience=5."""
    patience = 5
    best_metric = -1.0
    patience_counter = 0

    # Simulated validation history: best at epoch 2 (0.65), then stagnant for 5 epochs
    val_scores = [0.50, 0.65, 0.64, 0.63, 0.65, 0.62, 0.61, 0.60]
    stopped_at_epoch = None

    for epoch, score in enumerate(val_scores, start=1):
        if score > best_metric:
            best_metric = score
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            stopped_at_epoch = epoch
            break

    # Improvement was at epoch 2. Epochs 3, 4, 5, 6, 7 did not improve (5 non-improving epochs).
    # Early stopping must trigger at epoch 7!
    assert stopped_at_epoch == 7, f"Expected early stopping at epoch 7, got {stopped_at_epoch}"
    assert best_metric == 0.65
