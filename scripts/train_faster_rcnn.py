"""
Faster R-CNN ResNet-50 FPN V2 Training Pipeline for CottonGuard AI.

Architecture: TorchVision Faster R-CNN ResNet-50 FPN V2 (7 classes: 0=background, 1..6=diseases).
Dataset: Verified published 50-image dataset (35 train / 15 val, 514 bounding boxes).
Hardware Target: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM) / Graceful CPU fallback.

Features:
- Config-driven execution (reads grad_scaler_init_scale, batch_size, epochs, lr, seed from YAML).
- Four-component loss tracking (classifier, box_reg, objectness, rpn_box_reg).
- AMP mixed precision with calibrated GradScaler (init_scale=1024.0).
- Gradient unscaling before clipping at max_norm=10.0.
- Per-batch warmup (54 steps) + cosine decay (486 steps) schedule (540 total steps).
- Successful-optimizer-step-aware scheduler progression (protects against AMP skipped steps).
- Per-epoch validation with mAP@0.50, mAP@0.50:0.95, and per-class AP tables.
- Early stopping with patience=5 on val_map50.
- Isolated checkpoint destination (artifacts/checkpoints/detection/) protecting classifier & U-Net.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.detection_dataset import (
    CottonDetectionDataset,
    FOREGROUND_CLASSES,
    detection_collate_fn,
)
from src.models.faster_rcnn import build_faster_rcnn_model
from src.training.detection_losses import compute_detection_metrics


def save_checkpoint_atomic(payload: Dict[str, Any], dest_path: Union[str, Path]) -> Path:
    """
    Saves a checkpoint atomically:
    1. Writes to a temporary file `<dest_path>.tmp` in the same directory.
    2. Atomically replaces destination path via os.replace.
    3. Cleans up temporary file if serialization fails before replacement.
    4. Preserves any existing valid checkpoint on write failure.
    """
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest.with_name(f"{dest.name}.tmp")
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, dest)
        return dest
    except Exception:
        if temp_path.is_file():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


def validate_checkpoint_paths(
    cfg: Dict[str, Any],
    project_root: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Validates and resolves checkpoint output paths from configuration.
    Enforces strict isolation, containment, and collision invariants:
    - Rule A: Both paths must be contained within `artifacts/checkpoints/detection/`.
    - Rule B: Best and latest checkpoint paths must be distinct.
    - Rule C: Must not collide with classifier or U-Net checkpoints or directories.
    - Rule D: Rejects path traversal (`..`), external absolute paths, and symlink escapes.
    - Rule E: Fails safely before creating any directories or files.
    """
    if project_root is None:
        project_root = PROJECT_ROOT
    project_root = Path(project_root).resolve()

    ckpt_policy = cfg.get("evaluation", {}).get("checkpoint_policy", {})
    if not isinstance(ckpt_policy, dict):
        raise ValueError("Configuration error: 'evaluation.checkpoint_policy' must be a dictionary.")

    best_raw = ckpt_policy.get("best_checkpoint_path")
    latest_raw = ckpt_policy.get("latest_checkpoint_path")

    # Missing / empty validation
    if not best_raw or not isinstance(best_raw, str) or not best_raw.strip():
        raise ValueError("Configuration error: 'evaluation.checkpoint_policy.best_checkpoint_path' is missing or empty.")
    if not latest_raw or not isinstance(latest_raw, str) or not latest_raw.strip():
        raise ValueError("Configuration error: 'evaluation.checkpoint_policy.latest_checkpoint_path' is missing or empty.")

    # Rule D: Path traversal check on raw path components
    best_raw_path = Path(best_raw.strip())
    latest_raw_path = Path(latest_raw.strip())
    if ".." in best_raw_path.parts:
        raise ValueError(f"Path traversal detected in best_checkpoint_path: '{best_raw}' contains '..'")
    if ".." in latest_raw_path.parts:
        raise ValueError(f"Path traversal detected in latest_checkpoint_path: '{latest_raw}' contains '..'")

    # Resolve paths against project root if relative
    best_resolved = (best_raw_path if best_raw_path.is_absolute() else (project_root / best_raw_path)).resolve()
    latest_resolved = (latest_raw_path if latest_raw_path.is_absolute() else (project_root / latest_raw_path)).resolve()

    allowed_dir = (project_root / "artifacts" / "checkpoints" / "detection").resolve()
    clf_dir = (project_root / "artifacts" / "checkpoints" / "classification").resolve()
    unet_dir = (project_root / "artifacts" / "checkpoints" / "segmentation").resolve()
    clf_ckpt = (clf_dir / "best_model.pt").resolve()
    unet_ckpt = (unet_dir / "exp4b_w20_best_unet.pt").resolve()

    # Rule A & Rule D: Path containment validation
    try:
        best_resolved.relative_to(allowed_dir)
    except ValueError:
        raise ValueError(
            f"Configuration error: best_checkpoint_path '{best_resolved}' escapes allowed directory '{allowed_dir}'."
        )

    try:
        latest_resolved.relative_to(allowed_dir)
    except ValueError:
        raise ValueError(
            f"Configuration error: latest_checkpoint_path '{latest_resolved}' escapes allowed directory '{allowed_dir}'."
        )

    # Must be files inside allowed_dir, not the directory itself
    if best_resolved == allowed_dir or latest_resolved == allowed_dir:
        raise ValueError("Configuration error: Checkpoint path cannot be the checkpoint directory itself.")

    # Suffix check
    if best_resolved.suffix not in [".pt", ".pth"]:
        raise ValueError(f"Configuration error: best_checkpoint_path must have .pt or .pth extension, got '{best_resolved.suffix}'.")
    if latest_resolved.suffix not in [".pt", ".pth"]:
        raise ValueError(f"Configuration error: latest_checkpoint_path must have .pt or .pth extension, got '{latest_resolved.suffix}'.")

    # Rule B: Distinct filenames
    if best_resolved == latest_resolved:
        raise ValueError(
            f"Configuration error: best and latest checkpoint paths must be distinct, both resolved to '{best_resolved}'."
        )

    # Rule C: Existing model protection
    if best_resolved == clf_ckpt or latest_resolved == clf_ckpt:
        raise ValueError("Security violation: Checkpoint path collides with production classifier checkpoint!")
    if best_resolved == unet_ckpt or latest_resolved == unet_ckpt:
        raise ValueError("Security violation: Checkpoint path collides with production U-Net checkpoint!")
    if clf_dir in best_resolved.parents or clf_dir in latest_resolved.parents:
        raise ValueError("Security violation: Checkpoint path redirects into classification directory!")
    if unet_dir in best_resolved.parents or unet_dir in latest_resolved.parents:
        raise ValueError("Security violation: Checkpoint path redirects into segmentation directory!")

    return best_resolved, latest_resolved


def check_existing_checkpoint_guard(
    best_ckpt_path: Path,
    overwrite_existing: bool = False,
    resume_path: Optional[str] = None,
) -> Optional[Path]:
    """
    Enforces pre-training checkpoint overwrite protection.
    - Rule A: If best_ckpt_path does not exist, allows fresh run.
    - Rule B: If best_ckpt_path exists and neither overwrite nor resume supplied, raises FileExistsError.
    - Rule C: If overwrite_existing is True and best_ckpt_path exists, creates and SHA-256 verifies a timestamped backup.
    - Rule D: If resume_path is supplied, raises NotImplementedError (deferred).
    - Rule E: Executes prior to dataset, model, and optimizer initialization.
    Returns: backup_path if backup created, else None.
    """
    if resume_path is not None:
        raise NotImplementedError(
            "Resume functionality is deferred and not supported in this version. "
            "Please remove the '--resume' argument or wait for future resume authorization."
        )

    if not best_ckpt_path.is_file():
        # Rule A: Fresh run when no checkpoint exists
        return None

    # Best checkpoint exists!
    if not overwrite_existing:
        # Rule B: Existing best checkpoint without explicit overwrite flag
        raise FileExistsError(
            f"Pre-existing trained checkpoint found at '{best_ckpt_path}'. "
            "To prevent accidental data loss, training is blocked. "
            "Provide '--overwrite-existing' to explicitly authorize creating a timestamped backup and replacing it."
        )

    # Rule C: Explicit overwrite authorized. Create timestamped backup and verify SHA-256.
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = best_ckpt_path.with_name(f"{best_ckpt_path.stem}_bak_{ts}.pt")

    orig_bytes = best_ckpt_path.read_bytes()
    orig_hash = hashlib.sha256(orig_bytes).hexdigest().upper()

    shutil.copy2(best_ckpt_path, backup_path)

    if not backup_path.is_file():
        raise RuntimeError(f"Backup creation failed: '{backup_path}' does not exist.")

    bak_hash = hashlib.sha256(backup_path.read_bytes()).hexdigest().upper()
    if bak_hash != orig_hash:
        try:
            backup_path.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"Backup verification failed: SHA-256 mismatch for '{backup_path}' "
            f"(original={orig_hash}, backup={bak_hash}). Training aborted."
        )

    print(f"[Backup] Successfully archived pre-existing best checkpoint to: {backup_path} (SHA-256: {bak_hash})")
    return backup_path


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 54,
    total_steps: int = 540,
    warmup_factor: float = 0.001,
    base_lr: float = 0.005,
    min_lr: float = 0.00001,
) -> LambdaLR:
    """
    Constructs a per-batch learning rate schedule:
    - Steps 0 .. warmup_steps-1: Linear warmup from (base_lr * warmup_factor) to base_lr.
    - Steps warmup_steps .. total_steps-1: Half-cycle Cosine Annealing from base_lr down to min_lr.
    """
    min_factor = float(min_lr) / float(base_lr)
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            alpha = float(step) / float(max(1, warmup_steps))
            return warmup_factor + (1.0 - warmup_factor) * alpha
        else:
            progress = float(step - warmup_steps) / float(decay_steps)
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine_decay

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def evaluate_detection_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_classes: int = 6,
) -> Dict[str, Any]:
    """
    Evaluates detector on validation set in model.eval() mode without gradient computation.
    Maps model output class labels (1..6) to 0-based IDs (0..5) for metric calculation.
    """
    model.eval()
    val_preds_list: List[List[Dict[str, Any]]] = []
    val_gts_list: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch_imgs, batch_targets in val_loader:
            imgs_dev = [im.to(device) for im in batch_imgs]
            preds = model(imgs_dev)

            for b in range(len(preds)):
                pred_dict = preds[b]
                tgt_dict = batch_targets[b]

                p_boxes = pred_dict["boxes"].cpu().numpy()
                p_labels = pred_dict["labels"].cpu().numpy()
                p_scores = pred_dict["scores"].cpu().numpy()

                gt_boxes = tgt_dict["boxes"].cpu().numpy()
                gt_labels = tgt_dict["labels"].cpu().numpy()

                img_preds = []
                for p_box, p_lbl, p_scr in zip(p_boxes, p_labels, p_scores):
                    # TorchVision classes are 1..6; map to 0..5 (ignore background label 0)
                    if p_lbl >= 1:
                        img_preds.append({
                            "class_id": int(p_lbl - 1),
                            "confidence": float(p_scr),
                            "bbox": p_box.tolist(),
                        })

                # Map ground truth labels 1..6 -> 0..5
                gt_mapped = (gt_labels - 1) if gt_labels.size > 0 else np.zeros((0,), dtype=np.int64)

                val_preds_list.append(img_preds)
                val_gts_list.append({
                    "boxes": gt_boxes,
                    "labels": gt_mapped,
                })

    metrics = compute_detection_metrics(
        predictions_per_image=val_preds_list,
        ground_truths_per_image=val_gts_list,
        num_classes=num_classes,
    )
    return metrics


def train_faster_rcnn(
    config_path: Path,
    device_name: Optional[str] = None,
    dry_run_check_only: bool = False,
    overwrite_existing: bool = False,
    resume_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes or pre-verifies the Faster R-CNN ResNet-50 FPN V2 training workflow.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Checkpoint Path Validation & Overwrite Protection Guard
    best_ckpt_p, latest_ckpt_p = validate_checkpoint_paths(cfg, PROJECT_ROOT)
    check_existing_checkpoint_guard(
        best_ckpt_path=best_ckpt_p,
        overwrite_existing=overwrite_existing,
        resume_path=resume_path,
    )
    ckpt_dir = best_ckpt_p.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 2. Reproducibility Seeding
    seed = int(cfg.get("experiment", {}).get("reproducibility_seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 3. Device Setup
    if device_name is not None:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using compute device: {device}")

    # 4. Dataset & DataLoader Instantiation
    ds_cfg = cfg.get("dataset", {})
    manifest_p = ds_cfg.get("manifest_path")
    data_d = ds_cfg.get("data_dir")
    ignore_d = ds_cfg.get("ignore_dir")

    train_ds = CottonDetectionDataset(
        split="train",
        manifest_path=(PROJECT_ROOT / manifest_p) if manifest_p else None,
        data_dir=(PROJECT_ROOT / data_d) if data_d else None,
        ignore_dir=(PROJECT_ROOT / ignore_d) if ignore_d else None,
    )
    val_ds = CottonDetectionDataset(
        split="val",
        manifest_path=(PROJECT_ROOT / manifest_p) if manifest_p else None,
        data_dir=(PROJECT_ROOT / data_d) if data_d else None,
        ignore_dir=(PROJECT_ROOT / ignore_d) if ignore_d else None,
    )
    expected_train = int(ds_cfg.get("train_samples", 35))
    expected_val = int(ds_cfg.get("val_samples", 15))
    assert len(train_ds) == expected_train, f"Expected {expected_train} train images, got {len(train_ds)}"
    assert len(val_ds) == expected_val, f"Expected {expected_val} val images, got {len(val_ds)}"

    batch_size = int(cfg.get("training", {}).get("batch_size", 2))
    num_workers = int(cfg.get("training", {}).get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detection_collate_fn,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        collate_fn=detection_collate_fn,
        num_workers=num_workers,
    )

    batches_per_epoch = len(train_loader)
    epochs = int(cfg.get("training", {}).get("max_epochs", 30))
    total_steps = epochs * batches_per_epoch

    warmup_epochs = int(cfg.get("training", {}).get("lr_scheduler", {}).get("warmup_epochs", 3))
    warmup_steps = warmup_epochs * batches_per_epoch
    warmup_factor = float(cfg.get("training", {}).get("lr_scheduler", {}).get("warmup_factor", 0.001))
    base_lr = float(cfg.get("training", {}).get("optimizer", {}).get("lr", 0.005))
    min_lr = float(cfg.get("training", {}).get("lr_scheduler", {}).get("min_lr", 0.00001))
    momentum = float(cfg.get("training", {}).get("optimizer", {}).get("momentum", 0.9))
    weight_decay = float(cfg.get("training", {}).get("optimizer", {}).get("weight_decay", 0.0005))
    clip_norm = float(cfg.get("training", {}).get("gradient_clipping", {}).get("max_norm", 10.0))

    # 5. Model Instantiation
    model = build_faster_rcnn_model(
        num_classes=int(cfg.get("model", {}).get("num_classes", 7)),
        pretrained=bool(cfg.get("model", {}).get("pretrained", True)),
        trainable_backbone_layers=int(cfg.get("model", {}).get("trainable_backbone_layers", 3)),
        device=device,
        enable_ignore_regions=bool(cfg.get("model", {}).get("enable_ignore_regions", False)),
        rpn_ignore_threshold=float(cfg.get("model", {}).get("rpn_ignore_threshold", 0.30)),
        roi_ignore_threshold=float(cfg.get("model", {}).get("roi_ignore_threshold", 0.30)),
    )

    # 6. Optimizer & Scaler Setup (Supports AdamW and SGD)
    opt_cfg = cfg.get("training", {}).get("optimizer", {})
    opt_type = str(opt_cfg.get("type", "SGD")).upper()
    if opt_type == "ADAMW":
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=base_lr,
            betas=(
                float(opt_cfg.get("beta1", 0.9)),
                float(opt_cfg.get("beta2", 0.999)),
            ),
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    amp_cfg = cfg.get("training", {}).get("mixed_precision", {})
    amp_enabled = bool(amp_cfg.get("enabled", True) and device.type == "cuda")
    grad_scaler_init_scale = float(amp_cfg.get("grad_scaler_init_scale", 1024.0))

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled, init_scale=grad_scaler_init_scale)
    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        warmup_factor=warmup_factor,
        base_lr=base_lr,
        min_lr=min_lr,
    )

    print("\n[Configuration Verified]")
    print(f"  - Dataset: {len(train_ds)} train images ({batches_per_epoch} batches/epoch), {len(val_ds)} val images")
    print(f"  - Schedule: {epochs} epochs, {total_steps} total steps, {warmup_steps} warmup steps")
    print(f"  - Learning Rate: start={base_lr * warmup_factor:.6f}, base={base_lr}, min={min_lr}")
    print(f"  - AMP: enabled={amp_enabled}, GradScaler init_scale={grad_scaler_init_scale}")
    print(f"  - Checkpoint Directory: {ckpt_dir}")

    # If dry-run check only, verify graph & setup without executing any training steps
    if dry_run_check_only:
        print("[Dry-Run Check] Verified all configuration parameters, data loaders, and model graph.")
        print("[Dry-Run Check] ZERO training epochs or optimizer steps executed.")
        return {
            "status": "DRY_RUN_CHECK_PASSED",
            "epochs": epochs,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "grad_scaler_init_scale": grad_scaler_init_scale,
            "batches_per_epoch": batches_per_epoch,
            "checkpoint_destination": str(ckpt_dir),
            "detector_status": "NOT_TRAINED",
        }

    # 7. Training Loop with Protection Invariants
    patience = int(cfg.get("evaluation", {}).get("checkpoint_policy", {}).get("early_stopping_patience", 5))
    best_map50 = -1.0
    best_map50_95 = -1.0
    patience_counter = 0
    training_history: List[Dict[str, Any]] = []

    global_step = 0
    total_skipped_steps = 0
    t_start = time.time()

    print(f"\n[Training Execution Started] Target: {epochs} epochs, early stopping patience: {patience}")

    try:
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss_total = 0.0
            epoch_loss_cls = 0.0
            epoch_loss_box = 0.0
            epoch_loss_obj = 0.0
            epoch_loss_rpn = 0.0
            successful_steps = 0
            epoch_skipped_steps = 0

            ep_t0 = time.time()

            for batch_idx, (batch_imgs, batch_targets) in enumerate(train_loader):
                imgs_dev = [im.to(device) for im in batch_imgs]
                targets_dev = [{k: v.to(device) for k, v in tgt.items()} for tgt in batch_targets]

                optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
                    loss_dict = model(imgs_dev, targets_dev)
                    loss_total = sum(loss for loss in loss_dict.values())

                # Check finite loss
                if not torch.isfinite(loss_total):
                    print(f"WARNING: Non-finite loss at epoch {epoch}, batch {batch_idx}: {loss_total.item()}")
                    continue

                # Backward pass with AMP scaling
                scaler.scale(loss_total).backward()

                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

                # Optimizer step with scale tracking
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()

                # Check if optimizer actually stepped (if scale decreased, step was skipped)
                optimizer_stepped = (scale_after >= scale_before)
                if optimizer_stepped:
                    scheduler.step()
                    global_step += 1
                    successful_steps += 1
                else:
                    epoch_skipped_steps += 1
                    total_skipped_steps += 1

                # Accumulate loss components
                epoch_loss_total += float(loss_total.item())
                epoch_loss_cls += float(loss_dict.get("loss_classifier", 0.0).item())
                epoch_loss_box += float(loss_dict.get("loss_box_reg", 0.0).item())
                epoch_loss_obj += float(loss_dict.get("loss_objectness", 0.0).item())
                epoch_loss_rpn += float(loss_dict.get("loss_rpn_box_reg", 0.0).item())

            avg_loss_total = epoch_loss_total / max(1, batches_per_epoch)
            avg_loss_cls = epoch_loss_cls / max(1, batches_per_epoch)
            avg_loss_box = epoch_loss_box / max(1, batches_per_epoch)
            avg_loss_obj = epoch_loss_obj / max(1, batches_per_epoch)
            avg_loss_rpn = epoch_loss_rpn / max(1, batches_per_epoch)
            current_lr = float(optimizer.param_groups[0]["lr"])

            # 8. Per-Epoch Validation
            val_metrics = evaluate_detection_epoch(
                model=model,
                val_loader=val_loader,
                device=device,
                num_classes=6,
            )

            val_map50 = float(val_metrics.get("mAP50", 0.0))
            val_map50_95 = float(val_metrics.get("mAP50_95", 0.0))
            val_prec = float(val_metrics.get("precision", 0.0))
            val_rec = float(val_metrics.get("recall", 0.0))
            per_class_res = val_metrics.get("per_class", {})

            ep_time = time.time() - ep_t0

            print(
                f"[Epoch {epoch:02d}/{epochs:02d}] "
                f"Loss: {avg_loss_total:.4f} (cls:{avg_loss_cls:.3f}, box:{avg_loss_box:.3f}, obj:{avg_loss_obj:.3f}, rpn:{avg_loss_rpn:.3f}) | "
                f"Val mAP50: {val_map50:.4f} | mAP50:95: {val_map50_95:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | "
                f"LR: {current_lr:.6f} | Scale: {scaler.get_scale():.1f} ({ep_time:.1f}s)"
            )

            # Per-class AP logging
            for cid, cname in enumerate(FOREGROUND_CLASSES):
                c_info = per_class_res.get(cid, {})
                c_ap50 = c_info.get("ap50", 0.0)
                c_gt = c_info.get("gt_boxes", 0)
                print(f"    - {cname}: AP50={c_ap50:.4f} (GT={c_gt})")

            # Record history
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "successful_steps": successful_steps,
                "skipped_steps": epoch_skipped_steps,
                "total_skipped_steps": total_skipped_steps,
                "lr": current_lr,
                "train_loss": round(avg_loss_total, 4),
                "loss_classifier": round(avg_loss_cls, 4),
                "loss_box_reg": round(avg_loss_box, 4),
                "loss_objectness": round(avg_loss_obj, 4),
                "loss_rpn_box_reg": round(avg_loss_rpn, 4),
                "val_mAP50": round(val_map50, 4),
                "val_mAP50_95": round(val_map50_95, 4),
                "val_precision": round(val_prec, 4),
                "val_recall": round(val_rec, 4),
                "per_class": per_class_res,
                "scaler_scale": scaler.get_scale(),
                "epoch_time_sec": round(ep_time, 2),
            }
            training_history.append(epoch_record)

            # 9. Checkpoint Policy
            is_best = val_map50 > best_map50
            if is_best:
                best_map50 = val_map50
                best_map50_95 = val_map50_95
                patience_counter = 0
                print(f"  >>> New Best Model at Epoch {epoch}! (val_mAP50 = {best_map50:.4f})")

                checkpoint_payload = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_map50": best_map50,
                    "best_map50_95": best_map50_95,
                    "val_metrics": val_metrics,
                    "config": cfg,
                    "pretrained_weights_sha256": "DD69338A24B8D7381807E247652BDC356325BCBAF1CD3E092E00E0A1A58706BF",
                    "seed": seed,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                save_checkpoint_atomic(checkpoint_payload, best_ckpt_p)
                print(f"  Saved best checkpoint to: {best_ckpt_p}")
            else:
                patience_counter += 1
                print(f"  No improvement for {patience_counter}/{patience} epochs.")

            # Always save latest checkpoint
            latest_payload = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "current_map50": val_map50,
                "best_map50": best_map50,
                "val_metrics": val_metrics,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            save_checkpoint_atomic(latest_payload, latest_ckpt_p)

            # 10. Early Stopping Check
            if patience_counter >= patience:
                print(f"\n[Early Stopping] No improvement in val_mAP50 for {patience} consecutive epochs. Terminating training.")
                break

    except torch.cuda.OutOfMemoryError as oom_err:
        print(f"\n[CUDA OOM FATAL] Caught CUDA Out Of Memory Error during training: {oom_err}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise oom_err
    except Exception as e:
        print(f"\n[Training Exception] Unexpected error during training: {e}")
        raise e

    total_training_time = time.time() - t_start
    print(f"\n[Training Completed] Total elapsed time: {total_training_time:.2f}s ({total_training_time / 60:.2f}m)")
    print(f"Final Best val_mAP50: {best_map50:.4f} (mAP50:95: {best_map50_95:.4f})")

    # Save training history JSON
    history_file = PROJECT_ROOT / "artifacts" / "metrics" / "phase9_2_training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump({
            "best_map50": best_map50,
            "best_map50_95": best_map50_95,
            "total_epochs_trained": len(training_history),
            "total_training_time_sec": round(total_training_time, 2),
            "history": training_history,
        }, f, indent=2)
    print(f"Saved training history to {history_file}")

    return {
        "status": "TRAINING_COMPLETED",
        "best_map50": best_map50,
        "best_map50_95": best_map50_95,
        "epochs_trained": len(training_history),
        "best_checkpoint": str(best_ckpt_p),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN ResNet-50 FPN V2 for Cotton Leaf AI")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "phase9_2a_detector_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument("--device", type=str, default=None, help="Target device (cuda or cpu)")
    parser.add_argument(
        "--dry-run-check",
        action="store_true",
        help="Validate configuration, loaders, and model graph without running training",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Explicitly authorize archiving and overwriting an existing best detector checkpoint",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from an existing checkpoint (currently deferred / unsupported)",
    )
    args = parser.parse_args()

    train_faster_rcnn(
        config_path=args.config,
        device_name=args.device,
        dry_run_check_only=args.dry_run_check,
        overwrite_existing=args.overwrite_existing,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
