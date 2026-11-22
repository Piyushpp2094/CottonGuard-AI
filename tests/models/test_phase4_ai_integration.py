"""
Phase 4 AI Integration Tests:
- class_names.json integrity vs src.data.dataset.CLASS_NAMES & checkpoint class head
- YOLOv8n gated NOT_TRAINED wrapper (without requiring AGPL-3.0 ultralytics install)
- UNet-ResNet34 (MIT) 512x512 architecture, Albumentations pipeline, weighted CE+Dice loss, NOT_TRAINED gate
- 6-tier severity schema (category_validated=false, calculation_status, backward-compatible level)
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CLASS_NAMES
from src.metrics.severity import SeverityEstimator
from src.models.unet_resnet34 import (
    UNetResNet34Segmentor,
    WeightedCombinedDiceCELoss,
    build_unet_resnet34,
    get_segmentation_eval_transforms,
    get_segmentation_train_transforms,
)
from src.models.yolov8_detector import YOLOV8_DETECTION_CLASSES, YOLOv8DiseaseDetector


class TestPhase4AIIntegration(unittest.TestCase):
    def test_01_class_names_manifest_matches_dataset(self):
        manifest_path = PROJECT_ROOT / "artifacts" / "manifests" / "class_names.json"
        self.assertTrue(manifest_path.is_file())
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(data["num_classes"], 7)
        self.assertEqual(data["classes"], list(CLASS_NAMES))
        for idx, name in enumerate(CLASS_NAMES):
            self.assertEqual(data["class_to_idx"][name], idx)

    def test_02_yolov8n_wrapper_honest_not_trained_gate(self):
        det = YOLOv8DiseaseDetector(checkpoint_path=PROJECT_ROOT / "artifacts" / "checkpoints" / "detection" / "nonexistent_yolo.pt")
        self.assertFalse(det.is_trained)
        self.assertEqual(det.status, "NOT_TRAINED")
        self.assertEqual(len(YOLOV8_DETECTION_CLASSES), 6)
        self.assertNotIn("Healthy Leaf", YOLOV8_DETECTION_CLASSES)
        out = det.predict(Image.new("RGB", (640, 640), color=(40, 120, 50)))
        self.assertEqual(out["status"], "NOT_TRAINED")
        self.assertFalse(out["available"])
        self.assertEqual(out["detections"], [])

    def test_03_unet_resnet34_architecture_loss_transforms_and_gate(self):
        # Gated NOT_TRAINED inference
        seg = UNetResNet34Segmentor(checkpoint_path=PROJECT_ROOT / "artifacts" / "checkpoints" / "segmentation" / "nonexistent_unet.pt")
        self.assertFalse(seg.is_trained)
        self.assertEqual(seg.status, "NOT_TRAINED")
        out = seg.predict(Image.new("RGB", (512, 512), color=(40, 120, 50)))
        self.assertEqual(out["status"], "NOT_TRAINED")
        self.assertFalse(out["available"])

        # Architecture forward pass (512x512, 3 classes, offline encoder_weights=None)
        model = build_unet_resnet34(num_classes=3, encoder_name="resnet34", encoder_weights=None)
        dummy_x = torch.randn(1, 3, 512, 512)
        logits = model(dummy_x)
        self.assertEqual(logits.shape, (1, 3, 512, 512))

        # Weighted CE + Dice loss
        criterion = WeightedCombinedDiceCELoss(class_weights=(0.3, 1.0, 2.5), ce_weight=0.5, dice_weight=0.5)
        targets = torch.randint(0, 3, (1, 512, 512))
        loss = criterion(logits, targets)
        self.assertGreater(float(loss.item()), 0.0)

        # Albumentations 512x512 transforms
        img_np = np.zeros((600, 700, 3), dtype=np.uint8)
        mask_np = np.zeros((600, 700), dtype=np.int64)
        tr_out = get_segmentation_train_transforms(512)(image=img_np, mask=mask_np)
        ev_out = get_segmentation_eval_transforms(512)(image=img_np, mask=mask_np)
        self.assertEqual(tuple(tr_out["image"].shape), (3, 512, 512))
        self.assertEqual(tuple(ev_out["image"].shape), (3, 512, 512))

    def test_04_severity_six_tier_schema_and_validation_flag(self):
        est = SeverityEstimator()
        self.assertFalse(est.category_validated)

        unavail = est.evaluate(None, "Bacterial Blight", 0.95)
        self.assertEqual(unavail["calculation_status"], "model_unavailable")
        self.assertFalse(unavail["category_validated"])

        no_leaf = est.evaluate(0.0, "Healthy Leaf", 0.95, no_leaf_detected=True)
        self.assertEqual(no_leaf["calculation_status"], "no_leaf_detected")

        cases = [
            (0.0, "None"),
            (5.5, "Trace"),
            (25.0, "Mild"),
            (35.0, "Moderate"),
            (60.0, "Severe"),
            (85.0, "Very Severe"),
        ]
        for pct, expected_cat in cases:
            res = est.evaluate(pct, "Bacterial Blight", 0.94, leaf_pixels=10000, disease_pixels=int(pct * 100), healthy_pixels=10000 - int(pct * 100))
            self.assertEqual(res["calculation_status"], "success")
            self.assertEqual(res["severity_category"], expected_cat)
            self.assertFalse(res["category_validated"])
            self.assertEqual(res["calculation_method"], "unet_pixel_ratio_v1")


if __name__ == "__main__":
    unittest.main()
