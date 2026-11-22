"""
Unit tests for Segmentation, Affected Area Estimation, Severity Analysis,
and Farmer Recommendation Engine (Phases 4, 5, 6).
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet import CottonLeafUNet
from src.training.segmentation_losses import SoftDiceLoss, CombinedDiceCELoss, compute_segmentation_metrics
from src.metrics.area_calculation import calculate_affected_area_from_mask, generate_segmentation_overlays
from src.metrics.severity import SeverityEstimator
from src.recommendations.recommendation_engine import RecommendationEngine
from src.data.dataset import CLASS_NAMES


class TestSegmentationSeverityRecommendations(unittest.TestCase):

    def test_unet_architecture(self):
        model = CottonLeafUNet(in_channels=3, num_classes=3, init_features=16)
        dummy_input = torch.randn(1, 3, 128, 128)
        logits = model(dummy_input)
        self.assertEqual(logits.shape, (1, 3, 128, 128))

    def test_segmentation_losses_and_metrics(self):
        criterion = CombinedDiceCELoss()
        dummy_logits = torch.randn(2, 3, 64, 64)
        dummy_targets = torch.randint(0, 3, (2, 64, 64))
        loss = criterion(dummy_logits, dummy_targets)
        self.assertGreater(loss.item(), 0.0)

        preds = torch.argmax(dummy_logits, dim=1)
        metrics = compute_segmentation_metrics(preds, dummy_targets, num_classes=3)
        self.assertIn("mean_iou", metrics)
        self.assertIn("mean_dice", metrics)
        self.assertGreaterEqual(metrics["mean_iou"], 0.0)
        self.assertLessEqual(metrics["mean_iou"], 1.0)

    def test_affected_area_calculation(self):
        # Create a synthetic 100x100 mask
        mask = np.zeros((100, 100), dtype=np.int64)
        # 4000 pixels healthy leaf (class 1)
        mask[20:80, 20:60] = 1
        # 1000 pixels diseased tissue (class 2)
        mask[20:70, 60:80] = 2

        # Total leaf pixels = 60*40 + 50*20 = 2400 + 1000 = 3400
        # Disease pixels = 1000
        # Expected % = (1000 / 3400) * 100 = 29.41%
        res = calculate_affected_area_from_mask(mask)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["leaf_pixels"], 3400)
        self.assertEqual(res["disease_pixels"], 1000)
        self.assertAlmostEqual(res["affected_area_percentage"], 29.41, places=1)

        # Insufficient leaf pixels test (< 500)
        tiny_mask = np.zeros((100, 100), dtype=np.int64)
        tiny_mask[10:15, 10:15] = 1 # 25 pixels
        tiny_res = calculate_affected_area_from_mask(tiny_mask)
        self.assertFalse(tiny_res["is_valid"])
        self.assertIsNone(tiny_res["affected_area_percentage"])

        # Overlay generation test
        dummy_img = Image.new("RGB", (100, 100), color=(100, 150, 80))
        overlays = generate_segmentation_overlays(dummy_img, mask)
        self.assertIn("leaf_mask", overlays)
        self.assertIn("disease_mask", overlays)
        self.assertIn("overlay", overlays)
        self.assertEqual(overlays["overlay"].size, (100, 100))

    def test_leaf_restricted_area_calculation(self):
        """Verifies disease pixels outside leaf mask are zeroed out and do not inflate severity."""
        pred_mask = np.zeros((100, 100), dtype=np.int64)
        leaf_mask = np.zeros((100, 100), dtype=np.int64)
        leaf_mask[20:80, 20:80] = 1 # 3600 px leaf area

        # Inside leaf: 900 px of disease (class 2)
        pred_mask[20:50, 20:50] = 2
        # Inside leaf remainder: 2700 px of healthy (class 1)
        for r in range(20, 80):
            for c in range(20, 80):
                if pred_mask[r, c] != 2:
                    pred_mask[r, c] = 1

        # OUTSIDE leaf: 1000 px of spurious disease predictions in background
        pred_mask[0:10, 0:100] = 2

        # Calculate with leaf_mask_override
        res = calculate_affected_area_from_mask(pred_mask, leaf_mask_override=leaf_mask)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["disease_pixels"], 900)
        self.assertEqual(res["leaf_pixels"], 3600)
        self.assertEqual(res["affected_area_percentage"], 25.0)

    def test_severity_estimation(self):
        estimator = SeverityEstimator()

        # None -> UNAVAILABLE
        res_none = estimator.evaluate(None, "Bacterial Blight", 0.95)
        self.assertEqual(res_none["level"], "UNAVAILABLE")
        self.assertFalse(res_none["is_available"])

        # 0.2% -> NONE
        res_0 = estimator.evaluate(0.2, "Healthy Leaf", 0.92)
        self.assertEqual(res_0["level"], "NONE")

        # 10% -> LOW
        res_low = estimator.evaluate(10.0, "Curl Virus", 0.94)
        self.assertEqual(res_low["level"], "LOW")

        # 25% -> MODERATE
        res_mod = estimator.evaluate(25.0, "Bacterial Blight", 0.91)
        self.assertEqual(res_mod["level"], "MODERATE")

        # 50% -> HIGH
        res_high = estimator.evaluate(50.0, "Leaf Hopper Jassids", 0.96)
        self.assertEqual(res_high["level"], "HIGH")
        # Ensure jassid agronomic note is present
        self.assertTrue(any("economic threshold" in n.lower() for n in res_high["agronomic_notes"]))

        # Low confidence (< 0.60)
        res_unc = estimator.evaluate(12.0, "Leaf Redding", 0.45)
        self.assertIsNotNone(res_unc["uncertainty_warning"])

    def test_recommendation_engine(self):
        engine = RecommendationEngine()

        for class_name in CLASS_NAMES:
            rec = engine.get_recommendation(class_name, confidence=0.92, severity_level="MODERATE")
            self.assertEqual(rec["status"], "SUCCESS")
            self.assertEqual(rec["condition"], class_name)
            self.assertGreater(len(rec["immediate_first_steps"]), 0)
            self.assertGreater(len(rec["preventive_management"]), 0)
            self.assertTrue(len(rec["critical_warning"]) > 0)

        # Check Jassid recommendation notes ICAR-CICR ETL
        jassid_rec = engine.get_recommendation("Leaf Hopper Jassids", confidence=0.95)
        text = " ".join(jassid_rec["immediate_first_steps"]).lower()
        self.assertTrue("economic threshold" in text or "nymphs" in text or "etl" in text)

        # Check Leaf Redding notes abiotic/physiological disorder
        red_rec = engine.get_recommendation("Leaf Redding", confidence=0.95)
        self.assertTrue("physiological" in red_rec["pathogen_or_cause"].lower() or "abiotic" in red_rec["pathogen_or_cause"].lower())


if __name__ == "__main__":
    unittest.main()
