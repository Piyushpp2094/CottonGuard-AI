"""
Unit tests for Cotton Leaf AI Classification Model and Training Components (Phase 3).
"""

import sys
import unittest
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.classifier import CottonLeafClassifier
from src.models.model_factory import build_classifier
from src.training.losses import get_classification_loss
from src.training.callbacks import EarlyStopping, ModelCheckpoint


class TestClassifier(unittest.TestCase):

    def test_model_forward_and_shapes(self):
        model = CottonLeafClassifier(num_classes=7, pretrained=False, dropout=0.3)
        dummy_input = torch.randn(2, 3, 224, 224)
        logits = model(dummy_input)
        self.assertEqual(logits.shape, (2, 7))

        probs = model.predict_proba(dummy_input)
        self.assertEqual(probs.shape, (2, 7))
        prob_sums = probs.sum(dim=1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones(2), atol=1e-5))

    def test_backbone_freezing(self):
        model = CottonLeafClassifier(num_classes=7, pretrained=False)
        model.freeze_backbone()
        for param in model.backbone.features.parameters():
            self.assertFalse(param.requires_grad)
        for param in model.backbone.classifier.parameters():
            self.assertTrue(param.requires_grad)

        model.unfreeze_backbone()
        for param in model.backbone.features.parameters():
            self.assertTrue(param.requires_grad)

    def test_loss_function(self):
        class_weights = torch.ones(7)
        criterion = get_classification_loss(class_weights=class_weights, label_smoothing=0.05)
        dummy_logits = torch.randn(4, 7)
        dummy_targets = torch.tensor([0, 2, 4, 6])
        loss = criterion(dummy_logits, dummy_targets)
        self.assertGreater(loss.item(), 0.0)

    def test_early_stopping(self):
        es = EarlyStopping(patience=3, min_delta=0.01, mode="max", verbose=False)
        self.assertFalse(es(0.80))  # initial best
        self.assertFalse(es(0.805)) # improvement < min_delta -> counter=1
        self.assertFalse(es(0.79))  # worse -> counter=2
        self.assertTrue(es(0.78))   # worse -> counter=3 -> triggers stop

    def test_model_checkpoint(self):
        tmp_dir = PROJECT_ROOT / "artifacts" / "metrics" / "test_tmp_ckpt"
        ckpt = ModelCheckpoint(checkpoint_dir=tmp_dir, filename="test_best.pt", mode="max")
        model = CottonLeafClassifier(num_classes=7, pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Step 1: initial best
        improved = ckpt.step(0.85, model, optimizer, epoch=1, metrics={"macro_f1": 0.85})
        self.assertTrue(improved)
        self.assertTrue((tmp_dir / "test_best.pt").is_file())

        # Cleanup test file
        if (tmp_dir / "test_best.pt").is_file():
            (tmp_dir / "test_best.pt").unlink()
        if (tmp_dir / "last_model.pt").is_file():
            (tmp_dir / "last_model.pt").unlink()
        if tmp_dir.is_dir():
            tmp_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
