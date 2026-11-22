"""
Unit and integration tests for Cotton Leaf AI Data Pipeline (Phase 2).
Compatible with both standard Python unittest and pytest.
"""

import os
import sys
import unittest
from pathlib import Path
import torch
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CottonLeafDataset, CLASS_NAMES, CLASS_TO_IDX
from src.data.preprocessing import (
    get_train_transforms,
    get_eval_transforms,
    denormalize_tensor,
    tensor_to_pil,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from src.data.dataloader import (
    compute_class_weights,
    create_weighted_sampler,
    get_dataloaders,
)


class TestDataPipeline(unittest.TestCase):

    def test_class_names_and_mapping(self):
        self.assertEqual(len(CLASS_NAMES), 7)
        expected_classes = [
            "Bacterial Blight",
            "Curl Virus",
            "Healthy Leaf",
            "Herbicide Growth Damage",
            "Leaf Hopper Jassids",
            "Leaf Redding",
            "Leaf Variegation",
        ]
        self.assertEqual(CLASS_NAMES, expected_classes)
        self.assertEqual(CLASS_TO_IDX["Leaf Hopper Jassids"], 4)
        for idx, name in enumerate(CLASS_NAMES):
            self.assertEqual(CLASS_TO_IDX[name], idx)

    def test_preprocessing_transforms(self):
        dummy_img = Image.new("RGB", (300, 400), color=(120, 150, 80))
        
        train_tf = get_train_transforms(image_size=224)
        train_tensor = train_tf(dummy_img)
        self.assertEqual(train_tensor.shape, (3, 224, 224))
        self.assertEqual(train_tensor.dtype, torch.float32)

        eval_tf = get_eval_transforms(image_size=224)
        eval_tensor = eval_tf(dummy_img)
        self.assertEqual(eval_tensor.shape, (3, 224, 224))
        self.assertEqual(eval_tensor.dtype, torch.float32)

        # Denormalization test
        denorm = denormalize_tensor(eval_tensor)
        self.assertGreaterEqual(denorm.min().item(), 0.0)
        self.assertLessEqual(denorm.max().item(), 1.0)

        # Tensor to PIL test
        pil_img = tensor_to_pil(eval_tensor)
        self.assertEqual(pil_img.size, (224, 224))
        self.assertEqual(pil_img.mode, "RGB")

    def test_dataset_splits_and_counts(self):
        splits_dir = PROJECT_ROOT / "artifacts" / "manifests" / "splits"
        train_csv = splits_dir / "train.csv"
        val_csv = splits_dir / "validation.csv"
        test_csv = splits_dir / "test.csv"

        self.assertTrue(train_csv.is_file(), f"Missing {train_csv}")
        self.assertTrue(val_csv.is_file(), f"Missing {val_csv}")
        self.assertTrue(test_csv.is_file(), f"Missing {test_csv}")

        train_ds = CottonLeafDataset(train_csv, transform=get_eval_transforms())
        val_ds = CottonLeafDataset(val_csv, transform=get_eval_transforms())
        test_ds = CottonLeafDataset(test_csv, transform=get_eval_transforms())

        self.assertEqual(len(train_ds), 1492)
        self.assertEqual(len(val_ds), 319)
        self.assertEqual(len(test_ds), 321)
        self.assertEqual(len(train_ds) + len(val_ds) + len(test_ds), 2132)

    def test_class_weights_and_sampler(self):
        splits_dir = PROJECT_ROOT / "artifacts" / "manifests" / "splits"
        train_csv = splits_dir / "train.csv"

        weights = compute_class_weights(train_csv)
        self.assertEqual(weights.shape, (7,))
        self.assertEqual(weights.dtype, torch.float32)

        # Leaf Redding (index 5) has highest count (404), Leaf Variegation (index 6) has lowest count (81)
        self.assertGreater(weights[6].item(), weights[5].item())

        sampler = create_weighted_sampler(train_csv)
        self.assertEqual(len(sampler), 1492)

    def test_dataloaders_and_batch_shapes(self):
        loaders = get_dataloaders(
            project_root=PROJECT_ROOT,
            batch_size=16,
            num_workers=0,  # 0 workers for fast safe test execution
            pin_memory=False,
            use_weighted_sampler=False
        )

        train_loader = loaders["train"]
        val_loader = loaders["val"]
        test_loader = loaders["test"]

        # Fetch one batch from train loader
        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape, (16, 3, 224, 224))
        self.assertEqual(labels.shape, (16,))
        self.assertEqual(labels.dtype, torch.int64)
        self.assertGreaterEqual(labels.min().item(), 0)
        self.assertLess(labels.max().item(), 7)

        # Fetch one batch from val loader
        val_images, val_labels = next(iter(val_loader))
        self.assertEqual(val_images.shape, (16, 3, 224, 224))
        self.assertEqual(val_labels.shape, (16,))

    def test_gpu_transfer(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA GPU not available in test environment")

        device = torch.device("cuda:0")
        dummy_batch = torch.randn(8, 3, 224, 224)
        dummy_labels = torch.randint(0, 7, (8,))

        gpu_batch = dummy_batch.to(device)
        gpu_labels = dummy_labels.to(device)

        self.assertTrue(gpu_batch.is_cuda)
        self.assertTrue(gpu_labels.is_cuda)
        self.assertTrue(str(gpu_batch.device).startswith("cuda"))
        
        # Simple tensor operation on RTX 3050 GPU
        result = gpu_batch * 2.0
        self.assertEqual(result.shape, (8, 3, 224, 224))
        self.assertEqual(result.device, gpu_batch.device)


if __name__ == "__main__":
    unittest.main()

