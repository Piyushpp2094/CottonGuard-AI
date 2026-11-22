"""
DataLoader construction and batching pipeline for Cotton Leaf AI.

Provides utilities to:
- Compute class frequencies and inverse-frequency class weights for loss balancing.
- Create WeightedRandomSampler instances to balance minor classes (e.g. Leaf Variegation).
- Build PyTorch DataLoaders for training, validation, and testing with pin_memory and GPU optimization.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import CottonLeafDataset, CLASS_NAMES, CLASS_TO_IDX
from src.data.preprocessing import get_train_transforms, get_eval_transforms


def compute_class_weights(
    manifest_csv: Union[str, Path],
    num_classes: int = len(CLASS_NAMES),
    class_to_idx: Optional[Dict[str, int]] = None
) -> torch.Tensor:
    """
    Computes inverse-frequency class weights from a dataset manifest CSV.
    
    Formula: weight[c] = total_samples / (num_classes * count[c])
    Normalized so weights sum to num_classes.
    """
    if class_to_idx is None:
        class_to_idx = CLASS_TO_IDX

    df = pd.read_csv(manifest_csv)
    counts = np.zeros(num_classes, dtype=np.float32)

    for class_name, count in df["class_folder_label"].value_counts().items():
        if class_name in class_to_idx:
            idx = class_to_idx[class_name]
            counts[idx] = float(count)

    # Avoid zero division
    counts = np.maximum(counts, 1.0)
    total_samples = float(len(df))
    weights = total_samples / (num_classes * counts)

    # Normalize weights so that mean weight is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def create_weighted_sampler(
    manifest_csv: Union[str, Path],
    class_to_idx: Optional[Dict[str, int]] = None
) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler that samples each class with equal probability,
    counteracting severe class imbalances.
    """
    if class_to_idx is None:
        class_to_idx = CLASS_TO_IDX

    df = pd.read_csv(manifest_csv)
    class_counts = df["class_folder_label"].value_counts().to_dict()
    
    # Weight per class = 1.0 / count
    class_weights = {
        cls: 1.0 / count if count > 0 else 0.0
        for cls, count in class_counts.items()
    }
    
    # Assign weight to each individual sample
    sample_weights = [
        class_weights[label]
        for label in df["class_folder_label"]
    ]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def get_dataloaders(
    project_root: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
    use_weighted_sampler: bool = False,
    image_size: int = 224,
) -> Dict[str, DataLoader]:
    """
    Instantiates PyTorch DataLoaders for train, validation, and test splits.

    Args:
        project_root: Path to cotton_leaf_ai root directory.
        batch_size: Number of images per batch (default 32 for RTX 3050 6GB).
        num_workers: Number of subprocesses for data loading (default 2 on Windows).
        pin_memory: If True, copies Tensors to CUDA pinned memory before returning.
        use_weighted_sampler: If True, uses WeightedRandomSampler instead of shuffle=True.
        image_size: Target image dimensions (224 x 224).

    Returns:
        Dictionary with keys 'train', 'val', 'test', and each mapped to its DataLoader.
    """
    if project_root is None:
        # Default to cotton_leaf_ai root (two levels up from this file)
        project_root = Path(__file__).resolve().parents[2]
    else:
        project_root = Path(project_root)

    splits_dir = project_root / "artifacts" / "manifests" / "splits"
    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "validation.csv"
    test_csv = splits_dir / "test.csv"

    for split_name, csv_path in [("train", train_csv), ("val", val_csv), ("test", test_csv)]:
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing {split_name} split manifest: {csv_path}")

    train_transform = get_train_transforms(image_size=image_size)
    eval_transform = get_eval_transforms(image_size=image_size)

    train_dataset = CottonLeafDataset(csv_path=train_csv, transform=train_transform)
    val_dataset = CottonLeafDataset(csv_path=val_csv, transform=eval_transform)
    test_dataset = CottonLeafDataset(csv_path=test_csv, transform=eval_transform)

    if use_weighted_sampler:
        sampler = create_weighted_sampler(train_csv)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=False
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
