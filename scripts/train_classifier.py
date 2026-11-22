"""
Training script for Cotton Leaf Disease Classification (Phase 3).

Trains an EfficientNet-B0 model with ImageNet transfer learning,
class-weighted CrossEntropyLoss, AdamW optimizer, cosine annealing schedule,
and early stopping based on validation macro F1-score.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CLASS_NAMES, CLASS_TO_IDX
from src.data.dataloader import compute_class_weights, get_dataloaders
from src.models.model_factory import build_classifier
from src.training.losses import get_classification_loss
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.trainer import ClassificationTrainer


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 Cotton Leaf Classifier")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size (32 optimized for RTX 3050 6GB)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate for AdamW")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for regularization")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs)")
    parser.add_argument("--dropout", type=float, default=0.3, help="Classifier head dropout rate")
    parser.add_argument("--use-sampler", action="store_true", help="Use WeightedRandomSampler to balance classes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts/checkpoints/classification", help="Checkpoint directory")
    parser.add_argument("--metrics-dir", type=str, default="artifacts/metrics", help="Directory to save training metrics")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[SETUP] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    checkpoint_dir = PROJECT_ROOT / args.checkpoint_dir
    metrics_dir = PROJECT_ROOT / args.metrics_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = PROJECT_ROOT / "artifacts" / "manifests" / "splits"
    train_csv = splits_dir / "train.csv"

    # Compute class weights for loss balancing
    class_weights = compute_class_weights(train_csv).to(device)
    print(f"[SETUP] Normalized Class Weights:\n" + "\n".join(f"  - {CLASS_NAMES[i]}: {w:.3f}" for i, w in enumerate(class_weights.tolist())))

    # Load DataLoaders
    loaders = get_dataloaders(
        project_root=PROJECT_ROOT,
        batch_size=args.batch_size,
        num_workers=2 if os.name != "nt" else 0,  # Safe default on Windows
        pin_memory=True,
        use_weighted_sampler=args.use_sampler,
        image_size=224
    )

    # Instantiate model
    print(f"\n[SETUP] Initializing EfficientNet-B0 with ImageNet pretrained weights...")
    model = build_classifier(
        model_name="efficientnet_b0",
        num_classes=len(CLASS_NAMES),
        pretrained=True,
        dropout=args.dropout,
        device=device
    )

    criterion = get_classification_loss(class_weights=class_weights, label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    early_stopping = EarlyStopping(patience=args.patience, mode="max", verbose=True)
    checkpoint_callback = ModelCheckpoint(checkpoint_dir=checkpoint_dir, filename="best_model.pt", mode="max")

    config = vars(args)

    trainer = ClassificationTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_callback=checkpoint_callback,
        early_stopping=early_stopping,
        use_amp=True
    )

    history = trainer.fit(num_epochs=args.epochs, config=config)

    # Save training history
    history_file = metrics_dir / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[COMPLETE] Saved training history to {history_file}")


if __name__ == "__main__":
    main()
