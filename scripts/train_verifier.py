"""
Training & Calibration Script for Cotton Leaf Verification Model.

Model: EfficientNet-B0 (3 classes: 0=cotton_leaf, 1=non_cotton_leaf, 2=non_leaf)
Goal: Minimize False Acceptance Rate (FAR) for non-cotton/non-leaf images.
Hardware: CUDA AMP FP16 on NVIDIA RTX GPU.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Ensure project root is in sys.path
PROJECT_ROOT = Path("D:/Main_Project/cotton_leaf_ai")
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.verifier import CottonLeafVerifier, VERIFIER_CLASSES


class FoliarVerificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        label = int(row["label"])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback black image if corrupted
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, label, img_path


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.inference_mode():
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, targets)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs)

    total_loss /= len(dataloader.dataset)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Multi-class metrics
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    # Binary Cotton (Class 0) vs Non-Cotton (Classes 1 and 2)
    # Target: 1 if cotton, 0 if not cotton
    bin_targets = (all_targets == 0).astype(int)
    cotton_probs = all_probs[:, 0]
    bin_preds = (all_preds == 0).astype(int)

    tp = np.sum((bin_preds == 1) & (bin_targets == 1))
    fp = np.sum((bin_preds == 1) & (bin_targets == 0))
    tn = np.sum((bin_preds == 0) & (bin_targets == 0))
    fn = np.sum((bin_preds == 0) & (bin_targets == 1))

    # FAR: False Acceptance Rate = FP / (FP + TN) [negative = non-cotton]
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # FRR: False Rejection Rate = FN / (TP + FN) [positive = cotton]
    frr = fn / (tp + fn) if (tp + fn) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(bin_targets, cotton_probs)
    except Exception:
        roc_auc = 0.0

    cm = confusion_matrix(all_targets, all_preds)

    return {
        "loss": total_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "far": far,
        "frr": frr,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "targets": all_targets,
        "preds": all_preds,
        "probs": all_probs,
        "bin_counts": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }


def main():
    print("=" * 70)
    print("COTTONGUARD AI - COTTON LEAF VERIFIER TRAINING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_dir = PROJECT_ROOT / "data" / "verification" / "splits"
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = FoliarVerificationDataset(train_df, transform=train_transform)
    val_ds = FoliarVerificationDataset(val_df, transform=val_transform)
    test_ds = FoliarVerificationDataset(test_df, transform=val_transform)

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Class balance weighting
    class_counts = train_df["label"].value_counts().sort_index().values
    total_samples = len(train_df)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class counts: {class_counts}")
    print(f"Computed class weights: {class_weights_t.cpu().numpy().round(3)}")

    # Model
    model = CottonLeafVerifier(num_classes=3, dropout=0.30).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    epochs = 8
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    save_dir = PROJECT_ROOT / "artifacts" / "checkpoints" / "verification"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_verifier.pt"

    best_val_score = -1.0
    best_epoch = 0

    print("\nStarting Training...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            total_train += targets.size(0)

        scheduler.step()
        train_loss /= total_train
        train_acc = train_correct / total_train

        # Validation evaluation
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Objective score: prioritize high accuracy & macro F1 and low FAR
        # score = val_acc + val_f1 - 2 * far
        val_score = val_metrics["macro_f1"] - (1.5 * val_metrics["far"])

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"F1: {val_metrics['macro_f1']:.4f} FAR: {val_metrics['far']:.4f} FRR: {val_metrics['frr']:.4f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": {
                        "accuracy": val_metrics["accuracy"],
                        "macro_f1": val_metrics["macro_f1"],
                        "far": val_metrics["far"],
                        "frr": val_metrics["frr"],
                        "roc_auc": val_metrics["roc_auc"],
                    },
                    "classes": VERIFIER_CLASSES,
                },
                best_model_path,
            )

    train_duration = time.time() - start_time
    print(f"\nTraining completed in {train_duration:.1f}s. Best Epoch: {best_epoch}")

    # Load best model for rigorous testing
    print("\nLoading Best Model Checkpoint for Final Evaluation...")
    best_ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Final Test Set Evaluation
    test_metrics = evaluate(model, test_loader, criterion, device)
    test_cm = test_metrics["confusion_matrix"]

    print("\n" + "=" * 70)
    print("INDEPENDENT TEST SET RESULTS:")
    print("=" * 70)
    print(f"Test Accuracy:          {test_metrics['accuracy']*100:.2f}%")
    print(f"Test Macro F1:          {test_metrics['macro_f1']:.4f}")
    print(f"Test ROC-AUC:           {test_metrics['roc_auc']:.4f}")
    print(f"False Acceptance Rate:  {test_metrics['far']*100:.2f}%")
    print(f"False Rejection Rate:   {test_metrics['frr']*100:.2f}%")
    print(f"Confusion Matrix (Rows=True, Cols=Pred):")
    print(f"Classes: {VERIFIER_CLASSES}")
    for row in test_cm:
        print(f"  {row}")

    print("\nDetailed Classification Report:")
    print(classification_report(test_metrics["targets"], test_metrics["preds"], target_names=VERIFIER_CLASSES, digits=4))

    # Evaluate across thresholds
    print("\nThreshold Calibration Analysis (Cotton vs Non-Cotton):")
    bin_targets = (test_metrics["targets"] == 0).astype(int)
    cotton_probs = test_metrics["probs"][:, 0]

    calibration_results = []
    for th in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]:
        th_preds = (cotton_probs >= th).astype(int)
        th_tp = np.sum((th_preds == 1) & (bin_targets == 1))
        th_fp = np.sum((th_preds == 1) & (bin_targets == 0))
        th_tn = np.sum((th_preds == 0) & (bin_targets == 0))
        th_fn = np.sum((th_preds == 0) & (bin_targets == 1))

        th_far = th_fp / (th_fp + th_tn) if (th_fp + th_tn) > 0 else 0.0
        th_frr = th_fn / (th_tp + th_fn) if (th_tp + th_fn) > 0 else 0.0
        th_prec = th_tp / (th_tp + th_fp) if (th_tp + th_fp) > 0 else 0.0
        th_rec = th_tp / (th_tp + th_fn) if (th_tp + th_fn) > 0 else 0.0

        print(
            f"Threshold {th:.2f} -> FAR: {th_far*100:5.2f}% | FRR: {th_frr*100:5.2f}% | "
            f"Precision: {th_prec*100:5.2f}% | Recall: {th_rec*100:5.2f}% | "
            f"FP: {th_fp:2d}/{th_fp+th_tn} | FN: {th_fn:2d}/{th_tp+th_fn}"
        )
        calibration_results.append({
            "threshold": th,
            "far": float(th_far),
            "frr": float(th_frr),
            "precision": float(th_prec),
            "recall": float(th_rec),
            "tp": int(th_tp),
            "fp": int(th_fp),
            "tn": int(th_tn),
            "fn": int(th_fn),
        })

    # Compute Checkpoint Hash
    sha256 = compute_sha256(best_model_path)
    print(f"\nBest Checkpoint SHA-256: {sha256}")

    # Save summary report
    summary = {
        "model_architecture": "EfficientNet-B0 Tri-Class Foliar Verifier",
        "classes": VERIFIER_CLASSES,
        "best_epoch": best_epoch,
        "epochs_trained": epochs,
        "training_duration_sec": round(train_duration, 1),
        "checkpoint_path": str(best_model_path),
        "checkpoint_sha256": sha256,
        "test_metrics": {
            "accuracy": float(test_metrics["accuracy"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "roc_auc": float(test_metrics["roc_auc"]),
            "far": float(test_metrics["far"]),
            "frr": float(test_metrics["frr"]),
            "confusion_matrix": test_cm,
        },
        "threshold_calibration": calibration_results,
    }

    report_path = PROJECT_ROOT / "reports" / "cotton_leaf_verifier_evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluation report written to {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
