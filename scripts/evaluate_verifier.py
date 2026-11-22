"""
Independent Evaluation & Calibration Script for Cotton Leaf Verifier Model.
Evaluates best_verifier.pt on the 637-image independent test set.
"""

import hashlib
import json
from pathlib import Path
import sys

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
        except Exception:
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


def evaluate_test_set():
    print("=" * 70)
    print("COTTON LEAF VERIFIER - INDEPENDENT TEST SET EVALUATION")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = PROJECT_ROOT / "data" / "verification" / "splits"
    test_df = pd.read_csv(data_dir / "test.csv")
    print(f"Test samples: {len(test_df)}")

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_ds = FoliarVerificationDataset(test_df, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    best_model_path = PROJECT_ROOT / "artifacts" / "checkpoints" / "verification" / "best_verifier.pt"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {best_model_path}")

    model = CottonLeafVerifier(num_classes=3)
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.inference_mode():
        for images, targets, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Multi-class metrics
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")

    # Binary Cotton (Class 0) vs Non-Cotton (Classes 1 and 2)
    bin_targets = (all_targets == 0).astype(int)
    cotton_probs = all_probs[:, 0]
    bin_preds = (all_preds == 0).astype(int)

    tp = np.sum((bin_preds == 1) & (bin_targets == 1))
    fp = np.sum((bin_preds == 1) & (bin_targets == 0))
    tn = np.sum((bin_preds == 0) & (bin_targets == 0))
    fn = np.sum((bin_preds == 0) & (bin_targets == 1))

    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    frr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(bin_targets, cotton_probs))
    except Exception:
        roc_auc = 0.0

    cm = confusion_matrix(all_targets, all_preds).tolist()
    report_dict = classification_report(all_targets, all_preds, target_names=VERIFIER_CLASSES, digits=4, output_dict=True)

    print("\n" + "=" * 70)
    print("INDEPENDENT TEST SET RESULTS:")
    print("=" * 70)
    print(f"Overall Accuracy:       {acc * 100:.2f}%")
    print(f"Macro F1 Score:         {macro_f1:.4f}")
    print(f"ROC-AUC (Cotton vs Non): {roc_auc:.4f}")
    print(f"False Acceptance Rate:  {far * 100:.2f}% (FP={fp}, TN={tn})")
    print(f"False Rejection Rate:   {frr * 100:.2f}% (FN={fn}, TP={tp})")
    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    print(f"Classes: {VERIFIER_CLASSES}")
    for i, row in enumerate(cm):
        print(f"  {VERIFIER_CLASSES[i]:16s}: {row}")

    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=VERIFIER_CLASSES, digits=4))

    # Threshold calibration
    print("\nThreshold Calibration Analysis (Cotton vs Non-Cotton):")
    calibration_results = []
    for th in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        th_preds = (cotton_probs >= th).astype(int)
        th_tp = np.sum((th_preds == 1) & (bin_targets == 1))
        th_fp = np.sum((th_preds == 1) & (bin_targets == 0))
        th_tn = np.sum((th_preds == 0) & (bin_targets == 0))
        th_fn = np.sum((th_preds == 0) & (bin_targets == 1))

        th_far = float(th_fp / (th_fp + th_tn)) if (th_fp + th_tn) > 0 else 0.0
        th_frr = float(th_fn / (th_tp + th_fn)) if (th_tp + th_fn) > 0 else 0.0
        th_prec = float(th_tp / (th_tp + th_fp)) if (th_tp + th_fp) > 0 else 0.0
        th_rec = float(th_tp / (th_tp + th_fn)) if (th_tp + th_fn) > 0 else 0.0

        print(
            f"Threshold {th:.2f} -> FAR: {th_far*100:5.2f}% | FRR: {th_frr*100:5.2f}% | "
            f"Precision: {th_prec*100:5.2f}% | Recall: {th_rec*100:5.2f}% | "
            f"FP: {th_fp:2d}/{th_fp+th_tn} | FN: {th_fn:2d}/{th_tp+th_fn}"
        )
        calibration_results.append({
            "threshold": th,
            "far": th_far,
            "frr": th_frr,
            "precision": th_prec,
            "recall": th_rec,
            "tp": int(th_tp),
            "fp": int(th_fp),
            "tn": int(th_tn),
            "fn": int(th_fn),
        })

    sha256 = compute_sha256(best_model_path)
    print(f"\nCheckpoint SHA-256: {sha256}")

    summary = {
        "model_architecture": "EfficientNet-B0 Tri-Class Foliar Verifier",
        "classes": VERIFIER_CLASSES,
        "checkpoint_path": str(best_model_path),
        "checkpoint_sha256": sha256,
        "test_dataset_size": len(test_df),
        "test_metrics": {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "roc_auc": float(roc_auc),
            "far": float(far),
            "frr": float(frr),
            "confusion_matrix": cm,
            "classification_report": report_dict,
        },
        "threshold_calibration": calibration_results,
    }

    report_path = PROJECT_ROOT / "reports" / "cotton_leaf_verifier_evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved evaluation report to: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_test_set()
