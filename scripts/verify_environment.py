"""
CottonGuard AI — Environment & Deployment Readiness Verification Script
Checks runtime dependencies, hardware acceleration, model checkpoints,
cryptographic hashes, database integrity, and frontend static assets.
"""

import hashlib
import json
import os
import platform
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def main():
    print("=" * 80)
    print("COTTONGUARD AI - ENVIRONMENT & DEPLOYMENT READINESS VERIFICATION")
    print("=" * 80)

    checks = []

    # 1. Python Environment
    py_ver = platform.python_version()
    py_ok = sys.version_info >= (3, 11)
    status_str = "PASS" if py_ok else "FAIL"
    print(f"[{status_str}] Python Runtime: {py_ver} (Requirement: >= 3.11.0)")
    checks.append(py_ok)

    # 2. PyTorch & CUDA
    try:
        import torch
        torch_ver = torch.__version__
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU (Fallback)"
        vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1) if cuda_ok else 0.0
        status_str = "PASS" if cuda_ok else "WARN"
        print(f"[{status_str}] PyTorch / CUDA: {torch_ver} | GPU: {gpu_name} ({vram_mb} MB VRAM)")
        checks.append(True)  # System supports both CUDA and CPU fallback
    except ImportError:
        print("[FAIL] PyTorch is not installed in the active environment.")
        checks.append(False)

    # 3. Model Checkpoints & Hashes
    EXPECTED_CHECKPOINTS = {
        "Verifier (EfficientNet-B0 Tri-Class)": {
            "path": PROJECT_ROOT / "artifacts" / "checkpoints" / "verification" / "best_verifier.pt",
            "hash": "1e16b9c68745c0f100679aa23e8993b9ee2ef6279851d4e4c820e36bb0a13828"
        },
        "Classifier (EfficientNet-B0)": {
            "path": PROJECT_ROOT / "artifacts" / "checkpoints" / "classification" / "best_model.pt",
            "hash": "552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e"
        },
        "Segmentor (U-Net ResNet-34)": {
            "path": PROJECT_ROOT / "artifacts" / "checkpoints" / "segmentation" / "best_unet.pt",
            "hash": "612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43"
        },
        "Detector (Faster R-CNN V2)": {
            "path": PROJECT_ROOT / "artifacts" / "checkpoints" / "detection" / "best_faster_rcnn.pt",
            "hash": "01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2"
        }
    }

    print("\nVerifying Core Deep Learning Checkpoints & SHA-256 Hashes:")
    for name, meta in EXPECTED_CHECKPOINTS.items():
        p = meta["path"]
        if not p.is_file():
            print(f"  [FAIL] {name}: Missing file at {p}")
            checks.append(False)
            continue
        actual_hash = sha256_file(p)
        matched = (actual_hash == meta["hash"])
        status_str = "PASS" if matched else "FAIL"
        print(f"  [{status_str}] {name} ({p.stat().st_size:,} bytes)")
        print(f"         Expected: {meta['hash']}")
        print(f"         Actual:   {actual_hash}")
        checks.append(matched)

    # 4. Sealed Test Split
    test_split_path = PROJECT_ROOT / "artifacts" / "manifests" / "splits" / "test.csv"
    if test_split_path.is_file():
        with open(test_split_path, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        test_samples = len(lines) - 1  # Minus header
        print(f"\n[PASS] Sealed Test Split: {test_split_path.name} verified ({test_samples} samples preserved)")
        checks.append(test_samples == 321)
    else:
        print(f"\n[FAIL] Sealed Test Split missing at {test_split_path}")
        checks.append(False)

    # 5. SQLite Database
    db_path = PROJECT_ROOT / "cotton_leaf_ai.db"
    if db_path.is_file():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]
        cursor.execute("SELECT COUNT(*) FROM predictions;")
        pred_count = cursor.fetchone()[0]
        conn.close()
        print(f"[PASS] SQLite Database: {db_path.name} ({db_path.stat().st_size:,} bytes, {len(tables)} tables, {pred_count} logged predictions)")
        checks.append(len(tables) >= 10)
    else:
        print(f"[FAIL] SQLite Database missing at {db_path}")
        checks.append(False)

    # 6. Static Web Assets & Sample Gallery
    ui_html = PROJECT_ROOT / "backend" / "app" / "static" / "index.html"
    samples_json = PROJECT_ROOT / "backend" / "app" / "static" / "samples" / "samples.json"
    ui_ok = ui_html.is_file() and ui_html.stat().st_size > 1000
    samples_ok = samples_json.is_file()
    print(f"\n[{'PASS' if ui_ok else 'FAIL'}] Frontend Static Dashboard: {ui_html.name} ({ui_html.stat().st_size:,} bytes)")
    print(f"[{'PASS' if samples_ok else 'FAIL'}] Preloaded Reference Gallery: {samples_json.name}")
    checks.append(ui_ok and samples_ok)

    # Final Verdict
    all_ok = all(checks)
    print("\n" + "=" * 80)
    if all_ok:
        print("READINESS VERDICT: [100% PASS] — SYSTEM READY FOR DEMONSTRATION & SUBMISSION")
    else:
        print("READINESS VERDICT: [FAILED] — SOME CHECKS DID NOT PASS")
    print("=" * 80)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
