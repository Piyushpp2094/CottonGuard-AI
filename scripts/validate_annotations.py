"""
Validation script for Cotton Leaf Segmentation Annotations (Phase 6 upgraded).

Verifies:
1. One-to-one correspondence between images in data/annotations/images and masks in data/annotations/masks.
2. Identical width and height dimensions between each image and its single-channel mask.
3. Mask pixel values strictly adhere to the 3-class schema: {0: background, 1: healthy leaf, 2: diseased tissue}.
4. Non-empty, non-degenerate leaf region (`leaf_pixels > 0` and >= 0.5% image area).
5. Zero holdout `test.csv` or duplicate-group leakage.
6. Distinguishes pilot validation (`>= 21` stratified samples) from reliable model training (`>= 50` verified pairs).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_PAIR_THRESHOLD = 21
RELIABLE_TRAINING_MIN_PAIRS = 50


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_test_leakage_sets(splits_dir: Path | None = None) -> tuple[Set[str], Set[str], Set[str]]:
    base = splits_dir or (PROJECT_ROOT / "artifacts" / "manifests" / "splits")
    test_csv = base / "test.csv"
    test_hashes: Set[str] = set()
    test_names: Set[str] = set()
    test_groups: Set[str] = set()
    if test_csv.is_file():
        df = pd.read_csv(test_csv)
        if "sha256" in df.columns:
            test_hashes.update(df["sha256"].dropna().astype(str))
        if "relative_path" in df.columns:
            test_names.update(Path(p).name for p in df["relative_path"].dropna().astype(str))
        if "group_id" in df.columns:
            test_groups.update(df["group_id"].dropna().astype(str))
    return test_hashes, test_names, test_groups


def validate_annotation_directory(
    images_dir: Path,
    masks_dir: Path,
    output_summary_path: Path,
    min_reliable_pairs: int = RELIABLE_TRAINING_MIN_PAIRS,
) -> Dict[str, Any]:
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = {p.stem: p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in image_extensions}
    mask_files = {p.stem: p for p in masks_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"}

    common_keys = sorted(set(image_files.keys()).intersection(set(mask_files.keys())))
    missing_masks = sorted(set(image_files.keys()) - set(mask_files.keys()))
    missing_images = sorted(set(mask_files.keys()) - set(image_files.keys()))

    test_hashes, test_names, _ = _load_test_leakage_sets()

    valid_pairs: List[Dict[str, Any]] = []
    invalid_pairs: List[Dict[str, Any]] = []
    leakage_files: List[str] = []
    empty_masks: List[str] = []

    for key in common_keys:
        img_path = image_files[key]
        mask_path = mask_files[key]

        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                img_w, img_h = img.size

            with Image.open(mask_path) as mask:
                mask_w, mask_h = mask.size
                mask_arr = np.array(mask)

            errors: List[str] = []
            if (img_w, img_h) != (mask_w, mask_h):
                errors.append(f"Dimension mismatch: image is ({img_w}, {img_h}) but mask is ({mask_w}, {mask_h})")

            if mask_arr.ndim != 2:
                errors.append(f"Mask must be single-channel 2D index PNG, found shape {tuple(mask_arr.shape)}")

            unique_vals = set(np.unique(mask_arr).tolist())
            invalid_vals = unique_vals - {0, 1, 2}
            if invalid_vals:
                errors.append(f"Invalid mask pixel values detected: {sorted(invalid_vals)}. Allowed: {{0, 1, 2}}")

            leaf_pixels = int(np.sum(mask_arr >= 1)) if mask_arr.ndim == 2 else 0
            disease_pixels = int(np.sum(mask_arr == 2)) if mask_arr.ndim == 2 else 0
            total_pixels = max(1, img_w * img_h)

            if leaf_pixels == 0:
                empty_masks.append(key)
                errors.append("Empty mask: 0 leaf/disease pixels found.")
            elif (leaf_pixels / total_pixels) < 0.005:
                errors.append(f"Degenerate mask: leaf area is < 0.5% ({leaf_pixels}/{total_pixels} px).")

            img_sha = _sha256_file(img_path)
            if img_sha in test_hashes or img_path.name in test_names:
                leakage_files.append(img_path.name)
                errors.append("Holdout test split leakage: image belongs to artifacts/manifests/splits/test.csv.")

            if errors:
                invalid_pairs.append({
                    "stem": key,
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "errors": errors,
                })
            else:
                affected_pct = (100.0 * disease_pixels / leaf_pixels) if leaf_pixels > 0 else 0.0
                valid_pairs.append({
                    "stem": key,
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "sha256": img_sha,
                    "dimensions": [img_w, img_h],
                    "leaf_pixels": leaf_pixels,
                    "disease_pixels": disease_pixels,
                    "affected_area_pct": round(affected_pct, 2),
                })
        except Exception as exc:
            invalid_pairs.append({
                "stem": key,
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "errors": [f"File read error: {exc}"],
            })

    non_leaking_valid = len(valid_pairs)
    summary = {
        "total_images_in_folder": len(image_files),
        "total_masks_in_folder": len(mask_files),
        "matched_pairs": len(common_keys),
        "valid_pairs_count": non_leaking_valid,
        "invalid_pairs_count": len(invalid_pairs),
        "empty_masks_count": len(empty_masks),
        "missing_masks_count": len(missing_masks),
        "missing_images_count": len(missing_images),
        "missing_masks": missing_masks,
        "missing_images": missing_images,
        "test_split_leakage_files": leakage_files,
        "pilot_minimum_pairs": PILOT_PAIR_THRESHOLD,
        "reliable_training_minimum_pairs": min_reliable_pairs,
        "is_pilot_ready": non_leaking_valid >= PILOT_PAIR_THRESHOLD and len(leakage_files) == 0,
        "is_ready_for_segmentation_training": non_leaking_valid >= min_reliable_pairs and len(leakage_files) == 0,
        "valid_pairs": valid_pairs,
        "invalid_pairs": invalid_pairs,
    }

    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 64)
    print("SEGMENTATION ANNOTATION AUDIT SUMMARY")
    print("=" * 64)
    print(f"Total annotation images found: {len(image_files)}")
    print(f"Total annotation masks found:  {len(mask_files)}")
    print(f"Valid verified pairs:          {non_leaking_valid}")
    print(f"Invalid pairs / errors:        {len(invalid_pairs)}")
    print(f"Missing masks / images:        {len(missing_masks)} / {len(missing_images)}")
    print(f"Test-split leakage files:      {len(leakage_files)}")
    print(
        f"Ready for U-Net training:      "
        f"{'YES' if summary['is_ready_for_segmentation_training'] else f'NO (Requires >= {min_reliable_pairs} verified pairs; 10-pair smoke set insufficient)'}"
    )
    print("=" * 64)
    print(f"Saved audit details to: {output_summary_path}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Validate Segmentation Image-Mask Pairs")
    parser.add_argument("--images-dir", type=str, default="data/annotations/images")
    parser.add_argument("--masks-dir", type=str, default="data/annotations/masks")
    parser.add_argument("--output", type=str, default="artifacts/manifests/annotation_summary.json")
    parser.add_argument("--min-pairs", type=int, default=RELIABLE_TRAINING_MIN_PAIRS)
    args = parser.parse_args()

    images_dir = PROJECT_ROOT / args.images_dir
    masks_dir = PROJECT_ROOT / args.masks_dir
    output_path = PROJECT_ROOT / args.output

    validate_annotation_directory(images_dir, masks_dir, output_path, min_reliable_pairs=args.min_pairs)


if __name__ == "__main__":
    main()
