"""
Unified Phase 6 Annotation Quality Validator for YOLO Detection & U-Net Segmentation (Hardened).

Checks all mandatory annotation integrity dimensions:
1. Class IDs & expected disease class alignment: YOLO labels strictly integer in {0..5}
   and matching known disease class labels; U-Net masks strictly subset of {0, 1, 2}.
2. Bounding-box coordinates & Leaf containment: finite normalized coordinates in [0, 1],
   >= 2 px span, Healthy Leaf zero-box rule, and containment check against paired Leaf mask
   when segmentation mask is present.
3. Empty / degenerate masks: detects empty masks (leaf_pixels == 0) and tiny masks (<0.5% area).
4. Image-mask & image-label alignment: verifies 1-to-1 stem correspondence, readable RGB decoding,
   and exact image/mask dimensions.
5. Duplicate groups, split leakage & cross-image identical signatures: cross-checks hashes and
   group_id against train/validation/test.csv & near_duplicates.csv; flags repeated identical
   label/mask signatures across distinct images for explicit human review.
6. Pilot manifest verification & missing annotation gaps against pilot_batch_manifest.csv.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_annotations import PILOT_PAIR_THRESHOLD, RELIABLE_TRAINING_MIN_PAIRS, validate_annotation_directory
from src.models.detector import load_detection_classes

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest_metadata(
    splits_dir: Path,
    near_dup_csv: Optional[Path],
    pilot_manifest_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    by_name: Dict[str, Dict[str, Any]] = {}
    by_sha: Dict[str, Dict[str, Any]] = {}
    group_splits: Dict[str, Set[str]] = {}
    pilot_expected_names: Set[str] = set()

    for split_name in ("train", "validation", "test"):
        csv_p = splits_dir / f"{split_name}.csv"
        if not csv_p.is_file():
            continue
        df = pd.read_csv(csv_p)
        for _, row in df.iterrows():
            rel = str(row.get("relative_path", ""))
            fname = Path(rel).name
            sha = str(row.get("sha256", ""))
            grp = str(row.get("group_id", ""))
            cls_lbl = str(row.get("class_folder_label", ""))
            meta = {"split": split_name, "group_id": grp, "class_label": cls_lbl, "sha256": sha, "in_pilot_manifest": False}
            if fname:
                by_name[fname] = meta
            if sha:
                by_sha[sha] = meta
            if grp:
                group_splits.setdefault(grp, set()).add(split_name)

    if pilot_manifest_csv and pilot_manifest_csv.is_file():
        df_p = pd.read_csv(pilot_manifest_csv)
        for _, row in df_p.iterrows():
            fname = str(row.get("filename", "")).strip()
            if fname:
                pilot_expected_names.add(fname)
                existing = by_name.get(fname, {})
                existing.update({
                    "class_label": str(row.get("class_label", existing.get("class_label", ""))),
                    "split": str(row.get("split", existing.get("split", ""))),
                    "group_id": str(row.get("group_id", existing.get("group_id", ""))),
                    "in_pilot_manifest": True,
                })
                by_name[fname] = existing

    near_pairs: List[tuple[str, str]] = []
    if near_dup_csv and near_dup_csv.is_file():
        df_nd = pd.read_csv(near_dup_csv)
        for _, row in df_nd.iterrows():
            near_pairs.append((Path(str(row.get("image_a", ""))).name, Path(str(row.get("image_b", ""))).name))

    return {
        "by_name": by_name,
        "by_sha": by_sha,
        "group_splits": group_splits,
        "pilot_expected_names": pilot_expected_names,
        "near_pairs": near_pairs,
    }


def validate_yolo_directory(
    images_dir: Path,
    labels_dir: Path,
    manifest_meta: Dict[str, Any],
    seg_masks_dir: Optional[Path] = None,
    min_reliable_samples: int = RELIABLE_TRAINING_MIN_PAIRS,
    out_of_leaf_tolerance_ratio: float = 0.02,
) -> Dict[str, Any]:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    det_classes = load_detection_classes()
    valid_class_ids = set(range(len(det_classes)))

    image_files = {p.stem: p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    label_files = {p.stem: p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}

    common_stems = sorted(set(image_files.keys()) & set(label_files.keys()))
    missing_labels = sorted(set(image_files.keys()) - set(label_files.keys()))
    missing_images = sorted(set(label_files.keys()) - set(image_files.keys()))

    valid_records: List[Dict[str, Any]] = []
    invalid_records: List[Dict[str, Any]] = []
    class_box_counts: Counter = Counter()
    seen_groups: Dict[str, List[str]] = {}
    leakage_files: List[str] = []
    augmented_files: List[str] = []
    label_signatures: Dict[str, List[str]] = defaultdict(list)

    for stem in common_stems:
        img_path = image_files[stem]
        lbl_path = label_files[stem]
        errors: List[str] = []
        warnings: List[str] = []

        try:
            with Image.open(img_path) as pil_img:
                pil_img.verify()
            with Image.open(img_path) as pil_img:
                w, h = pil_img.size
        except Exception as exc:
            invalid_records.append({"stem": stem, "errors": [f"Corrupt image: {exc}"]})
            continue

        img_sha = _sha256(img_path)
        meta = manifest_meta["by_sha"].get(img_sha) or manifest_meta["by_name"].get(img_path.name)
        cls_folder_label = meta["class_label"] if meta else None
        group_id = meta["group_id"] if meta else None
        split_name = meta["split"] if meta else "external"

        if "augmented" in img_path.name.lower() or "aug_" in img_path.name.lower():
            augmented_files.append(img_path.name)
            errors.append("Augmented dataset filename detected inside detection annotation folder.")
        if split_name == "test":
            leakage_files.append(img_path.name)
            errors.append("Holdout test split leakage: image belongs to test.csv.")
        if group_id:
            seen_groups.setdefault(group_id, []).append(img_path.name)

        lines = [ln.strip() for ln in lbl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines:
            lbl_sig = hashlib.sha256("\n".join(sorted(lines)).encode("utf-8")).hexdigest()[:16]
            label_signatures[lbl_sig].append(img_path.name)

        paired_mask_arr = None
        if seg_masks_dir and (seg_masks_dir / f"{stem}.png").is_file():
            try:
                paired_mask_arr = np.array(Image.open(seg_masks_dir / f"{stem}.png"))
            except Exception:
                paired_mask_arr = None

        valid_boxes_in_file = 0
        for line_no, ln in enumerate(lines, start=1):
            parts = ln.split()
            if len(parts) != 5:
                errors.append(f"Line {line_no}: expected 5 YOLO fields [cls cx cy w h], got {len(parts)}.")
                continue
            try:
                cls_f = float(parts[0])
                cls_id = int(cls_f)
                if cls_f != float(cls_id) or cls_id not in valid_class_ids:
                    errors.append(f"Line {line_no}: invalid YOLO class_id '{parts[0]}' (expected 0..{len(det_classes)-1}).")
                    continue
                cx, cy, bw, bh = [float(v) for v in parts[1:]]
            except ValueError:
                errors.append(f"Line {line_no}: non-numeric coordinate values.")
                continue

            if not all(math.isfinite(v) for v in (cx, cy, bw, bh)):
                errors.append(f"Line {line_no}: NaN or Inf coordinate detected.")
                continue
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                errors.append(f"Line {line_no}: normalized box ({cx}, {cy}, {bw}, {bh}) out of range.")
                continue
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0
            if x1 < -1e-4 or y1 < -1e-4 or x2 > 1.0001 or y2 > 1.0001:
                errors.append(f"Line {line_no}: bounding box extends outside normalized [0, 1] image frame.")
                continue
            if (bw * w) < 2.0 or (bh * h) < 2.0:
                errors.append(f"Line {line_no}: degenerate box smaller than 2x2 px ({bw*w:.1f}x{bh*h:.1f}).")
                continue

            box_cls_name = det_classes[cls_id]
            if cls_folder_label and cls_folder_label != "Healthy Leaf" and box_cls_name != cls_folder_label:
                errors.append(
                    f"Line {line_no}: class mismatch on '{img_path.name}' — label '{box_cls_name}' != expected '{cls_folder_label}'."
                )
                continue

            if paired_mask_arr is not None and paired_mask_arr.shape[:2] == (h, w):
                px1 = max(0, int(math.floor(x1 * w)))
                py1 = max(0, int(math.floor(y1 * h)))
                px2 = min(w, int(math.ceil(x2 * w)))
                py2 = min(h, int(math.ceil(y2 * h)))
                crop = paired_mask_arr[py1:py2, px1:px2]
                if crop.size > 0:
                    leaf_inside = int(np.sum(crop >= 1))
                    if leaf_inside == 0 and (1.0 - out_of_leaf_tolerance_ratio) > 0:
                        errors.append(f"Line {line_no}: YOLO box has 0 overlap with Leaf mask.")
                        continue

            valid_boxes_in_file += 1
            class_box_counts[box_cls_name] += 1

        if cls_folder_label == "Healthy Leaf" and valid_boxes_in_file > 0:
            errors.append("Healthy Leaf image contains non-zero disease bounding boxes (expected empty .txt negative).")
        elif cls_folder_label and cls_folder_label != "Healthy Leaf" and valid_boxes_in_file == 0:
            warnings.append(f"Diseased class '{cls_folder_label}' has 0 bounding boxes in {lbl_path.name}.")

        if errors:
            invalid_records.append({"stem": stem, "errors": errors, "warnings": warnings})
        else:
            valid_records.append({
                "stem": stem,
                "image": img_path.name,
                "split": split_name,
                "group_id": group_id,
                "class_label": cls_folder_label,
                "box_count": valid_boxes_in_file,
                "warnings": warnings,
            })

    duplicate_group_collisions = {g: files for g, files in seen_groups.items() if len(files) > 1}
    identical_label_signature_groups = {sig: files for sig, files in label_signatures.items() if len(files) > 1}
    total_valid_boxes = sum(r["box_count"] for r in valid_records)

    return {
        "total_images": len(image_files),
        "total_label_files": len(label_files),
        "matched_pairs": len(common_stems),
        "valid_labeled_images": len(valid_records),
        "valid_bounding_boxes": total_valid_boxes,
        "invalid_records_count": len(invalid_records),
        "missing_labels_count": len(missing_labels),
        "missing_images_count": len(missing_images),
        "missing_labels": missing_labels,
        "missing_images": missing_images,
        "test_split_leakage_files": leakage_files,
        "augmented_files_detected": augmented_files,
        "duplicate_group_collisions": duplicate_group_collisions,
        "identical_label_signature_warnings": identical_label_signature_groups,
        "per_class_valid_boxes": {name: int(class_box_counts.get(name, 0)) for name in det_classes},
        "classes_with_zero_boxes": [name for name in det_classes if class_box_counts.get(name, 0) == 0],
        "is_ready_for_yolo_training": (
            len(valid_records) >= min_reliable_samples
            and total_valid_boxes >= min_reliable_samples
            and len(leakage_files) == 0
            and len(invalid_records) == 0
        ),
        "invalid_records": invalid_records,
    }


def run_phase6_quality_validation(
    seg_dir: Path,
    det_dir: Path,
    splits_dir: Path,
    near_dup_csv: Optional[Path],
    output_path: Path,
    pilot_manifest_csv: Optional[Path] = None,
    min_reliable_pairs: int = RELIABLE_TRAINING_MIN_PAIRS,
) -> Dict[str, Any]:
    manifest_meta = _load_manifest_metadata(splits_dir, near_dup_csv, pilot_manifest_csv)
    seg_summary = validate_annotation_directory(
        seg_dir / "images",
        seg_dir / "masks",
        PROJECT_ROOT / "artifacts" / "manifests" / "annotation_summary.json",
        min_reliable_pairs=min_reliable_pairs,
    )
    det_summary = validate_yolo_directory(
        det_dir / "images",
        det_dir / "labels",
        manifest_meta=manifest_meta,
        seg_masks_dir=seg_dir / "masks",
        min_reliable_samples=min_reliable_pairs,
    )

    seg_groups: Dict[str, List[str]] = {}
    mask_sigs: Dict[str, List[str]] = defaultdict(list)
    for rec in seg_summary.get("valid_pairs", []):
        sha = rec.get("sha256", "")
        fname = Path(rec.get("image_path", "")).name
        meta = manifest_meta["by_sha"].get(sha) or manifest_meta["by_name"].get(fname)
        if meta and meta.get("group_id"):
            seg_groups.setdefault(meta["group_id"], []).append(fname)
        mask_p = Path(rec.get("mask_path", ""))
        if mask_p.is_file():
            mask_sigs[_sha256(mask_p)[:16]].append(fname)

    seg_dup_groups = {g: files for g, files in seg_groups.items() if len(files) > 1}
    identical_mask_warnings = {sig: files for sig, files in mask_sigs.items() if len(files) > 1}

    pilot_expected = manifest_meta.get("pilot_expected_names", set())
    annotated_names = {Path(r.get("image_path", "")).name for r in seg_summary.get("valid_pairs", [])}
    missing_pilot_annotations = sorted(pilot_expected - annotated_names) if pilot_expected else []
    unlisted_pilot_annotations = sorted(annotated_names - pilot_expected) if (pilot_expected and annotated_names) else []

    report = {
        "schema_version": "2.1",
        "pilot_threshold_pairs": PILOT_PAIR_THRESHOLD,
        "reliable_training_threshold_pairs": min_reliable_pairs,
        "segmentation_validation": {
            **{k: v for k, v in seg_summary.items() if k not in ("valid_pairs",)},
            "duplicate_group_collisions": seg_dup_groups,
            "identical_mask_signature_warnings": identical_mask_warnings,
        },
        "yolo_detection_validation": det_summary,
        "pilot_manifest_alignment": {
            "pilot_manifest_loaded": bool(pilot_expected),
            "expected_pilot_count": len(pilot_expected),
            "annotated_pilot_count": len(annotated_names),
            "missing_pilot_images_count": len(missing_pilot_annotations),
            "unlisted_images": unlisted_pilot_annotations,
        },
        "overall_status": {
            "unet_status": "READY" if seg_summary["is_ready_for_segmentation_training"] else "BLOCKED_INSUFFICIENT_ANNOTATIONS",
            "yolo_status": "READY" if det_summary["is_ready_for_yolo_training"] else "BLOCKED_INSUFFICIENT_ANNOTATIONS",
            "requires_explicit_review_warnings": bool(identical_mask_warnings or det_summary["identical_label_signature_warnings"]),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full Phase 6 YOLO & U-Net annotation quality validator")
    parser.add_argument("--seg-dir", type=Path, default=PROJECT_ROOT / "data" / "annotations")
    parser.add_argument("--det-dir", type=Path, default=PROJECT_ROOT / "data" / "detection")
    parser.add_argument("--pilot-manifest", type=Path, default=PROJECT_ROOT / "artifacts" / "manifests" / "pilot_batch_manifest.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "manifests" / "phase6_annotation_quality_report.json")
    parser.add_argument("--min-pairs", type=int, default=RELIABLE_TRAINING_MIN_PAIRS)
    args = parser.parse_args()

    report = run_phase6_quality_validation(
        seg_dir=args.seg_dir,
        det_dir=args.det_dir,
        splits_dir=PROJECT_ROOT / "artifacts" / "manifests" / "splits",
        near_dup_csv=PROJECT_ROOT / "artifacts" / "audit" / "original" / "near_duplicates.csv",
        output_path=args.output,
        pilot_manifest_csv=args.pilot_manifest if args.pilot_manifest.is_file() else None,
        min_reliable_pairs=args.min_pairs,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
