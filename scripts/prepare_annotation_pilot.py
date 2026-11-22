"""
Representative Split-Safe Annotation Pilot Generator (Phase 6).

Selects a deterministic, stratified pilot batch (default: 3 images per class = 21 images
across all 7 cotton classes: 2 from train.csv + 1 from validation.csv per class) while
enforcing strict leakage guards:
  - 0 images from artifacts/manifests/splits/test.csv
  - 0 images from Augmented Dataset
  - 0 duplicate group_id collisions across train, validation, or test splits

Exports read-only working image copies into `data/pilot_batch_01/images/` and writes
`artifacts/manifests/pilot_batch_manifest.csv` and `artifacts/manifests/pilot_batch_summary.json`
without touching the source dataset or split CSVs.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CLASS_NAMES


def _resolve_existing_source_path(raw_abs_path: str, rel_path: str) -> Path | None:
    p = Path(raw_abs_path)
    if p.is_file():
        return p
    win_root = Path(r"D:\Main_Project\dataset\SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection\Original Dataset")
    wsl_root = Path("/mnt/d/Main_Project/dataset/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection/Original Dataset")
    for base in (win_root, wsl_root):
        candidate = base / rel_path
        if candidate.is_file():
            return candidate
    return None


def generate_annotation_pilot(
    splits_dir: Path,
    pilot_dir: Path,
    manifest_csv_path: Path,
    summary_json_path: Path,
    train_per_class: int = 2,
    val_per_class: int = 1,
    seed: int = 42,
    copy_images: bool = True,
) -> Dict[str, Any]:
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "validation.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    test_groups: Set[str] = set(test_df["group_id"].dropna().astype(str))
    test_hashes: Set[str] = set(test_df["sha256"].dropna().astype(str))
    test_names: Set[str] = {Path(p).name for p in test_df["relative_path"].dropna().astype(str)}

    val_groups: Set[str] = set(val_df["group_id"].dropna().astype(str))
    train_groups: Set[str] = set(train_df["group_id"].dropna().astype(str))

    # Cross-split duplicate group exclusion check
    disallowed_cross_groups = test_groups | (train_groups & val_groups)

    selected_rows: List[Dict[str, Any]] = []
    used_groups: Set[str] = set()

    def _pick_from_df(df: pd.DataFrame, target_split: str, n_per_class: int) -> None:
        for cls_name in CLASS_NAMES:
            sub = df[df["class_folder_label"] == cls_name].copy()
            sub = sub.sample(frac=1.0, random_state=seed)
            picked = 0
            for _, row in sub.iterrows():
                if picked >= n_per_class:
                    break
                rel = str(row["relative_path"])
                fname = Path(rel).name
                sha = str(row["sha256"])
                grp = str(row["group_id"])
                abs_str = str(row.get("absolute_path", ""))

                if "augmented" in abs_str.lower() or "augmented" in rel.lower():
                    continue
                if fname in test_names or sha in test_hashes or grp in disallowed_cross_groups or grp in used_groups:
                    continue

                src_path = _resolve_existing_source_path(abs_str, rel)
                if copy_images and src_path is None:
                    continue

                used_groups.add(grp)
                selected_rows.append(
                    {
                        "filename": fname,
                        "relative_path": rel,
                        "class_label": cls_name,
                        "split": target_split,
                        "group_id": grp,
                        "sha256": sha,
                        "width": int(row.get("width", 0)),
                        "height": int(row.get("height", 0)),
                        "source_path": str(src_path or abs_str),
                        "requires_yolo_boxes": cls_name != "Healthy Leaf",
                        "requires_unet_leaf_mask": True,
                        "annotation_status": "PENDING_CVAT_ANNOTATION",
                    }
                )
                picked += 1

    _pick_from_df(train_df, "train", train_per_class)
    _pick_from_df(val_df, "validation", val_per_class)

    images_out_dir = pilot_dir / "images"
    if copy_images:
        if images_out_dir.exists():
            for existing in images_out_dir.iterdir():
                if existing.is_file():
                    existing.unlink()
        images_out_dir.mkdir(parents=True, exist_ok=True)
        for rec in selected_rows:
            src = Path(rec["source_path"])
            if src.is_file():
                shutil.copy2(src, images_out_dir / rec["filename"])

    out_df = pd.DataFrame(selected_rows)
    manifest_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(manifest_csv_path, index=False)

    per_class_counts = {
        cls_name: int(sum(1 for r in selected_rows if r["class_label"] == cls_name))
        for cls_name in CLASS_NAMES
    }
    split_counts = {
        "train": int(sum(1 for r in selected_rows if r["split"] == "train")),
        "validation": int(sum(1 for r in selected_rows if r["split"] == "validation")),
        "test": 0,
    }

    summary = {
        "pilot_batch_name": "CottonGuard_Phase6_Pilot_Batch_01",
        "total_pilot_images": len(selected_rows),
        "train_per_class_target": train_per_class,
        "val_per_class_target": val_per_class,
        "split_counts": split_counts,
        "per_class_counts": per_class_counts,
        "test_split_overlap_count": 0,
        "augmented_split_overlap_count": 0,
        "unique_group_count": len(used_groups),
        "images_copied_to": str(images_out_dir) if copy_images else None,
        "manifest_csv": str(manifest_csv_path),
        "training_note": (
            "Representative 21-image pilot strictly for CVAT label-schema QA, box/mask precision review, "
            "and export pipeline validation. Not sufficient for reliable production U-Net or YOLO training."
        ),
    }

    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare zero-leakage representative CVAT annotation pilot batch")
    parser.add_argument("--train-per-class", type=int, default=2)
    parser.add_argument("--val-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-copy", action="store_true")
    args = parser.parse_args()

    summary = generate_annotation_pilot(
        splits_dir=PROJECT_ROOT / "artifacts" / "manifests" / "splits",
        pilot_dir=PROJECT_ROOT / "data" / "pilot_batch_01",
        manifest_csv_path=PROJECT_ROOT / "artifacts" / "manifests" / "pilot_batch_manifest.csv",
        summary_json_path=PROJECT_ROOT / "artifacts" / "manifests" / "pilot_batch_summary.json",
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        seed=args.seed,
        copy_images=not args.no_copy,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
