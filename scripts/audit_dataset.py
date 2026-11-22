#!/usr/bin/env python3
"""Read-only audit for the cotton leaf primary classification dataset.

This script NEVER deletes, moves, renames, edits, or rewrites source images.
It writes CSV/JSON reports only to the selected output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageFile, UnidentifiedImageError
    import imagehash
except ImportError as exc:  # clear actionable message
    raise SystemExit(
        "Missing audit dependency. Activate the project environment and run: "
        "python -m pip install -r requirements.txt"
    ) from exc

ImageFile.LOAD_TRUNCATED_IMAGES = False

EXPECTED_CLASSES = [
    "Bacterial Blight",
    "Curl Virus",
    "Healthy Leaf",
    "Herbicide Growth Damage",
    "Leaf Hopper Jassids",
    "Leaf Redding",
    "Leaf Variegation",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only cotton dataset audit")
    parser.add_argument("--dataset-root", required=True, type=Path,
                        help="Path to ...\\dataset\\Original Dataset")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Separate report directory; must not be inside dataset root")
    parser.add_argument("--near-duplicate-threshold", type=int, default=4,
                        help="Perceptual-hash Hamming distance threshold (default: 4; range 0-64)")
    parser.add_argument("--min-width", type=int, default=64,
                        help="Flag images narrower than this (default: 64 pixels)")
    parser.add_argument("--min-height", type=int, default=64,
                        help="Flag images shorter than this (default: 64 pixels)")
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    threshold = args.near_duplicate_threshold

    if not dataset_root.exists() or not dataset_root.is_dir():
        print(f"ERROR: Dataset folder does not exist: {dataset_root}", file=sys.stderr)
        return 2
    if not 0 <= threshold <= 64:
        parser.error("--near-duplicate-threshold must be between 0 and 64")
    if output_dir == dataset_root or dataset_root in output_dir.parents:
        print("ERROR: Output directory must be outside the source dataset folder.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    class_dirs = {p.name for p in dataset_root.iterdir() if p.is_dir()}
    missing_classes = sorted(set(EXPECTED_CLASSES) - class_dirs)
    unexpected_classes = sorted(class_dirs - set(EXPECTED_CLASSES))

    all_files = sorted(p for p in dataset_root.rglob("*") if p.is_file())
    inventory: list[dict[str, Any]] = []
    valid_for_phash: list[dict[str, Any]] = []
    exact_groups: dict[str, list[str]] = defaultdict(list)
    class_counts: Counter[str] = Counter()
    class_valid: Counter[str] = Counter()
    review_rows: list[dict[str, Any]] = []

    print(f"Dataset root : {dataset_root}")
    print(f"Output folder: {output_dir}")
    print(f"Files found  : {len(all_files)}")
    print("Reading images. Source files will not be modified...\n")

    for index, path in enumerate(all_files, start=1):
        relative = safe_relative(path, dataset_root)
        parts = Path(relative).parts
        class_name = parts[0] if len(parts) >= 2 else "__UNCLASSIFIED__"
        extension = path.suffix.lower()
        is_image_candidate = extension in IMAGE_EXTENSIONS
        row: dict[str, Any] = {
            "relative_path": relative,
            "class_folder_label": class_name,
            "extension": extension,
            "file_size_bytes": path.stat().st_size,
            "is_image_extension": is_image_candidate,
            "decode_status": "NOT_CHECKED_NON_IMAGE",
            "width": "",
            "height": "",
            "mode": "",
            "sha256": "",
            "phash": "",
            "review_flags": "",
        }
        flags: list[str] = []

        if class_name == "__UNCLASSIFIED__":
            flags.append("FILE_DIRECTLY_UNDER_DATASET_ROOT_NO_CLASS_FOLDER")
        elif class_name not in EXPECTED_CLASSES:
            flags.append("UNEXPECTED_CLASS_FOLDER")

        if is_image_candidate:
            class_counts[class_name] += 1
            try:
                # verify() checks file integrity; reopen and load() forces pixel decoding.
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    image.load()
                    width, height = image.size
                    mode = image.mode
                    phash_value = str(imagehash.phash(image.convert("RGB")))
                file_hash = sha256_file(path)
                row.update({
                    "decode_status": "OK",
                    "width": width,
                    "height": height,
                    "mode": mode,
                    "sha256": file_hash,
                    "phash": phash_value,
                })
                class_valid[class_name] += 1
                exact_groups[file_hash].append(relative)
                valid_for_phash.append({
                    "relative_path": relative,
                    "class_folder_label": class_name,
                    "sha256": file_hash,
                    "phash_int": int(phash_value, 16),
                    "width": width,
                    "height": height,
                })
                if width < args.min_width or height < args.min_height:
                    flags.append("LOW_RESOLUTION_REVIEW")
                if width == height:
                    flags.append("SQUARE_IMAGE_REVIEW_CONTEXT")
                if width > 0 and height > 0 and max(width / height, height / width) > 4:
                    flags.append("EXTREME_ASPECT_RATIO_REVIEW")
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
                row["decode_status"] = "CORRUPT_OR_UNREADABLE"
                flags.append("CORRUPT_OR_UNREADABLE_IMAGE")
                row["decode_error"] = f"{type(exc).__name__}: {exc}"[:500]
            except Exception as exc:  # retain audit progress for an unusual file
                row["decode_status"] = "ERROR_DURING_AUDIT"
                flags.append("AUDIT_ERROR_REVIEW")
                row["decode_error"] = f"{type(exc).__name__}: {exc}"[:500]
        else:
            flags.append("NON_IMAGE_FILE_IN_DATASET")

        row["review_flags"] = ";".join(flags)
        inventory.append(row)
        if flags:
            review_rows.append({
                "relative_path": relative,
                "class_folder_label": class_name,
                "decode_status": row["decode_status"],
                "review_flags": ";".join(flags),
                "review_status": "PENDING_HUMAN_REVIEW",
                "reviewed_label": "",
                "review_note": "",
            })
        if index % 200 == 0 or index == len(all_files):
            print(f"Processed {index}/{len(all_files)} files")

    # Exact duplicates: report paths only; never remove any file.
    exact_rows: list[dict[str, Any]] = []
    exact_group_id = 0
    for digest, paths in sorted(exact_groups.items()):
        if len(paths) > 1:
            exact_group_id += 1
            for relative in paths:
                exact_rows.append({
                    "duplicate_group": exact_group_id,
                    "sha256": digest,
                    "relative_path": relative,
                    "group_size": len(paths),
                    "action": "REVIEW_ONLY_NO_FILES_CHANGED",
                })

    # Perceptual duplicates: pairwise pHash distance; exact byte duplicates omitted here.
    near_rows: list[dict[str, Any]] = []
    n = len(valid_for_phash)
    print(f"\nChecking perceptual similarity among {n} readable images...")
    near_pair_count = 0
    for i in range(n):
        left = valid_for_phash[i]
        for j in range(i + 1, n):
            right = valid_for_phash[j]
            if left["sha256"] == right["sha256"]:
                continue
            distance = (left["phash_int"] ^ right["phash_int"]).bit_count()
            if distance <= threshold:
                near_pair_count += 1
                near_rows.append({
                    "image_a": left["relative_path"],
                    "label_a": left["class_folder_label"],
                    "image_b": right["relative_path"],
                    "label_b": right["class_folder_label"],
                    "phash_hamming_distance": distance,
                    "same_class_folder": left["class_folder_label"] == right["class_folder_label"],
                    "action": "VISUAL_REVIEW_ONLY_NO_FILES_CHANGED",
                })
        if (i + 1) % 200 == 0 or i + 1 == n:
            print(f"Compared {i + 1}/{n} image rows")

    # Add exact/near duplicate candidates to review queue, retaining all source images.
    already_queued = {r["relative_path"] for r in review_rows}
    for row in exact_rows:
        relative = row["relative_path"]
        if relative not in already_queued:
            review_rows.append({
                "relative_path": relative,
                "class_folder_label": Path(relative).parts[0],
                "decode_status": "OK",
                "review_flags": "EXACT_DUPLICATE_CANDIDATE",
                "review_status": "PENDING_HUMAN_REVIEW",
                "reviewed_label": "",
                "review_note": "",
            })
            already_queued.add(relative)
    for row in near_rows:
        for key, label_key in (("image_a", "label_a"), ("image_b", "label_b")):
            relative = row[key]
            if relative not in already_queued:
                review_rows.append({
                    "relative_path": relative,
                    "class_folder_label": row[label_key],
                    "decode_status": "OK",
                    "review_flags": "NEAR_DUPLICATE_CANDIDATE",
                    "review_status": "PENDING_HUMAN_REVIEW",
                    "reviewed_label": "",
                    "review_note": "",
                })
                already_queued.add(relative)

    # Human review template includes all readable image candidates. No automatic relabeling.
    label_template = []
    for row in inventory:
        if row["decode_status"] == "OK":
            label_template.append({
                "relative_path": row["relative_path"],
                "folder_label": row["class_folder_label"],
                "reviewed_label": "",
                "label_correct": "",
                "reviewer": "",
                "review_note": "",
                "review_status": "NOT_REVIEWED",
            })

    summary_rows = []
    for class_name in EXPECTED_CLASSES + sorted(set(class_counts) - set(EXPECTED_CLASSES)):
        summary_rows.append({
            "class": class_name,
            "image_files": class_counts[class_name],
            "readable_images": class_valid[class_name],
            "unreadable_images": class_counts[class_name] - class_valid[class_name],
        })

    write_csv(output_dir / "image_inventory.csv", inventory,
              ["relative_path", "class_folder_label", "extension", "file_size_bytes",
               "is_image_extension", "decode_status", "width", "height", "mode",
               "sha256", "phash", "review_flags", "decode_error"])
    write_csv(output_dir / "class_summary.csv", summary_rows,
              ["class", "image_files", "readable_images", "unreadable_images"])
    write_csv(output_dir / "exact_duplicates.csv", exact_rows,
              ["duplicate_group", "sha256", "relative_path", "group_size", "action"])
    write_csv(output_dir / "near_duplicates.csv", near_rows,
              ["image_a", "label_a", "image_b", "label_b", "phash_hamming_distance",
               "same_class_folder", "action"])
    write_csv(output_dir / "review_queue.csv", review_rows,
              ["relative_path", "class_folder_label", "decode_status", "review_flags",
               "review_status", "reviewed_label", "review_note"])
    write_csv(output_dir / "label_review_template.csv", label_template,
              ["relative_path", "folder_label", "reviewed_label", "label_correct",
               "reviewer", "review_note", "review_status"])

    total_image_files = sum(class_counts.values())
    readable_count = sum(class_valid.values())
    summary = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "source_images_modified": False,
        "source_images_deleted": 0,
        "source_images_moved": 0,
        "source_images_renamed": 0,
        "total_files_discovered": len(all_files),
        "image_files_discovered": total_image_files,
        "readable_images": readable_count,
        "unreadable_images": total_image_files - readable_count,
        "exact_duplicate_groups": exact_group_id,
        "exact_duplicate_file_rows": len(exact_rows),
        "near_duplicate_pairs": near_pair_count,
        "near_duplicate_threshold": threshold,
        "review_queue_rows": len(review_rows),
        "label_review_template_rows": len(label_template),
        "expected_classes": EXPECTED_CLASSES,
        "missing_class_folders": missing_classes,
        "unexpected_class_folders": unexpected_classes,
        "class_counts": dict(class_counts),
        "readable_class_counts": dict(class_valid),
        "important_note": (
            "Duplicate and label flags are review suggestions only. This audit cannot prove "
            "biological label correctness. No source images were changed."
        ),
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nAUDIT COMPLETE")
    print(f"Images discovered : {total_image_files}")
    print(f"Readable images   : {readable_count}")
    print(f"Unreadable images : {total_image_files - readable_count}")
    print(f"Exact dup groups  : {exact_group_id}")
    print(f"Near dup pairs    : {near_pair_count}")
    print(f"Reports saved to  : {output_dir}")
    print("Source dataset    : READ-ONLY; no files changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
