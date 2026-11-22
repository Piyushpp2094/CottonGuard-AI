"""
Create a clean image manifest and leakage-aware train/validation/test splits.

This script:
- Reads the audit image_inventory.csv.
- Reads near_duplicate_review_final.csv.
- Groups only pairs the user confirmed are the same photograph.
- Keeps visually different images separate.
- Creates approximate stratified 70/15/15 splits at GROUP level.
- Never modifies, moves, renames, or deletes source images.

Run from the cotton_leaf_ai project root:
python scripts\create_manifest.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DEFAULT_DATASET_ROOT = Path(
    r"D:\Main_Project\dataset\SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection\Original Dataset"
)
DEFAULT_AUDIT_DIR = Path("artifacts/audit/original")
DEFAULT_OUTPUT_DIR = Path("artifacts/manifests")

SAME_IMAGE_DECISIONS = {
    "Same image; labels agree",
    "Same image; labels conflict",
}
DIFFERENT_IMAGE_DECISIONS = {
    "Different images; labels appear correct",
}


class UnionFind:
    def __init__(self, items):
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {args.dataset_root}")

    inventory_path = args.audit_dir / "image_inventory.csv"
    review_path = args.audit_dir / "near_duplicate_review_final.csv"

    if not inventory_path.is_file():
        raise FileNotFoundError(f"Missing inventory: {inventory_path}")
    if not review_path.is_file():
        raise FileNotFoundError(f"Missing reviewed pairs: {review_path}")

    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("Train, validation, and test ratios must be positive and sum to 1.")

    inventory = pd.read_csv(inventory_path)
    required_inventory = {"relative_path", "class_folder_label", "decode_status", "sha256"}
    missing = required_inventory - set(inventory.columns)
    if missing:
        raise ValueError(f"Inventory is missing columns: {sorted(missing)}")

    # Keep only readable image records, and independently verify each source path.
    inventory = inventory[inventory["decode_status"].astype(str).str.upper() == "OK"].copy()
    inventory["relative_path"] = inventory["relative_path"].astype(str)
    inventory["class_folder_label"] = inventory["class_folder_label"].astype(str)

    if inventory["relative_path"].duplicated().any():
        duplicates = inventory.loc[
            inventory["relative_path"].duplicated(keep=False), "relative_path"
        ].tolist()
        raise ValueError(f"Duplicate relative_path entries in inventory: {duplicates[:10]}")

    missing_files = [
        rel for rel in inventory["relative_path"]
        if not (args.dataset_root / Path(rel)).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} inventory files are missing from dataset root. "
            f"Examples: {missing_files[:10]}"
        )

    # The current audit reports zero exact duplicate groups. Still guard against
    # repeated hashes so the split cannot accidentally leak identical file bytes.
    repeated_hashes = inventory[
        inventory["sha256"].notna()
        & inventory["sha256"].astype(str).ne("")
    ].groupby("sha256")["relative_path"].nunique()
    repeated_hashes = repeated_hashes[repeated_hashes > 1]
    if not repeated_hashes.empty:
        raise ValueError(
            "Exact duplicate SHA-256 hashes remain in the inventory. "
            "Review them before splitting. Example hashes: "
            f"{repeated_hashes.index[:5].tolist()}"
        )

    reviews = pd.read_csv(review_path)
    required_reviews = {"image_a", "image_b", "decision", "review_status"}
    missing = required_reviews - set(reviews.columns)
    if missing:
        raise ValueError(f"Review file is missing columns: {sorted(missing)}")

    known_paths = set(inventory["relative_path"])
    union_find = UnionFind(known_paths)

    for row in reviews.itertuples(index=False):
        image_a = str(getattr(row, "image_a"))
        image_b = str(getattr(row, "image_b"))
        decision = str(getattr(row, "decision")).strip()
        review_status = str(getattr(row, "review_status")).strip().lower()

        if image_a not in known_paths or image_b not in known_paths:
            raise ValueError(
                f"Reviewed pair is not present in the current inventory: {image_a} | {image_b}"
            )
        if review_status != "reviewed":
            raise ValueError(f"Pair has not been marked Reviewed: {image_a} | {image_b}")

        if decision in SAME_IMAGE_DECISIONS:
            union_find.union(image_a, image_b)
        elif decision in DIFFERENT_IMAGE_DECISIONS:
            # Deliberately do not group visually different images.
            continue
        else:
            raise ValueError(
                f"Unrecognized or unresolved decision '{decision}' for "
                f"{image_a} | {image_b}. Finalize the review CSV first."
            )

    # Assign stable group IDs based on sorted member paths.
    components = {}
    for path in sorted(known_paths):
        root = union_find.find(path)
        components.setdefault(root, []).append(path)

    group_for_path = {}
    group_rows = []
    for members in sorted(components.values(), key=lambda x: x[0]):
        group_id = "group_" + str(len(group_rows) + 1).zfill(5)
        labels = set(
            inventory.loc[
                inventory["relative_path"].isin(members), "class_folder_label"
            ].tolist()
        )
        if len(labels) != 1:
            raise ValueError(
                "A confirmed same-photograph group contains conflicting class labels: "
                f"{members} -> {sorted(labels)}. Resolve the label conflict before training."
            )
        for member in members:
            group_for_path[member] = group_id
        group_rows.append({
            "group_id": group_id,
            "group_size": len(members),
            "class_label": next(iter(labels)),
            "members": " | ".join(members),
        })

    manifest = inventory.copy()
    manifest["group_id"] = manifest["relative_path"].map(group_for_path)
    manifest["absolute_path"] = manifest["relative_path"].map(
        lambda rel: str(args.dataset_root / Path(rel))
    )

    # Split groups, not individual images, to prevent confirmed same-photo
    # variants from crossing train/validation/test boundaries.
    group_table = pd.DataFrame(group_rows)
    if group_table["class_label"].value_counts().min() < 3:
        raise ValueError("At least one class has fewer than 3 groups; cannot stratify safely.")

    train_groups, temporary_groups = train_test_split(
        group_table,
        test_size=(args.val_ratio + args.test_ratio),
        random_state=args.seed,
        stratify=group_table["class_label"],
    )

    relative_test_share = args.test_ratio / (args.val_ratio + args.test_ratio)
    val_groups, test_groups = train_test_split(
        temporary_groups,
        test_size=relative_test_share,
        random_state=args.seed,
        stratify=temporary_groups["class_label"],
    )

    split_by_group = {}
    for split_name, frame in (
        ("train", train_groups),
        ("validation", val_groups),
        ("test", test_groups),
    ):
        for group_id in frame["group_id"]:
            split_by_group[group_id] = split_name

    manifest["split"] = manifest["group_id"].map(split_by_group)

    # Verify group isolation.
    split_counts_per_group = manifest.groupby("group_id")["split"].nunique()
    if (split_counts_per_group > 1).any():
        raise RuntimeError("Leakage check failed: a group appears in multiple splits.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = args.output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    # Keep a clean, complete manifest plus convenient split CSVs.
    manifest = manifest.sort_values(["split", "class_folder_label", "relative_path"])
    manifest.to_csv(args.output_dir / "clean_manifest.csv", index=False)
    group_table.to_csv(args.output_dir / "duplicate_groups.csv", index=False)

    for split_name, filename in (
        ("train", "train.csv"),
        ("validation", "validation.csv"),
        ("test", "test.csv"),
    ):
        manifest[manifest["split"] == split_name].to_csv(split_dir / filename, index=False)

    summary = {
        "dataset_root": str(args.dataset_root),
        "source_images_modified": False,
        "source_images_deleted": 0,
        "source_images_moved": 0,
        "total_readable_images": int(len(manifest)),
        "unique_groups": int(manifest["group_id"].nunique()),
        "split_image_counts": {
            name: int((manifest["split"] == name).sum())
            for name in ("train", "validation", "test")
        },
        "split_group_counts": {
            name: int(group_table[group_table["group_id"].isin(
                manifest.loc[manifest["split"] == name, "group_id"].unique()
            )].shape[0])
            for name in ("train", "validation", "test")
        },
        "class_counts_by_split": {
            split_name: {
                str(label): int(count)
                for label, count in manifest[manifest["split"] == split_name][
                    "class_folder_label"
                ].value_counts().sort_index().items()
            }
            for split_name in ("train", "validation", "test")
        },
        "near_duplicate_review_rows": int(len(reviews)),
        "same_photo_groups_with_multiple_files": int(
            (group_table["group_size"] > 1).sum()
        ),
        "random_seed": args.seed,
        "requested_ratios": {
            "train": args.train_ratio,
            "validation": args.val_ratio,
            "test": args.test_ratio,
        },
        "note": "Splits are group-aware and approximately stratified; counts may differ slightly from exact ratios.",
    }
    with open(args.output_dir / "manifest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nMANIFEST CREATED SUCCESSFULLY")
    print(f"Images: {len(manifest)}")
    print(f"Unique groups: {manifest['group_id'].nunique()}")
    print("\nImages per split:")
    print(manifest["split"].value_counts().to_string())
    print("\nClass distribution by split:")
    print(pd.crosstab(manifest["class_folder_label"], manifest["split"]).to_string())
    print("\nOutput files:")
    print(f"  {args.output_dir / 'clean_manifest.csv'}")
    print(f"  {args.output_dir / 'duplicate_groups.csv'}")
    print(f"  {split_dir / 'train.csv'}")
    print(f"  {split_dir / 'validation.csv'}")
    print(f"  {split_dir / 'test.csv'}")
    print(f"  {args.output_dir / 'manifest_summary.json'}")
    print("\nSource images were not changed.")


if __name__ == "__main__":
    main()
