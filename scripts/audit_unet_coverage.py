from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
manifest_path = PROJECT_ROOT / "artifacts" / "manifests" / "consolidated_100_annotation_manifest.csv"
manifest = pd.read_csv(manifest_path)
masks_dir = PROJECT_ROOT / "data" / "annotations" / "masks"
mask_files = {p.stem: p for p in masks_dir.glob("*.png")}

print(f"Total masks in data/annotations/masks/: {len(mask_files)}")

manifest["has_mask"] = manifest["filename"].apply(lambda fn: Path(fn).stem in mask_files)
print("\nMask presence by CVAT Task ID:")
print(manifest.groupby("cvat_task_id")["has_mask"].value_counts())

print("\nMask presence by Batch:")
print(manifest.groupby("batch")["has_mask"].value_counts())

print("\nMask presence by Split:")
print(manifest.groupby("split")["has_mask"].value_counts())

# Check pixel values across all 50 masks
mask_stats = []
for stem, p in mask_files.items():
    arr = np.array(Image.open(p))
    u = np.unique(arr)
    row = manifest[manifest["filename"].str.startswith(stem)]
    cl = row.iloc[0]["class_label"] if len(row) > 0 else "Unknown"
    is_healthy = (cl == "Healthy Leaf")
    disease_px = int(np.sum(arr == 2))
    leaf_px = int(np.sum(arr == 1))
    bg_px = int(np.sum(arr == 0))
    mask_stats.append({
        "stem": stem,
        "class": cl,
        "unique_vals": list(u),
        "disease_px": disease_px,
        "leaf_px": leaf_px,
        "bg_px": bg_px,
        "healthy_control_violation": is_healthy and (disease_px > 0),
        "invalid_vals": not set(u).issubset({0, 1, 2}),
    })

df_m = pd.DataFrame(mask_stats)
print(f"\nAny healthy control violations? {df_m['healthy_control_violation'].any()}")
print(f"Any invalid pixel values? {df_m['invalid_vals'].any()}")
hl_masks = df_m[df_m["class"] == "Healthy Leaf"]
print(f"Healthy control masks ({len(hl_masks)} total):")
for _, r in hl_masks.iterrows():
    print(f"  {r['stem']}: unique={r['unique_vals']}, disease_px={r['disease_px']}")
