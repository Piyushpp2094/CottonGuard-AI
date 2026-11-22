"""
scripts/summarize_stage3_mapping.py
Summarizes the Stage 3 added shapes mapping.
"""

import json
from collections import Counter
from pathlib import Path

mapping_path = Path("artifacts/annotation_staging/phase12_2/stage3_added_shapes_mapping.json")
with open(mapping_path, "r", encoding="utf-8") as f:
    mapping = json.load(f)

print(f"Total mappings: {len(mapping)}")
new_ids = [x["new_cvat_shape_id"] for x in mapping]
print(f"New CVAT Shape IDs range: {min(new_ids)} - {max(new_ids)}")

frames = sorted(set(x["frame"] for x in mapping))
print(f"\nPer-Frame Summary ({len(frames)} frames):")
print(f"{'Frame':<6} | {'Image Filename':<15} | {'Global ID':<10} | {'Leaves':<8} | {'Lesions':<8} | {'Total':<6} | {'CVAT ID Range':<15}")
print("-" * 75)
for f in frames:
    f_items = [x for x in mapping if x["frame"] == f]
    leaves = [x for x in f_items if x["label_id"] == 5]
    lesions = [x for x in f_items if x["label_id"] == 6]
    f_ids = [x["new_cvat_shape_id"] for x in f_items]
    id_range = f"{min(f_ids)} - {max(f_ids)}" if len(f_ids) > 1 else f"{min(f_ids)}"
    img = f_items[0]["image_filename"]
    gid = f_items[0]["global_id"]
    print(f"{f:<6} | {img:<15} | {gid:<10} | {len(leaves):<8} | {len(lesions):<8} | {len(f_items):<6} | {id_range:<15}")

leaf_total = sum(1 for x in mapping if x["label_id"] == 5)
lesion_total = sum(1 for x in mapping if x["label_id"] == 6)
print("-" * 75)
print(f"{'TOTAL':<6} | {'14 frames':<15} | {'':<10} | {leaf_total:<8} | {lesion_total:<8} | {len(mapping):<6} | {min(new_ids)} - {max(new_ids)}")
