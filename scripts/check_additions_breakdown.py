"""
scripts/check_additions_breakdown.py
Checks per-frame distribution of the 106 additions.
"""

import json
from collections import Counter
from pathlib import Path

MANIFEST_PATH = Path("artifacts/annotation_staging/phase12_2/precision_additions_manifest.json")
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    items = json.load(f)

print(f"Total manifest entries: {len(items)}")

frames = sorted(set(x["frame_number"] for x in items))
print(f"Frames affected: {len(frames)} frames -> {frames}")

for f in frames:
    f_items = [x for x in items if x["frame_number"] == f]
    leaves = [x for x in f_items if x["label_id"] == 5]
    lesions = [x for x in f_items if x["label_id"] == 6]
    img = f_items[0]["image_filename"]
    gid = f_items[0]["global_id"]
    print(f"Frame {f:02d} (Global ID {gid}, {img}): Total={len(f_items)} (Leaves={len(leaves)}, Lesions={len(lesions)})")

total_leaves = sum(1 for x in items if x["label_id"] == 5)
total_lesions = sum(1 for x in items if x["label_id"] == 6)
print(f"Grand Total: {len(items)} (Leaves={total_leaves}, Lesions={total_lesions})")
