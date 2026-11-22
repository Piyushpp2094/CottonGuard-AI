"""
scripts/audit_303_deletions_full.py
Extracts full metadata for each of the 303 deletion IDs and validates all requirements.
"""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")
sys.path.insert(0, str(WORKSPACE))

from scripts.execute_gate4_stage0_verification import DELETIONS_BY_FRAME

POST_SNAPSHOT_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage3_post_addition_snapshot.json"
with open(POST_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
    snap = json.load(f)

post_shapes = {s["id"]: s for s in snap["shapes"]}

FRAME_INFO = {
    0:  {"filename": "BBC00001.jpg", "global_id": 50, "disease": "Bacterial Blight Cotton"},
    9:  {"filename": "CV00431.jpg",  "global_id": 59, "disease": "Cotton Vein Enation Virus"},
    15: {"filename": "LHJ00001.jpg", "global_id": 65, "disease": "Leaf Hopper and Jassids"},
    17: {"filename": "LHJ00116.jpg", "global_id": 67, "disease": "Leaf Hopper and Jassids"},
    18: {"filename": "LHJ00172.jpg", "global_id": 68, "disease": "Leaf Hopper and Jassids"},
    19: {"filename": "LHJ00225.jpg", "global_id": 69, "disease": "Leaf Hopper and Jassids"},
    20: {"filename": "LR00001.jpg",  "global_id": 70, "disease": "Leaf Reddening"},
    21: {"filename": "LR00140.jpg",  "global_id": 71, "disease": "Leaf Reddening"},
    22: {"filename": "LR00286.jpg",  "global_id": 72, "disease": "Leaf Reddening"},
    23: {"filename": "LR00440.jpg",  "global_id": 73, "disease": "Leaf Reddening"},
    24: {"filename": "LR00578.jpg",  "global_id": 74, "disease": "Leaf Reddening"},
    26: {"filename": "LV00028.jpg",  "global_id": 76, "disease": "Leaf Variegation"},
    28: {"filename": "LV00087.jpg",  "global_id": 78, "disease": "Leaf Variegation"},
    31: {"filename": "HL00070.jpg",  "global_id": 81, "disease": "Healthy Leaf"},
}

deletion_records = []
category_counts = Counter()
frame_counts = Counter()

for f_idx, d_dict in sorted(DELETIONS_BY_FRAME.items()):
    f_meta = FRAME_INFO[f_idx]
    
    # 1. Leaves
    for sid in d_dict["leaf"]:
        s = post_shapes.get(sid)
        assert s is not None, f"Missing Leaf deletion ID {sid}"
        assert s["frame"] == f_idx
        assert s["label_id"] == 5
        assert s["type"] == "polygon"
        rec = {
            "id": sid,
            "frame": f_idx,
            "filename": f_meta["filename"],
            "global_id": f_meta["global_id"],
            "disease": f_meta["disease"],
            "label_id": 5,
            "label_name": "Leaf",
            "type": s["type"],
            "vertex_count": len(s["points"]) // 2,
            "points": s["points"],
            "category": "Leaf Pseudo-Mask",
            "reason": "Legacy coarse/full-frame pseudo-leaf polygon replaced by organic contour"
        }
        deletion_records.append(rec)
        category_counts["Leaf Pseudo-Mask (polygon)"] += 1
        frame_counts[f_idx] += 1

    # 2. Rectangles
    for sid in d_dict["rect"]:
        s = post_shapes.get(sid)
        assert s is not None, f"Missing Rect deletion ID {sid}"
        assert s["frame"] == f_idx
        assert s["label_id"] == 6
        assert s["type"] == "rectangle"
        rec = {
            "id": sid,
            "frame": f_idx,
            "filename": f_meta["filename"],
            "global_id": f_meta["global_id"],
            "disease": f_meta["disease"],
            "label_id": 6,
            "label_name": "Diseased_Region",
            "type": s["type"],
            "vertex_count": 2, # CVAT rectangle has 4 points: [xtl, ytl, xbr, ybr]
            "points": s["points"],
            "category": "Lesion Detection Rectangle",
            "reason": "Obsolete coarse bounding rectangle replaced by organic lesion contour"
        }
        deletion_records.append(rec)
        category_counts["Lesion Detection Rectangle (rectangle)"] += 1
        frame_counts[f_idx] += 1

    # 3. Poly 4pt
    for sid in d_dict["poly_4pt"]:
        s = post_shapes.get(sid)
        assert s is not None, f"Missing Poly_4pt deletion ID {sid}"
        assert s["frame"] == f_idx
        assert s["label_id"] == 6
        assert s["type"] == "polygon"
        assert len(s["points"]) // 2 == 4
        rec = {
            "id": sid,
            "frame": f_idx,
            "filename": f_meta["filename"],
            "global_id": f_meta["global_id"],
            "disease": f_meta["disease"],
            "label_id": 6,
            "label_name": "Diseased_Region",
            "type": s["type"],
            "vertex_count": 4,
            "points": s["points"],
            "category": "Synthetic 4-Point Diamond Replica",
            "reason": "Synthetic diamond polygon replica replaced by organic lesion contour"
        }
        deletion_records.append(rec)
        category_counts["Synthetic 4-Point Polygon (polygon)"] += 1
        frame_counts[f_idx] += 1

print(f"Total deletion records: {len(deletion_records)}")
assert len(deletion_records) == 303

print("\n--- CATEGORY BREAKDOWN ---")
for cat, cnt in category_counts.items():
    print(f"  {cat:<40}: {cnt}")

print("\n--- GEOMETRY BREAKDOWN ---")
rect_cnt = sum(1 for r in deletion_records if r["type"] == "rectangle")
poly_cnt = sum(1 for r in deletion_records if r["type"] == "polygon")
print(f"  Rectangles: {rect_cnt} (Expected: 150)")
print(f"  Polygons:   {poly_cnt} (Expected: 153)")
assert rect_cnt == 150
assert poly_cnt == 153

print("\n--- LABEL BREAKDOWN ---")
leaf_cnt = sum(1 for r in deletion_records if r["label_id"] == 5)
lesion_cnt = sum(1 for r in deletion_records if r["label_id"] == 6)
print(f"  Leaf:            {leaf_cnt} (Expected: 10)")
print(f"  Diseased_Region: {lesion_cnt} (Expected: 293)")
assert leaf_cnt == 10
assert lesion_cnt == 293

print("\n--- PER-FRAME DELETION SUMMARY ---")
print(f"{'Frame':<6} | {'Image Filename':<15} | {'Global ID':<10} | {'Leaves':<8} | {'Rectangles':<12} | {'4pt Polygons':<14} | {'Total':<6}")
print("-" * 85)
for f_idx in sorted(FRAME_INFO.keys()):
    f_recs = [r for r in deletion_records if r["frame"] == f_idx]
    l = sum(1 for r in f_recs if r["category"] == "Leaf Pseudo-Mask")
    rc = sum(1 for r in f_recs if r["category"] == "Lesion Detection Rectangle")
    p4 = sum(1 for r in f_recs if r["category"] == "Synthetic 4-Point Diamond Replica")
    tot = len(f_recs)
    f_meta = FRAME_INFO[f_idx]
    print(f"{f_idx:<6} | {f_meta['filename']:<15} | {f_meta['global_id']:<10} | {l:<8} | {rc:<12} | {p4:<14} | {tot:<6}")
print("-" * 85)
print(f"{'TOTAL':<6} | {'14 frames':<15} | {'':<10} | {leaf_cnt:<8} | {rect_cnt:<12} | {sum(1 for r in deletion_records if r['category'] == 'Synthetic 4-Point Diamond Replica'):<14} | {len(deletion_records):<6}")

# Save full deletion records to scratch/deletion_records_303.json
out_path = Path("scratch/deletion_records_303.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(deletion_records, f, indent=2)
print(f"\nSaved full deletion records to {out_path}")
