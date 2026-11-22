"""
scripts/export_frozen_deletion_registry.py
Exports artifacts/audit_phase11/stage4_frozen_deletion_registry.json from DELETIONS_BY_FRAME and post-Stage-3 snapshot.
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")
sys.path.insert(0, str(WORKSPACE))

from scripts.execute_gate4_stage0_verification import DELETIONS_BY_FRAME

SNAPSHOT_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage3_post_addition_snapshot.json"
OUTPUT_PATH = WORKSPACE / "artifacts/audit_phase11/stage4_frozen_deletion_registry.json"

with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
    snap = json.load(f)
shapes_by_id = {s["id"]: s for s in snap["shapes"]}

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

records = []
for f_idx, d_dict in sorted(DELETIONS_BY_FRAME.items()):
    f_meta = FRAME_INFO[f_idx]
    
    # 1. Leaves
    for sid in d_dict["leaf"]:
        s = shapes_by_id[sid]
        records.append({
            "cvat_shape_id": sid,
            "frame": f_idx,
            "global_id": f_meta["global_id"],
            "image_filename": f_meta["filename"],
            "disease_class": f_meta["disease"],
            "label": "Leaf",
            "label_id": 5,
            "geometry_type": "polygon",
            "vertex_count": len(s["points"]) // 2,
            "points": s["points"],
            "removal_reason": "Legacy coarse/full-frame pseudo-leaf polygon replaced by organic contour",
            "category": "Leaf Pseudo-Mask"
        })

    # 2. Rectangles
    for sid in d_dict["rect"]:
        s = shapes_by_id[sid]
        records.append({
            "cvat_shape_id": sid,
            "frame": f_idx,
            "global_id": f_meta["global_id"],
            "image_filename": f_meta["filename"],
            "disease_class": f_meta["disease"],
            "label": "Diseased_Region",
            "label_id": 6,
            "geometry_type": "rectangle",
            "vertex_count": 2,
            "points": s["points"],
            "removal_reason": "Obsolete coarse bounding rectangle replaced by organic lesion contour",
            "category": "Lesion Detection Rectangle"
        })

    # 3. Poly 4pt
    for sid in d_dict["poly_4pt"]:
        s = shapes_by_id[sid]
        records.append({
            "cvat_shape_id": sid,
            "frame": f_idx,
            "global_id": f_meta["global_id"],
            "image_filename": f_meta["filename"],
            "disease_class": f_meta["disease"],
            "label": "Diseased_Region",
            "label_id": 6,
            "geometry_type": "polygon",
            "vertex_count": 4,
            "points": s["points"],
            "removal_reason": "Synthetic diamond polygon replica replaced by organic lesion contour",
            "category": "Synthetic 4-Point Diamond Replica"
        })

# Sort records strictly numerically by cvat_shape_id
records.sort(key=lambda x: x["cvat_shape_id"])

print(f"Total frozen deletion records: {len(records)}")
assert len(records) == 303
unique_ids = {r["cvat_shape_id"] for r in records}
assert len(unique_ids) == 303

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2)

h = hashlib.sha256()
with open(OUTPUT_PATH, "rb") as f:
    while chunk := f.read(65536):
        h.update(chunk)
digest = h.hexdigest().lower()
print(f"Exported frozen deletion registry to: {OUTPUT_PATH}")
print(f"SHA-256 Digest: {digest}")
