"""
scripts/verify_stage4_protected_resources.py
Verifies that none of the 303 deletion IDs intersect any protected resource:
- 37 DEC-01 rectangles (Frames 26 & 28)
- 449 holdout shapes (Frames 35-49)
- 106 newly added Stage 3 polygons (IDs 2224-2329)
"""

import json
import sys
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")
sys.path.insert(0, str(WORKSPACE))

from scripts.execute_gate4_stage0_verification import DELETIONS_BY_FRAME

POST_SNAPSHOT_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage3_post_addition_snapshot.json"
MAPPING_PATH = WORKSPACE / "artifacts/annotation_staging/phase12_2/stage3_added_shapes_mapping.json"

with open(POST_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
    post_data = json.load(f)
with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)

post_shapes = {s["id"]: s for s in post_data["shapes"]}

# 303 deletion IDs
del_ids = set()
for f, d in DELETIONS_BY_FRAME.items():
    del_ids.update(d["leaf"] + d["rect"] + d["poly_4pt"])
assert len(del_ids) == 303

# 1. Check 37 DEC-01 rectangles
f26_rect_ids = {s["id"] for s in post_shapes.values() if s.get("frame") == 26 and s.get("type") == "rectangle"}
f28_rect_ids = {s["id"] for s in post_shapes.values() if s.get("frame") == 28 and s.get("type") == "rectangle"}
dec01_ids = f26_rect_ids | f28_rect_ids
print(f"DEC-01 Rectangle IDs count: {len(dec01_ids)} (F26={len(f26_rect_ids)}, F28={len(f28_rect_ids)})")
assert len(f26_rect_ids) == 14
assert len(f28_rect_ids) == 23
assert len(dec01_ids) == 37

dec01_del_overlap = del_ids & dec01_ids
print(f"DEC-01 overlap with deletion IDs: {len(dec01_del_overlap)}")
assert len(dec01_del_overlap) == 0, f"DEC-01 rectangles scheduled for deletion: {dec01_del_overlap}"
print("  -> PASS: Zero DEC-01 rectangles in deletion roster.")

# 2. Check holdout shapes (Frames 35-49)
holdout_ids = {s["id"] for s in post_shapes.values() if 35 <= s.get("frame", -1) <= 49}
print(f"Holdout shape IDs count: {len(holdout_ids)}")
assert len(holdout_ids) == 449

holdout_del_overlap = del_ids & holdout_ids
print(f"Holdout overlap with deletion IDs: {len(holdout_del_overlap)}")
assert len(holdout_del_overlap) == 0, f"Holdout shapes scheduled for deletion: {holdout_del_overlap}"
print("  -> PASS: Zero holdout shapes in deletion roster.")

# 3. Check Stage 3 addition shapes (106 polygons, 2224-2329)
stage3_new_ids = {m["new_cvat_shape_id"] for m in mapping}
print(f"Stage 3 new shape IDs count: {len(stage3_new_ids)}")
assert len(stage3_new_ids) == 106
assert stage3_new_ids == set(range(2224, 2330))

stage3_del_overlap = del_ids & stage3_new_ids
print(f"Stage 3 new shape overlap with deletion IDs: {len(stage3_del_overlap)}")
assert len(stage3_del_overlap) == 0, f"Stage 3 additions scheduled for deletion: {stage3_del_overlap}"
print("  -> PASS: Zero Stage 3 addition shapes in deletion roster.")

# 4. Check frame boundaries of deletions
del_frames = {post_shapes[sid]["frame"] for sid in del_ids}
print(f"Deletion frames: {sorted(list(del_frames))}")
assert all(0 <= f <= 34 for f in del_frames), "Deletions exist outside development set!"
assert del_frames == {0, 9, 15, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28, 31}
print("  -> PASS: All deletion shapes belong strictly to the 14 approved remediation frames.")
