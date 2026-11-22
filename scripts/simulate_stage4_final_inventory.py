"""
scripts/simulate_stage4_final_inventory.py
Dry-run simulation of Stage 4 deletion of 303 shapes on a copy of post-Stage-3 snapshot.
"""

import json
import sys
from collections import Counter, defaultdict
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

post_shapes = post_data["shapes"]
assert len(post_shapes) == 1276

# 303 deletion IDs
del_ids = set()
for f, d in DELETIONS_BY_FRAME.items():
    del_ids.update(d["leaf"] + d["rect"] + d["poly_4pt"])
assert len(del_ids) == 303

# Simulate removal
simulated_shapes = [s for s in post_shapes if s["id"] not in del_ids]

print("=== DRY-RUN FINAL INVENTORY SIMULATION RESULTS ===")
print(f"Post-Stage-3 Input Shapes:    {len(post_shapes)}")
print(f"Simulated Deleted Shapes:   - {len(del_ids)}")
print(f"Simulated Final Shapes:     = {len(simulated_shapes)} (Target: 973)")
assert len(simulated_shapes) == 973

sim_types = Counter(s["type"] for s in simulated_shapes)
sim_labels = Counter(s["label_id"] for s in simulated_shapes)

print(f"\nSimulated Geometry Breakdown:")
print(f"  Rectangles: {sim_types['rectangle']} (Target: 345)")
print(f"  Polygons:   {sim_types['polygon']} (Target: 628)")
assert sim_types["rectangle"] == 345
assert sim_types["polygon"] == 628

print(f"\nSimulated Label Breakdown:")
print(f"  Leaf (label_id 5):            {sim_labels[5]} (Target: 52)")
print(f"  Diseased_Region (label_id 6): {sim_labels[6]} (Target: 921)")
assert sim_labels[5] == 52
assert sim_labels[6] == 921

# Verification of all 106 Stage 3 additions
stage3_ids = {m["new_cvat_shape_id"] for m in mapping}
sim_shape_ids = {s["id"] for s in simulated_shapes}
assert stage3_ids.issubset(sim_shape_ids), "Some Stage 3 additions were lost in simulation!"
print(f"\nStage 3 Additions Check: All {len(stage3_ids)} additions (2224-2329) fully preserved in simulated final.")

# Verification of 37 DEC-01 rectangles
f26_rects = [s for s in simulated_shapes if s["frame"] == 26 and s["type"] == "rectangle"]
f28_rects = [s for s in simulated_shapes if s["frame"] == 28 and s["type"] == "rectangle"]
print(f"DEC-01 Rectangles Check: Frame 26={len(f26_rects)}, Frame 28={len(f28_rects)}, Total={len(f26_rects) + len(f28_rects)} (Target: 37)")
assert len(f26_rects) == 14
assert len(f28_rects) == 23

# Verification of holdouts
holdout_shapes = [s for s in simulated_shapes if 35 <= s.get("frame", -1) <= 49]
print(f"Holdout Frames 35-49 Check: {len(holdout_shapes)} shapes (Target: 449)")
assert len(holdout_shapes) == 449

# Verification of unaltered non-deletion shapes
orig_post_by_id = {s["id"]: s for s in post_shapes}
unaltered_count = 0
for s in simulated_shapes:
    orig = orig_post_by_id[s["id"]]
    assert s == orig, f"Shape {s['id']} was modified in simulation!"
    unaltered_count += 1
print(f"Non-deletion shapes unaltered: {unaltered_count}/973")
assert unaltered_count == 973

print("\nALL SIMULATION CHECKS PASSED 100%!")
