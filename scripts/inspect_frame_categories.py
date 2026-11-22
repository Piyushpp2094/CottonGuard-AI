"""
scripts/inspect_frame_categories.py
Inspects the exact shape counts per frame in the baseline snapshot and retained shapes.
"""

import json
from collections import Counter
from pathlib import Path

with open("artifacts/backups/cvat_job6_stage0_pre_remediation_snapshot_20261118_205937.json") as f:
    snap = json.load(f)
shapes = snap["shapes"]

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.analyze_retained_baseline import all_del_ids

# Map shapes by frame
shapes_by_frame = Counter(s["frame"] for s in shapes)
retained_by_frame = Counter(s["frame"] for s in shapes if s["id"] not in all_del_ids)
deleted_by_frame = Counter(s["frame"] for s in shapes if s["id"] in all_del_ids)

print(f"{'Frame':<6} {'Total':<8} {'Deleted':<10} {'Retained':<10} {'Ret_Rect':<10} {'Ret_Poly':<10} {'Ret_Leaf':<10} {'Ret_Dis':<10}")
print("-" * 75)

for f in range(50):
    f_shapes = [s for s in shapes if s["frame"] == f]
    f_del = [s for s in f_shapes if s["id"] in all_del_ids]
    f_ret = [s for s in f_shapes if s["id"] not in all_del_ids]
    
    r_rect = sum(1 for s in f_ret if s["type"] == "rectangle")
    r_poly = sum(1 for s in f_ret if s["type"] == "polygon")
    r_leaf = sum(1 for s in f_ret if s["label_id"] == 5)
    r_dis = sum(1 for s in f_ret if s["label_id"] == 6)
    
    del_str = str(len(f_del)) if f_del else "-"
    print(f"{f:<6} {len(f_shapes):<8} {del_str:<10} {len(f_ret):<10} {r_rect:<10} {r_poly:<10} {r_leaf:<10} {r_dis:<10}")
