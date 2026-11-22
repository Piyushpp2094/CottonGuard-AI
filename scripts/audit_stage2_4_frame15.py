"""
scripts/audit_stage2_4_frame15.py
Audits Frame 15 shapes, candidate IDs, replaced shape IDs, and host-leaf containment.
"""

import json
import cv2
import numpy as np
import pandas as pd

# 1. Baseline snapshot shapes for Frame 15
with open("artifacts/backups/cvat_job6_stage0_pre_remediation_snapshot_20261118_205937.json") as f:
    snap = json.load(f)

f15_shapes = [s for s in snap["shapes"] if s["frame"] == 15]
print(f"Total shapes in Frame 15 baseline: {len(f15_shapes)}")

leaf_shapes = [s for s in f15_shapes if s["label_id"] == 5]
rect_shapes = [s for s in f15_shapes if s["type"] == "rectangle"]
poly_shapes = [s for s in f15_shapes if s["type"] == "polygon" and s["label_id"] != 5]

print("Leaf shapes in Frame 15 baseline:", [(s["id"], s["type"], len(s["points"])) for s in leaf_shapes])
print("Rectangle shapes (15 expected):", [s["id"] for s in rect_shapes])
print("Polygon shapes (15 expected):", [s["id"] for s in poly_shapes])

# 2. Check candidates_153.json for Frame 15
with open("artifacts/annotation_staging/phase12_2/candidates_153.json") as f:
    cands153 = json.load(f)

f15_153 = [c for c in cands153 if c["frame_number"] == 15]
print(f"\nCandidates in candidates_153.json for Frame 15: {len(f15_153)}")
for c in f15_153:
    print(f"  CID={c['candidate_id']}, label={c['annotation_label']}, rep_id={c.get('replaced_shape_id')}")

# 3. Check candidates_stage2_2_master.json for Frame 15
with open("artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json") as f:
    master_cands = json.load(f)

f15_master = [c for c in master_cands if c["frame_number"] == 15]
print(f"\nCandidates in candidates_stage2_2_master.json for Frame 15: {len(f15_master)}")
for c in f15_master:
    print(f"  CID={c['candidate_id']}, label={c['annotation_label']}, rep_id={c.get('replaced_shape_id')}, host_leaf={c.get('host_leaf_id')}, cont={c.get('containment_percentage')}")

# 4. Check candidates_adjudication_master.csv for Frame 15
adj_df = pd.read_csv("artifacts/annotation_staging/phase12_2/candidates_adjudication_master.csv")
f15_adj = adj_df[adj_df["frame"] == 15]
print(f"\nAdjudication rows for Frame 15: {len(f15_adj)}")
print(f15_adj[["candidate_id", "original_shape_id", "host_leaf_id", "containment_percentage", "final_decision"]])

# 5. Measure exact pixel intersection of each Frame 15 lesion candidate against:
#    - Primary leaf: CAND_F15_L_1157
#    - Secondary leaf: CAND_F15_L_SECONDARY_01
with open("artifacts/annotation_staging/phase12_2/new_secondary_leaf_candidates.json") as f:
    sec_leaves = json.load(f)

primary_lf = next(c for c in f15_master if c["candidate_id"] == "CAND_F15_L_1157")
sec_lf = sec_leaves["CAND_F15_L_SECONDARY_01"]

p_pts = np.array(primary_lf["polygon_coordinates"]).reshape(-1, 2)
s_pts = np.array(sec_lf["polygon_coordinates"]).reshape(-1, 2)

mask_p = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask_p, [np.round(p_pts).astype(np.int32)], 1)

mask_s = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask_s, [np.round(s_pts).astype(np.int32)], 1)

print("\nExact Pixel Raster Overlap of Frame 15 Lesions with Primary vs Secondary Leaf:")
for c in f15_master:
    if c["annotation_label"] != "Diseased_Region":
        continue
    l_pts = np.array(c["polygon_coordinates"]).reshape(-1, 2)
    l_mask = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_mask, [np.round(l_pts).astype(np.int32)], 1)
    tot = np.sum(l_mask == 1)
    ov_p = np.sum((mask_p == 1) & (l_mask == 1))
    ov_s = np.sum((mask_s == 1) & (l_mask == 1))
    pct_p = (ov_p / tot) * 100
    pct_s = (ov_s / tot) * 100
    print(f"  {c['candidate_id']} (rep {c.get('replaced_shape_id')}): tot={tot}px | Prim: {ov_p} ({pct_p:.1f}%) | Sec: {ov_s} ({pct_s:.1f}%)")
