"""
scripts/inspect_f15_lesions.py
Inspects the coordinates and bounding boxes of Frame 15 lesions 1173-1179.
"""

import json
import numpy as np

with open("artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json") as f:
    master_cands = json.load(f)

for c in master_cands:
    if c["frame_number"] == 15 and c["annotation_label"] == "Diseased_Region":
        pts = np.array(c["polygon_coordinates"]).reshape(-1, 2)
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        print(f"{c['candidate_id']} (rep {c['replaced_shape_id']}): bbox x=[{min_xy[0]:.1f}, {max_xy[0]:.1f}], y=[{min_xy[1]:.1f}, {max_xy[1]:.1f}], area={c['area']}")

with open("artifacts/annotation_staging/phase12_2/new_secondary_leaf_candidates.json") as f:
    sec_leaves = json.load(f)

sec_pts = np.array(sec_leaves["CAND_F15_L_SECONDARY_01"]["polygon_coordinates"]).reshape(-1, 2)
print("\nSecondary leaf bbox: x=[{:.1f}, {:.1f}], y=[{:.1f}, {:.1f}]".format(
    sec_pts[:, 0].min(), sec_pts[:, 0].max(), sec_pts[:, 1].min(), sec_pts[:, 1].max()
))
