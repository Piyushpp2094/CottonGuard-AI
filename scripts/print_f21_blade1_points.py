"""
scripts/print_f21_blade1_points.py
"""
import json, numpy as np

with open('artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json') as f:
    master = json.load(f)

lf1 = next(c for c in master if c['candidate_id'] == 'CAND_F21_L_1289')
b1_pts = np.array(lf1['polygon_coordinates']).reshape(-1, 2)
for i, pt in enumerate(b1_pts):
    print(f"{i:2d}: [{pt[0]:.1f}, {pt[1]:.1f}]")
