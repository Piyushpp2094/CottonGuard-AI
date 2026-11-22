"""
scripts/inspect_f21_leaf_structure.py
Inspects the full geometry of Frame 21 (LR00140.jpg).
"""

import json, cv2, numpy as np

with open('artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json') as f:
    master = json.load(f)

lf1 = next(c for c in master if c['candidate_id'] == 'CAND_F21_L_1289')
b1_pts = np.array(lf1['polygon_coordinates']).reshape(-1, 2)

img = cv2.imread('data/detection_consolidated_100/images/LR00140.jpg')

# Print vertices of Blade 1
print("Blade 1 vertex count:", len(b1_pts))
print("Blade 1 points sample:\n", b1_pts[:10])

# Check what the original CVAT Job 6 had for Frame 21 leaf!
with open("artifacts/backups/cvat_job6_stage0_pre_remediation_snapshot_20261118_205937.json") as f:
    snap = json.load(f)

f21_snap_shapes = [s for s in snap["shapes"] if s["frame"] == 21]
f21_snap_leaves = [s for s in f21_snap_shapes if s["label_id"] == 5]
print("\nCVAT snapshot leaves for Frame 21:")
for s in f21_snap_leaves:
    print(f"  Shape {s['id']}: type={s['type']}, points={s['points']}")
