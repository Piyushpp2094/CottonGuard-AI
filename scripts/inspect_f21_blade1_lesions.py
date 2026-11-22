"""
scripts/inspect_f21_blade1_lesions.py
"""
import json
import numpy as np

with open('artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json') as f:
    master = json.load(f)

lf1 = next(c for c in master if c['candidate_id'] == 'CAND_F21_L_1289')
b1_pts = np.array(lf1['polygon_coordinates']).reshape(-1, 2)
print("Blade 1 bbox: x=[{:.1f}, {:.1f}], y=[{:.1f}, {:.1f}]".format(
    b1_pts[:, 0].min(), b1_pts[:, 0].max(), b1_pts[:, 1].min(), b1_pts[:, 1].max()
))

for rep_id in [1324, 1327, 1329, 1330, 1331, 1334, 1335, 1337]:
    c = next(x for x in master if x.get('replaced_shape_id') == rep_id and x['frame_number'] == 21)
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    print("Lesion {}: bbox x=[{:.1f}, {:.1f}], y=[{:.1f}, {:.1f}], area={:.1f}".format(
        rep_id, pts[:,0].min(), pts[:,0].max(), pts[:,1].min(), pts[:,1].max(), c['area']
    ))
