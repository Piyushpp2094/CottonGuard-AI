"""
scripts/check_mask_b1.py
"""
import json, cv2, numpy as np

with open('artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json') as f:
    master = json.load(f)

lf1 = next(c for c in master if c['candidate_id'] == 'CAND_F21_L_1289')
b1_pts = np.array(lf1['polygon_coordinates']).reshape(-1, 2)
mask_b1 = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask_b1, [np.round(b1_pts).astype(np.int32)], 1)

for rep_id in [1324, 1327, 1329, 1330, 1331, 1334, 1335, 1337]:
    c = next(x for x in master if x.get('replaced_shape_id') == rep_id and x['frame_number'] == 21)
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    l_mask = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_mask, [np.round(pts).astype(np.int32)], 1)
    tot = np.sum(l_mask == 1)
    ov = np.sum((mask_b1 == 1) & (l_mask == 1))
    c_x, c_y = np.mean(pts, axis=0)
    inside_test = cv2.pointPolygonTest(b1_pts.astype(np.float32), (float(c_x), float(c_y)), True)
    print("Rep {}: tot={}, ov={}, pct={:.2f}%, centroid=[{:.1f}, {:.1f}], dist={:+.1f}".format(
        rep_id, tot, ov, ov/tot*100, c_x, c_y, inside_test
    ))
