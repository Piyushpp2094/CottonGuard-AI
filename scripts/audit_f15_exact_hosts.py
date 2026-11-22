"""
scripts/audit_f15_exact_hosts.py
Audits Frame 15 lesion positions relative to Primary Leaf CAND_F15_L_1157 and Secondary Leaf CAND_F15_L_SECONDARY_01.
"""

import json
import cv2
import numpy as np

with open("artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json") as f:
    master = json.load(f)

f15_cands = [c for c in master if c["frame_number"] == 15 and c["annotation_label"] == "Diseased_Region"]
prim_lf = next(c for c in master if c["candidate_id"] == "CAND_F15_L_1157")
sec_lf = next(c for c in master if c["candidate_id"] == "CAND_F15_L_SECONDARY_01")

p_pts = np.array(prim_lf["polygon_coordinates"]).reshape(-1, 2)
s_pts = np.array(sec_lf["polygon_coordinates"]).reshape(-1, 2)

print("Primary leaf bbox: x=[{:.1f}, {:.1f}], y=[{:.1f}, {:.1f}]".format(
    p_pts[:, 0].min(), p_pts[:, 0].max(), p_pts[:, 1].min(), p_pts[:, 1].max()
))
print("Secondary leaf bbox: x=[{:.1f}, {:.1f}], y=[{:.1f}, {:.1f}]".format(
    s_pts[:, 0].min(), s_pts[:, 0].max(), s_pts[:, 1].min(), s_pts[:, 1].max()
))

mask_p = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask_p, [np.round(p_pts).astype(np.int32)], 1)
mask_s = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask_s, [np.round(s_pts).astype(np.int32)], 1)

for c in f15_cands:
    cid = c["candidate_id"]
    rep_id = c["replaced_shape_id"]
    l_pts = np.array(c["polygon_coordinates"]).reshape(-1, 2)
    l_mask = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_mask, [np.round(l_pts).astype(np.int32)], 1)
    tot = np.sum(l_mask == 1)
    ov_p = np.sum((mask_p == 1) & (l_mask == 1))
    ov_s = np.sum((mask_s == 1) & (l_mask == 1))
    
    # Distance to primary contour and secondary contour
    c_x, c_y = np.mean(l_pts, axis=0)
    dist_p = cv2.pointPolygonTest(p_pts.astype(np.float32), (float(c_x), float(c_y)), True)
    dist_s = cv2.pointPolygonTest(s_pts.astype(np.float32), (float(c_x), float(c_y)), True)
    
    print(f"CID {cid} (rep {rep_id}): centroid=[{c_x:.1f}, {c_y:.1f}] | OvPrim={ov_p}/{tot} ({ov_p/tot*100:.1f}%, dist={dist_p:+.1f}) | OvSec={ov_s}/{tot} ({ov_s/tot*100:.1f}%, dist={dist_s:+.1f})")
