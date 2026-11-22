"""
scripts/deep_audit_f15_disputed.py
Deep audit of Frame 15 candidates 1176, 1177, 1178, 1179.
"""

import cv2, json, numpy as np
from pathlib import Path

img = cv2.imread("data/detection_consolidated_100/images/LHJ00001.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

with open("artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json") as f:
    master = json.load(f)

# Primary leaf CAND_F15_L_1157
lf1 = next(c for c in master if c["candidate_id"] == "CAND_F15_L_1157")
p_pts = np.array(lf1["polygon_coordinates"]).reshape(-1, 2)
mask_p = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask_p, [np.round(p_pts).astype(np.int32)], 1)

# Secondary leaf CAND_F15_L_SECONDARY_01
with open("artifacts/annotation_staging/phase12_2/new_secondary_leaf_candidates.json") as f:
    sec = json.load(f)
s_pts = np.array(sec["CAND_F15_L_SECONDARY_01"]["polygon_coordinates"]).reshape(-1, 2)
mask_s = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask_s, [np.round(s_pts).astype(np.int32)], 1)

mask_union = cv2.bitwise_or(mask_p, mask_s)

out_dir = Path("artifacts/audit_phase11/f15_inspect")
out_dir.mkdir(parents=True, exist_ok=True)

disputed = [1176, 1177, 1178, 1179]
print("=" * 80)
print("DEEP AUDIT: FRAME 15 DISPUTED CANDIDATES (1176, 1177, 1178, 1179)")
print("=" * 80)

for rep_id in disputed:
    c = next(x for x in master if x.get("replaced_shape_id") == rep_id and x["frame_number"] == 15)
    pts = np.array(c["polygon_coordinates"]).reshape(-1, 2)
    x1, y1 = np.floor(pts.min(axis=0)).astype(int)
    x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
    
    l_mask = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_mask, [np.round(pts).astype(np.int32)], 1)
    tot_px = np.sum(l_mask == 1)
    ov_p = np.sum((mask_p == 1) & (l_mask == 1))
    ov_s = np.sum((mask_s == 1) & (l_mask == 1))
    ov_un = np.sum((mask_union == 1) & (l_mask == 1))
    
    c_x, c_y = np.mean(pts, axis=0)
    dist_p = cv2.pointPolygonTest(p_pts.astype(np.float32), (float(c_x), float(c_y)), True)
    dist_s = cv2.pointPolygonTest(s_pts.astype(np.float32), (float(c_x), float(c_y)), True)
    
    l_hsv = hsv[l_mask == 1]
    l_bgr = img[l_mask == 1]
    mean_bgr = np.mean(l_bgr, axis=0).round(1)
    
    # In LHJ00001 (Leaf Hopper Jassids), symptoms are chlorosis/yellowing, necrosis
    green_cnt = np.sum((l_hsv[:, 0] >= 35) & (l_hsv[:, 0] <= 85) & (l_hsv[:, 1] >= 25) & (l_hsv[:, 2] >= 30))
    yellow_cnt = np.sum((l_hsv[:, 0] >= 15) & (l_hsv[:, 0] < 35) & (l_hsv[:, 1] >= 25) & (l_hsv[:, 2] >= 30))
    dark_cnt = np.sum(l_hsv[:, 2] < 50)
    
    # Context crop
    cx1 = max(0, x1 - 30)
    cy1 = max(0, y1 - 30)
    cx2 = min(800, x2 + 30)
    cy2 = min(800, y2 + 30)
    crop = img[cy1:cy2, cx1:cx2].copy()
    pts_rel = pts - np.array([cx1, cy1])
    cv2.polylines(crop, [np.round(pts_rel).astype(np.int32)], True, (0, 0, 255), 2)
    cv2.imwrite(str(out_dir / f"f15_crop_{rep_id}.png"), crop)
    
    print(f"\nCandidate {c['candidate_id']} (rep {rep_id}):")
    print(f"  BBox: x=[{x1}, {x2}], y=[{y1}, {y2}] | Centroid: [{c_x:.1f}, {c_y:.1f}]")
    print(f"  Area: {c['area']:.1f} px ({tot_px} raster pixels)")
    print(f"  Overlap Primary:   {ov_p}/{tot_px} ({ov_p/tot_px*100:.2f}%), dist={dist_p:+.1f}px")
    print(f"  Overlap Secondary: {ov_s}/{tot_px} ({ov_s/tot_px*100:.2f}%), dist={dist_s:+.1f}px")
    print(f"  Overlap Union:     {ov_un}/{tot_px} ({ov_un/tot_px*100:.2f}%)")
    print(f"  Tissue: Mean BGR={mean_bgr.tolist()}, Green={green_cnt/tot_px*100:.1f}%, Chlorotic/Yellow={yellow_cnt/tot_px*100:.1f}%, Dark/Shadow={dark_cnt/tot_px*100:.1f}%")
