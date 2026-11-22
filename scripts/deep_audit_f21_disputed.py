"""
scripts/deep_audit_f21_disputed.py
Deeply audits Frame 21 candidates 1329, 1330, 1331, 1335.
"""

import json
from pathlib import Path
import cv2
import numpy as np

img = cv2.imread("data/detection_consolidated_100/images/LR00140.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

with open("artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json") as f:
    master = json.load(f)

lf1 = next(c for c in master if c["candidate_id"] == "CAND_F21_L_1289")
b1_pts = np.array(lf1["polygon_coordinates"]).reshape(-1, 2)
mask_b1 = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask_b1, [np.round(b1_pts).astype(np.int32)], 1)

with open("artifacts/annotation_staging/phase12_2/new_secondary_leaf_candidates.json") as f:
    sec = json.load(f)
b2_pts = np.array(sec["CAND_F21_L_BLADE2"]["polygon_coordinates"]).reshape(-1, 2)
mask_b2 = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask_b2, [np.round(b2_pts).astype(np.int32)], 1)

mask_union = cv2.bitwise_or(mask_b1, mask_b2)

out_dir = Path("artifacts/audit_phase11/f21_inspect")
out_dir.mkdir(parents=True, exist_ok=True)

disputed_ids = [1329, 1330, 1331, 1335]
print("=" * 80)
print("DEEP AUDIT: FRAME 21 DISPUTED LESIONS (1329, 1330, 1331, 1335)")
print("=" * 80)

for rep_id in disputed_ids:
    c = next(x for x in master if x.get("replaced_shape_id") == rep_id and x["frame_number"] == 21)
    pts = np.array(c["polygon_coordinates"]).reshape(-1, 2)
    x1, y1 = np.floor(pts.min(axis=0)).astype(int)
    x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
    
    l_mask = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_mask, [np.round(pts).astype(np.int32)], 1)
    tot_px = np.sum(l_mask == 1)
    ov_b1 = np.sum((mask_b1 == 1) & (l_mask == 1))
    ov_b2 = np.sum((mask_b2 == 1) & (l_mask == 1))
    ov_un = np.sum((mask_union == 1) & (l_mask == 1))
    
    # Distance to Blade 1 contour
    c_x, c_y = np.mean(pts, axis=0)
    dist_b1 = cv2.pointPolygonTest(b1_pts.astype(np.float32), (float(c_x), float(c_y)), True)
    
    # Tissue properties inside lesion polygon
    l_hsv = hsv[l_mask == 1]
    l_bgr = img[l_mask == 1]
    
    mean_bgr = np.mean(l_bgr, axis=0).round(1)
    red_cnt = np.sum(((l_hsv[:, 0] < 15) | (l_hsv[:, 0] > 165)) & (l_hsv[:, 1] >= 30) & (l_hsv[:, 2] >= 30))
    green_cnt = np.sum((l_hsv[:, 0] >= 35) & (l_hsv[:, 0] <= 85) & (l_hsv[:, 1] >= 30))
    red_pct = red_cnt / len(l_hsv) * 100
    green_pct = green_cnt / len(l_hsv) * 100
    
    # Context crop (expanded by 30px)
    cx1 = max(0, x1 - 30)
    cy1 = max(0, y1 - 30)
    cx2 = min(800, x2 + 30)
    cy2 = min(800, y2 + 30)
    crop = img[cy1:cy2, cx1:cx2].copy()
    
    # Draw contour on crop
    pts_rel = pts - np.array([cx1, cy1])
    cv2.polylines(crop, [np.round(pts_rel).astype(np.int32)], True, (0, 0, 255), 2)
    cv2.imwrite(str(out_dir / f"f21_crop_{rep_id}.png"), crop)
    
    print(f"\nCandidate {c['candidate_id']} (rep {rep_id}):")
    print(f"  BBox: x=[{x1}, {x2}], y=[{y1}, {y2}] | Centroid: [{c_x:.1f}, {c_y:.1f}]")
    print(f"  Area: {c['area']:.1f} px ({tot_px} raster pixels)")
    print(f"  Overlap Blade 1: {ov_b1}/{tot_px} ({ov_b1/tot_px*100:.2f}%), dist={dist_b1:+.1f}px")
    print(f"  Overlap Blade 2: {ov_b2}/{tot_px} ({ov_b2/tot_px*100:.2f}%)")
    print(f"  Overlap Union:   {ov_un}/{tot_px} ({ov_un/tot_px*100:.2f}%)")
    print(f"  Tissue: Mean BGR={mean_bgr.tolist()}, Red={red_pct:.1f}%, Green={green_pct:.1f}%")
