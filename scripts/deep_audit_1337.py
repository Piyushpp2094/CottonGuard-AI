"""
scripts/deep_audit_1337.py
Deep audit of Candidate 1337.
"""

import json, cv2, numpy as np

img = cv2.imread("data/detection_consolidated_100/images/LR00140.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

with open("artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json") as f:
    master = json.load(f)

c1337 = next(c for c in master if c.get("replaced_shape_id") == 1337 and c["frame_number"] == 21)
pts = np.array(c1337["polygon_coordinates"]).reshape(-1, 2)
area = cv2.contourArea(pts.astype(np.float32))

# Load Blade 1
lf1 = next(c for c in master if c["candidate_id"] == "CAND_F21_L_1289")
b1_pts = np.array(lf1["polygon_coordinates"]).reshape(-1, 2)
mask_b1 = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask_b1, [np.round(b1_pts).astype(np.int32)], 1)

# Load Blade 2
with open("artifacts/annotation_staging/phase12_2/new_secondary_leaf_candidates.json") as f:
    sec = json.load(f)
b2_pts = np.array(sec["CAND_F21_L_BLADE2"]["polygon_coordinates"]).reshape(-1, 2)
mask_b2 = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask_b2, [np.round(b2_pts).astype(np.int32)], 1)

mask_union = cv2.bitwise_or(mask_b1, mask_b2)

l_mask = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(l_mask, [np.round(pts).astype(np.int32)], 1)

tot_px = np.sum(l_mask == 1)
ov_b1 = np.sum((mask_b1 == 1) & (l_mask == 1))
ov_b2 = np.sum((mask_b2 == 1) & (l_mask == 1))
ov_un = np.sum((mask_union == 1) & (l_mask == 1))

pct_b1 = ov_b1 / tot_px * 100
pct_b2 = ov_b2 / tot_px * 100
pct_un = ov_un / tot_px * 100

# Inspect non-contained pixels (where l_mask == 1 and mask_union == 0)
non_cont_mask = (l_mask == 1) & (mask_union == 0)
non_cont_px = np.sum(non_cont_mask)
non_cont_hsv = hsv[non_cont_mask]
non_cont_bgr = img[non_cont_mask]

# Tissue stats of non-contained pixels
# Green leaf margin: H in [35, 85], S >= 25
# Anthocyanin: H < 15 or H > 165
# Dark shadow: V < 40
nc_green = np.sum((non_cont_hsv[:,0] >= 35) & (non_cont_hsv[:,0] <= 85) & (non_cont_hsv[:,1] >= 20))
nc_red = np.sum(((non_cont_hsv[:,0] < 15) | (non_cont_hsv[:,0] > 165)) & (non_cont_hsv[:,1] >= 20))
nc_dark = np.sum(non_cont_hsv[:,2] < 40)

print("=" * 80)
print("CANDIDATE 1337 FORENSIC AUDIT")
print("=" * 80)
print(f"Candidate ID: {c1337['candidate_id']}")
print(f"Vertex Count: {len(pts)}")
print(f"Contour Area: {area:.1f} px ({tot_px} raster pixels)")
print(f"Bounding Box: x=[{pts[:,0].min():.1f}, {pts[:,0].max():.1f}], y=[{pts[:,1].min():.1f}, {pts[:,1].max():.1f}]")
print(f"Overlap with Blade 1: {ov_b1} / {tot_px} ({pct_b1:.2f}%)")
print(f"Overlap with Blade 2: {ov_b2} / {tot_px} ({pct_b2:.2f}%)")
print(f"Combined Union Overlap: {ov_un} / {tot_px} ({pct_un:.2f}%)")
print(f"Non-Contained Pixels: {non_cont_px} ({non_cont_px/tot_px*100:.2f}%)")
print(f"  Non-contained tissue: Green Margin={nc_green}/{non_cont_px} ({nc_green/non_cont_px*100:.1f}%), Anthocyanin Red={nc_red}/{non_cont_px} ({nc_red/non_cont_px*100:.1f}%), Shadow={nc_dark}/{non_cont_px} ({nc_dark/non_cont_px*100:.1f}%)")
print(f"  Mean BGR of non-contained pixels: {np.mean(non_cont_bgr, axis=0).round(1).tolist()}")

# Create diagnostic visual crop
x1, y1 = np.floor(pts.min(axis=0)).astype(int) - 20
x2, y2 = np.ceil(pts.max(axis=0)).astype(int) + 20
crop = img[y1:y2, x1:x2].copy()
cv2.polylines(crop, [np.round(pts - [x1, y1]).astype(np.int32)], True, (0, 255, 255), 2)
cv2.polylines(crop, [np.round(b1_pts - [x1, y1]).astype(np.int32)], False, (0, 255, 0), 2)
cv2.polylines(crop, [np.round(b2_pts - [x1, y1]).astype(np.int32)], False, (255, 255, 0), 2)
cv2.imwrite("artifacts/audit_phase11/f21_inspect/crop_1337_diagnostic.png", crop)
print("Saved artifacts/audit_phase11/f21_inspect/crop_1337_diagnostic.png")
