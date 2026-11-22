import cv2, json, numpy as np
from pathlib import Path

img = cv2.imread('data/detection_consolidated_100/images/LHJ00001.jpg')
h, w = img.shape[:2]

# Segment the secondary lower-left leaf blade
# Lesions 1173-1179 lie in x: [0, 350], y: [280, 800]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Threshold for green and yellow/brown leaf tissue
lower_tissue1 = np.array([10, 20, 25])
upper_tissue1 = np.array([85, 255, 255])
mask1 = cv2.inRange(hsv, lower_tissue1, upper_tissue1)

lower_tissue2 = np.array([0, 30, 25])
upper_tissue2 = np.array([10, 255, 255])
mask2 = cv2.inRange(hsv, lower_tissue2, upper_tissue2)

tissue_mask = cv2.bitwise_or(mask1, mask2)

# Restrict to the lower-left secondary leaf region
roi_mask = np.zeros((h, w), dtype=np.uint8)
# The boundary of the lower left blade
roi_mask[260:800, 0:380] = 1

blade2_mask = cv2.bitwise_and(tissue_mask, tissue_mask, mask=roi_mask)

# Morphological clean up
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
blade2_mask = cv2.morphologyEx(blade2_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
blade2_mask = cv2.morphologyEx(blade2_mask, cv2.MORPH_OPEN, kernel, iterations=1)

cnts, _ = cv2.findContours(blade2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
best_cnt = max(cnts, key=cv2.contourArea)

# Simplify contour with approxPolyDP while keeping smooth organic boundary and >= 15 vertices
peri = cv2.arcLength(best_cnt, True)
for eps_factor in [0.003, 0.004, 0.005, 0.006]:
    approx = cv2.approxPolyDP(best_cnt, eps_factor * peri, True)
    if len(approx) >= 15:
        break

print(f"Generated secondary leaf contour with {len(approx)} vertices, area: {cv2.contourArea(approx)}")

# Convert to flat float list
pts = approx.reshape(-1, 2).astype(np.float32)
# Ensure coordinates are within [0.0, 800.0]
pts[:, 0] = np.clip(pts[:, 0], 0.0, 800.0)
pts[:, 1] = np.clip(pts[:, 1], 0.0, 800.0)

# Check containment of lesions 1173-1179
sec_leaf_mask = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(sec_leaf_mask, [np.round(pts).astype(np.int32)], 1)

with open('artifacts/annotation_staging/phase12_2/candidates_153.json') as f:
    cands = json.load(f)

f15_7 = [c for c in cands if c['frame_number'] == 15 and c['replaced_shape_id'] in range(1173, 1180)]
print("\nContainment of lesions inside new secondary leaf:")
for c in f15_7:
    l_pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    l_mask = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_mask, [np.round(l_pts).astype(np.int32)], 1)
    overlap = np.sum((sec_leaf_mask == 1) & (l_mask == 1))
    total = np.sum(l_mask == 1)
    ratio = overlap / total if total > 0 else 0
    print(f"  {c['candidate_id']}: {overlap}/{total} ({ratio*100:.2f}%)")
