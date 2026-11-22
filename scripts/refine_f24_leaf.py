import cv2, json, numpy as np

# Load original F24 leaf candidate
with open('artifacts/annotation_staging/phase12_2/candidates_153.json') as f:
    cands = json.load(f)

f24_leaf = next(c for c in cands if c['candidate_id'] == 'CAND_F24_L_1462')
pts = np.array(f24_leaf['polygon_coordinates']).reshape(-1, 2)

# Create mask of current leaf
mask = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)

# Add the apical lobe region that contains 1497 and 1499
# 1497 bbox: [488, 57, 570, 91]
# 1499 bbox: [439, 19, 477, 82]
img = cv2.imread('data/detection_consolidated_100/images/LR00578.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# In the apical patch [0:120, 410:580], get leaf tissue
apical_roi = np.zeros((800, 800), dtype=np.uint8)
apical_roi[0:120, 410:580] = 1

plant_apical = ((hsv[:,:,0] >= 10) & (hsv[:,:,0] <= 90) & (hsv[:,:,1] >= 15)) | \
               (((hsv[:,:,0] < 15) | (hsv[:,:,0] > 165)) & (hsv[:,:,1] >= 20))
plant_apical = cv2.bitwise_and(plant_apical.astype(np.uint8) * 255, apical_roi * 255)

# Combine with existing mask
combined_mask = cv2.bitwise_or(mask, plant_apical)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

cnts, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
largest = max(cnts, key=cv2.contourArea)

peri = cv2.arcLength(largest, True)
for eps in np.linspace(0.002, 0.005, 30):
    approx = cv2.approxPolyDP(largest, eps * peri, True)
    if 25 <= len(approx) <= 45:
        break

new_pts = approx.reshape(-1, 2).astype(np.float32)
new_pts[:, 0] = np.clip(new_pts[:, 0], 0.0, 800.0)
new_pts[:, 1] = np.clip(new_pts[:, 1], 0.0, 800.0)

print(f"Refined F24 leaf: verts={len(new_pts)}, area={cv2.contourArea(new_pts)}")

# Test containment of 1497 and 1499
new_leaf_m = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(new_leaf_m, [np.round(new_pts).astype(np.int32)], 1)

for cid in ['CAND_F24_LES_1497', 'CAND_F24_LES_1499']:
    c = next(x for x in cands if x['candidate_id'] == cid)
    l_pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    l_m = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_m, [np.round(l_pts).astype(np.int32)], 1)
    ov = np.sum((new_leaf_m == 1) & (l_m == 1))
    tot = np.sum(l_m == 1)
    print(f"  {cid} containment in refined leaf: {ov}/{tot} ({ov/tot*100:.1f}%)")
