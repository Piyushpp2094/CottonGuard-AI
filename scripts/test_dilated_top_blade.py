import cv2, json, numpy as np

with open('artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json') as f:
    master_cands = json.load(f)

f21_top = [c for c in master_cands if c.get('replaced_shape_id') in [1341, 1342, 1343, 1344] and c['frame_number'] == 21]

mask = np.zeros((800, 800), dtype=np.uint8)
for c in f21_top:
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)

# Dilate by 12 pixels with an elliptical kernel to create natural organic host leaf margin around the lesions
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
dilated = cv2.dilate(mask, kernel, iterations=1)
# Ensure top border touches y=0
dilated[0:2, :] = cv2.dilate(dilated[0:2, :], kernel)

cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
largest = max(cnts, key=cv2.contourArea)
peri = cv2.arcLength(largest, True)
print(f"Dilated lesion mask raw points: {len(largest)}, area: {cv2.contourArea(largest)}")

for eps in np.linspace(0.002, 0.015, 20):
    approx = cv2.approxPolyDP(largest, eps * peri, True)
    print(f"eps={eps:.4f}: verts={len(approx)}")
