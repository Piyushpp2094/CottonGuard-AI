import cv2, json, numpy as np

img = cv2.imread('data/detection_consolidated_100/images/LR00578.jpg')
with open('artifacts/annotation_staging/phase12_2/candidates_153.json') as f:
    cands = json.load(f)

f24_cands = [c for c in cands if c['frame_number'] == 24]
print(f"Total Frame 24 candidates: {len(f24_cands)}")
f24_leaf = next(c for c in f24_cands if c['annotation_label'] == 'Leaf')
leaf_pts = np.array(f24_leaf['polygon_coordinates']).reshape(-1, 2)
leaf_mask = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(leaf_mask, [np.round(leaf_pts).astype(np.int32)], 1)

f24_lesions = [c for c in f24_cands if c['annotation_label'] == 'Diseased_Region']
for c in f24_lesions:
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    bx1, by1 = pts.min(axis=0)
    bx2, by2 = pts.max(axis=0)
    roi = img[int(by1):int(by2), int(bx1):int(bx2)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    red = np.sum(((h < 15) | (h > 165)) & (s >= 25))
    green = np.sum((h >= 25) & (h <= 85) & (s >= 30) & (v >= 30))
    tot = roi.shape[0] * roi.shape[1] if roi.size > 0 else 1
    
    l_mask = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_mask, [np.round(pts).astype(np.int32)], 1)
    overlap = np.sum((leaf_mask == 1) & (l_mask == 1))
    l_tot = np.sum(l_mask == 1)
    containment = overlap / l_tot if l_tot > 0 else 0
    
    print(f"{c['candidate_id']}: bbox=[{bx1:.0f}, {by1:.0f}, {bx2:.0f}, {by2:.0f}], containment={containment*100:.1f}%, red={red/tot*100:.1f}%, green={green/tot*100:.1f}%")
