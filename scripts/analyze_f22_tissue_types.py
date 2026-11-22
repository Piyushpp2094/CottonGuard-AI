"""
scripts/analyze_f22_tissue_types.py
Forensic analysis of tissue types for all 35 candidates in Frame 22 (LR00286.jpg).
"""

import cv2, json, numpy as np

img = cv2.imread('data/detection_consolidated_100/images/LR00286.jpg')
with open('artifacts/annotation_staging/phase12_2/candidates_153.json') as f:
    cands = json.load(f)

f22_lesions = [c for c in cands if c['frame_number'] == 22 and c['annotation_label'] == 'Diseased_Region']

for idx, c in enumerate(f22_lesions):
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    bx1, by1 = np.min(pts, axis=0)
    bx2, by2 = np.max(pts, axis=0)
    roi = img[int(by1):int(by2), int(bx1):int(bx2)]
    if roi.size == 0:
        continue
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green = np.sum((h >= 25) & (h <= 85) & (s >= 30) & (v >= 30))
    # Red / purple / anthocyanin
    red = np.sum(((h < 15) | (h > 165)) & (s >= 25))
    tot = roi.shape[0] * roi.shape[1]
    
    # Check if inside central primary green leaf [81, 109, 621, 671]
    in_primary = (bx1 >= 70) and (bx2 <= 630) and (by1 >= 100) and (by2 <= 680)
    
    print(f"[{idx+1:02d}/35] {c['candidate_id']}: bbox=[{bx1:.0f},{by1:.0f},{bx2:.0f},{by2:.0f}], in_primary={in_primary}, red={red/tot*100:.1f}%, green={green/tot*100:.1f}%")
