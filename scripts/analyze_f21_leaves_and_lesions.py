"""
scripts/analyze_f21_leaves_and_lesions.py
Forensic analysis of leaves and lesions in Frame 21 (LR00140.jpg).
"""

import cv2, json, numpy as np

img = cv2.imread('data/detection_consolidated_100/images/LR00140.jpg')
with open('artifacts/annotation_staging/phase12_2/candidates_153.json') as f:
    cands = json.load(f)

f21_lesions = [c for c in cands if c['frame_number'] == 21 and c['annotation_label'] == 'Diseased_Region']

for idx, c in enumerate(f21_lesions):
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    bx1, by1 = np.min(pts, axis=0)
    bx2, by2 = np.max(pts, axis=0)
    roi = img[int(by1):int(by2), int(bx1):int(bx2)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green = np.sum((h >= 25) & (h <= 85) & (s >= 30) & (v >= 30))
    # Reddening / anthocyanin: hue < 15 or hue > 165
    red = np.sum(((h < 15) | (h > 165)) & (s >= 30))
    tot = roi.shape[0] * roi.shape[1]
    
    # Check spatial location
    if by2 <= 65:
        blade = "Top Leaf Blade (Blade 2)"
    elif bx2 <= 170:
        blade = "Left Leaf Blade (Blade 3)"
    elif by1 >= 750:
        blade = "Bottom Lobe/Margin (Blade 4)"
    else:
        blade = "Primary Center Blade (Blade 1)"
        
    print(f"[{idx+1:02d}/26] {c['candidate_id']}: bbox=[{bx1:.0f},{by1:.0f},{bx2:.0f},{by2:.0f}], area={c['area']:.1f}, red={red/tot*100:.1f}%, green={green/tot*100:.1f}%, Blade={blade}")
