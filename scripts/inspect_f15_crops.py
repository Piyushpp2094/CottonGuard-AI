"""
scripts/inspect_f15_crops.py
Inspects the crops of 1173-1179 in LHJ00001.jpg.
"""

import cv2
import json
import numpy as np
from pathlib import Path

img = cv2.imread("data/detection_consolidated_100/images/LHJ00001.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

with open("artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json") as f:
    master = json.load(f)

f15_cands = [c for c in master if c["frame_number"] == 15 and c["annotation_label"] == "Diseased_Region"]

out_dir = Path("artifacts/audit_phase11/f15_inspect")
out_dir.mkdir(parents=True, exist_ok=True)

print("Frame 15 Lesion Crops Analysis:")
for c in f15_cands:
    rep_id = c["replaced_shape_id"]
    pts = np.array(c["polygon_coordinates"]).reshape(-1, 2)
    x1, y1 = np.floor(pts.min(axis=0)).astype(int)
    x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(800, x2), min(800, y2)
    crop = img[y1:y2, x1:x2]
    crop_hsv = hsv[y1:y2, x1:x2]
    
    # Calculate tissue stats
    # Jassid symptoms on cotton: chlorosis, yellowing, curl, necrosis
    # Chlorotic/yellow: H in [20, 35], S > 40
    # Green: H in [36, 85], S > 30
    # Dark/shadow/soil: V < 50
    total_px = (x2 - x1) * (y2 - y1)
    green_px = np.sum((crop_hsv[:,:,0] >= 36) & (crop_hsv[:,:,0] <= 85) & (crop_hsv[:,:,1] >= 30))
    yellow_px = np.sum((crop_hsv[:,:,0] >= 18) & (crop_hsv[:,:,0] < 36) & (crop_hsv[:,:,1] >= 30))
    dark_px = np.sum(crop_hsv[:,:,2] < 50)
    
    print(f"Lesion {rep_id} (CID {c['candidate_id']}): bbox=[{x1},{y1},{x2},{y2}], area={c['area']:.1f} | Green={green_px/total_px*100:.1f}%, Yellow/Chlorotic={yellow_px/total_px*100:.1f}%, Dark={dark_px/total_px*100:.1f}%")
    cv2.imwrite(str(out_dir / f"f15_lesion_{rep_id}.png"), crop)
