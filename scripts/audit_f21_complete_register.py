"""
scripts/audit_f21_complete_register.py
Audits all 26 Frame 21 lesion candidates with exact pixel containment,
spectral data, and adjudication decisions.
"""

import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

IMG_PATH = Path("data/detection_consolidated_100/images/LR00140.jpg")
img = cv2.imread(str(IMG_PATH))
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

with open("artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json") as f:
    master = json.load(f)

with open("artifacts/annotation_staging/phase12_2/new_secondary_leaf_candidates.json") as f:
    sec_leaves = json.load(f)

# Host leaf 1: CAND_F21_L_1289 (Blade 1)
lf1 = next(c for c in master if c["candidate_id"] == "CAND_F21_L_1289")
pts_b1 = np.array(lf1["polygon_coordinates"]).reshape(-1, 2)
mask_b1 = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask_b1, [np.round(pts_b1).astype(np.int32)], 1)

# Host leaf 2: CAND_F21_L_BLADE2 (Blade 2)
pts_b2 = np.array(sec_leaves["CAND_F21_L_BLADE2"]["polygon_coordinates"]).reshape(-1, 2)
mask_b2 = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask_b2, [np.round(pts_b2).astype(np.int32)], 1)

# Union of Blade 1 and Blade 2
mask_union = cv2.bitwise_or(mask_b1, mask_b2)

f21_cands = [c for c in master if c["frame_number"] == 21 and c["annotation_label"] == "Diseased_Region"]
f21_cands.sort(key=lambda x: x["replaced_shape_id"])

print(f"Total Frame 21 lesion candidates: {len(f21_cands)}")

records = []
for c in f21_cands:
    cid = c["candidate_id"]
    rep_id = c["replaced_shape_id"]
    pts = np.array(c["polygon_coordinates"]).reshape(-1, 2)
    l_mask = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_mask, [np.round(pts).astype(np.int32)], 1)
    tot_px = np.sum(l_mask == 1)
    
    # Overlaps
    ov_b1 = np.sum((mask_b1 == 1) & (l_mask == 1))
    ov_b2 = np.sum((mask_b2 == 1) & (l_mask == 1))
    ov_un = np.sum((mask_union == 1) & (l_mask == 1))
    
    pct_b1 = (ov_b1 / tot_px) * 100
    pct_b2 = (ov_b2 / tot_px) * 100
    pct_un = (ov_un / tot_px) * 100
    
    # Crop spectral evidence
    x1, y1 = np.floor(pts.min(axis=0)).astype(int)
    x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(800, x2), min(800, y2)
    crop_hsv = hsv[y1:y2, x1:x2]
    crop_mask = l_mask[y1:y2, x1:x2]
    
    l_hsv = crop_hsv[crop_mask == 1]
    if len(l_hsv) > 0:
        # Red / anthocyanin in HSV: H < 15 or H > 165, with S > 30, V > 30
        red_cnt = np.sum(((l_hsv[:, 0] < 15) | (l_hsv[:, 0] > 165)) & (l_hsv[:, 1] >= 30) & (l_hsv[:, 2] >= 30))
        # Green in HSV: H in [35, 85], S >= 30
        green_cnt = np.sum((l_hsv[:, 0] >= 35) & (l_hsv[:, 0] <= 85) & (l_hsv[:, 1] >= 30))
        # Chlorotic in HSV: H in [15, 34], S >= 30
        chlor_cnt = np.sum((l_hsv[:, 0] >= 15) & (l_hsv[:, 0] < 35) & (l_hsv[:, 1] >= 30))
        red_pct = red_cnt / len(l_hsv) * 100
        green_pct = green_cnt / len(l_hsv) * 100
        chlor_pct = chlor_cnt / len(l_hsv) * 100
    else:
        red_pct, green_pct, chlor_pct = 0.0, 0.0, 0.0
        
    records.append({
        "cid": cid,
        "rep_id": rep_id,
        "area": c["area"],
        "verts": c["vertex_count"],
        "tot_px": tot_px,
        "ov_b1": ov_b1, "pct_b1": pct_b1,
        "ov_b2": ov_b2, "pct_b2": pct_b2,
        "ov_un": ov_un, "pct_un": pct_un,
        "red_pct": red_pct,
        "green_pct": green_pct,
        "chlor_pct": chlor_pct
    })

df_f21 = pd.DataFrame(records)
print(df_f21[["cid", "rep_id", "pct_b1", "pct_b2", "pct_un", "red_pct", "green_pct"]].to_string())
