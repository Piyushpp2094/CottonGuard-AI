"""
scripts/deep_audit_f21_f22.py
Deep visual and spectral audit of:
- Frame 21: candidates 1319, 1338, 1340
- Frame 22: all 35 candidates (1384 to 1418)
"""

import cv2, json, numpy as np
import pandas as pd
from pathlib import Path

# Load candidates
with open('artifacts/annotation_staging/phase12_2/candidates_153.json') as f:
    cands = json.load(f)

# Load baseline snapshot to get shape 1345 for Frame 22
with open('artifacts/backups/cvat_job6_stage0_pre_remediation_snapshot_20261118_205937.json') as f:
    cvat = json.load(f)

s1345 = next(s for s in cvat['shapes'] if s['id'] == 1345)
leaf22_pts = np.array(s1345['points']).reshape(-1, 2)
leaf22_mask = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(leaf22_mask, [np.round(leaf22_pts).astype(np.int32)], 1)

# -----------------------------------------------------------------------------
# 1. Frame 21: Inspect 1319, 1338, 1340
# -----------------------------------------------------------------------------
print("=" * 80)
print("FRAME 21: DEEP AUDIT OF 1319, 1338, 1340")
print("=" * 80)
img21 = cv2.imread('data/detection_consolidated_100/images/LR00140.jpg')
f21_cands = [c for c in cands if c['frame_number'] == 21 and c['replaced_shape_id'] in [1319, 1338, 1340]]

for c in f21_cands:
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    bx1, by1 = pts.min(axis=0)
    bx2, by2 = pts.max(axis=0)
    roi = img21[int(by1):int(by2), int(bx1):int(bx2)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green = np.sum((h >= 25) & (h <= 85) & (s >= 30) & (v >= 30))
    red = np.sum(((h < 15) | (h > 165)) & (s >= 25))
    dark = np.sum(v < 40)
    tot = roi.shape[0] * roi.shape[1] if roi.size > 0 else 1
    mean_bgr = np.mean(roi, axis=(0, 1)) if roi.size > 0 else [0, 0, 0]
    
    print(f"\n{c['candidate_id']} (rep {c['replaced_shape_id']}):")
    area = cv2.contourArea(pts.astype(np.float32))
    print(f"  bbox: [{bx1:.0f}, {by1:.0f}, {bx2:.0f}, {by2:.0f}], area: {area:.1f} px, verts: {len(pts)}")
    print(f"  mean BGR: {mean_bgr.round(1)}")
    print(f"  tissue: red={red/tot*100:.1f}%, green={green/tot*100:.1f}%, dark/shadow={dark/tot*100:.1f}%")

# -----------------------------------------------------------------------------
# 2. Frame 22: Deep Audit of ALL 35 Candidates
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FRAME 22: DEEP AUDIT OF ALL 35 CANDIDATES")
print("=" * 80)
img22 = cv2.imread('data/detection_consolidated_100/images/LR00286.jpg')
f22_cands = [c for c in cands if c['frame_number'] == 22 and c['annotation_label'] == 'Diseased_Region']

f22_records = []
for c in f22_cands:
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    bx1, by1 = pts.min(axis=0)
    bx2, by2 = pts.max(axis=0)
    roi = img22[int(by1):int(by2), int(bx1):int(bx2)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green = np.sum((h >= 25) & (h <= 85) & (s >= 30) & (v >= 30))
    red = np.sum(((h < 15) | (h > 165)) & (s >= 25))
    chlorotic = np.sum((h >= 15) & (h < 25) & (s >= 30) & (v >= 30))
    dark = np.sum(v < 40)
    tot = roi.shape[0] * roi.shape[1] if roi.size > 0 else 1
    mean_bgr = np.mean(roi, axis=(0, 1)) if roi.size > 0 else [0, 0, 0]
    
    # Check containment in shape 1345
    l_mask = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_mask, [np.round(pts).astype(np.int32)], 1)
    ov = np.sum((leaf22_mask == 1) & (l_mask == 1))
    l_tot = np.sum(l_mask == 1)
    cont = ov / l_tot if l_tot > 0 else 0.0
    
    # Distance from center
    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
    
    area = cv2.contourArea(pts.astype(np.float32))
    f22_records.append({
        "cid": c['candidate_id'],
        "rep_id": c['replaced_shape_id'],
        "bbox": f"[{bx1:.0f}, {by1:.0f}, {bx2:.0f}, {by2:.0f}]",
        "cx": round(cx, 1),
        "cy": round(cy, 1),
        "area": round(area, 1),
        "verts": len(pts),
        "containment": round(cont * 100, 1),
        "red_pct": round(red / tot * 100, 1),
        "green_pct": round(green / tot * 100, 1),
        "chlorotic_pct": round(chlorotic / tot * 100, 1),
        "dark_pct": round(dark / tot * 100, 1),
        "mean_bgr": [round(x, 1) for x in mean_bgr]
    })

f22_df = pd.DataFrame(f22_records)
for _, r in f22_df.iterrows():
    print(f"[{r['rep_id']}] bbox={r['bbox']}, area={r['area']}, cont={r['containment']}%, red={r['red_pct']}%, green={r['green_pct']}%, chlor={r['chlorotic_pct']}%, bgr={r['mean_bgr']}")
