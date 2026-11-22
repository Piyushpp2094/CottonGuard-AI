"""
scripts/reconcile_f21_containment_math.py
Forensic pixel-by-pixel audit of Frame 21 candidates 1327, 1334, 1337
against Blade 1, Blade 2, and their documented union.
"""

import cv2, json, numpy as np

STAGE_DIR = 'artifacts/annotation_staging/phase12_2'

with open(f'{STAGE_DIR}/candidates_stage2_2_master.json') as f:
    master_cands = json.load(f)

# Load Blade 1 (CAND_F21_L_1289)
b1_cand = next(c for c in master_cands if c['candidate_id'] == 'CAND_F21_L_1289')
b1_pts = np.array(b1_cand['polygon_coordinates']).reshape(-1, 2)
mask_b1 = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask_b1, [np.round(b1_pts).astype(np.int32)], 1)

# Load Blade 2 (CAND_F21_L_BLADE2)
b2_cand = next(c for c in master_cands if c['candidate_id'] == 'CAND_F21_L_BLADE2')
b2_pts = np.array(b2_cand['polygon_coordinates']).reshape(-1, 2)
mask_b2 = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask_b2, [np.round(b2_pts).astype(np.int32)], 1)

# Documented union of verified host leaves
mask_union = cv2.bitwise_or(mask_b1, mask_b2)

img = cv2.imread('data/detection_consolidated_100/images/LR00140.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

target_ids = [1327, 1334, 1337]
print("=" * 80)
print("FRAME 21 CONTAINMENT RECONCILIATION: CANDIDATES 1327, 1334, 1337")
print("=" * 80)

for rep_id in target_ids:
    c = next(x for x in master_cands if x.get('replaced_shape_id') == rep_id and x['frame_number'] == 21)
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    l_mask = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_mask, [np.round(pts).astype(np.int32)], 1)
    
    tot_px = int(np.sum(l_mask == 1))
    ov_b1 = int(np.sum((mask_b1 == 1) & (l_mask == 1)))
    ov_b2 = int(np.sum((mask_b2 == 1) & (l_mask == 1)))
    ov_union = int(np.sum((mask_union == 1) & (l_mask == 1)))
    
    pct_b1 = (ov_b1 / tot_px) * 100 if tot_px > 0 else 0.0
    pct_b2 = (ov_b2 / tot_px) * 100 if tot_px > 0 else 0.0
    pct_union = (ov_union / tot_px) * 100 if tot_px > 0 else 0.0
    
    bx1, by1 = pts.min(axis=0)
    bx2, by2 = pts.max(axis=0)
    
    # Check pixels outside union: what are they?
    outside_mask = (l_mask == 1) & (mask_union == 0)
    out_px = int(np.sum(outside_mask))
    if out_px > 0:
        out_bgr = img[outside_mask]
        out_hsv = hsv[outside_mask]
        out_green = np.sum((out_hsv[:,0] >= 25) & (out_hsv[:,0] <= 85) & (out_hsv[:,1] >= 30) & (out_hsv[:,2] >= 30))
        out_red = np.sum(((out_hsv[:,0] < 15) | (out_hsv[:,0] > 165)) & (out_hsv[:,1] >= 25))
        out_dark = np.sum(out_hsv[:,2] < 45)
    else:
        out_green = out_red = out_dark = 0
        
    print(f"\n[{c['candidate_id']} (rep {rep_id})]:")
    print(f"  Bbox: [{bx1:.0f}, {by1:.0f}, {bx2:.0f}, {by2:.0f}], Area: {tot_px} px, Verts: {len(pts)}")
    print(f"  Blade 1 Overlap: {ov_b1} / {tot_px} px ({pct_b1:.2f}%)")
    print(f"  Blade 2 Overlap: {ov_b2} / {tot_px} px ({pct_b2:.2f}%)")
    print(f"  Documented Union Overlap: {ov_union} / {tot_px} px ({pct_union:.2f}%)")
    print(f"  Pixels Outside Union: {out_px} px ({out_px/tot_px*100:.2f}%)")
    if out_px > 0:
        print(f"    Outside tissue breakdown: plant red={out_red/out_px*100:.1f}%, plant green={out_green/out_px*100:.1f}%, dark background/shadow={out_dark/out_px*100:.1f}%")
