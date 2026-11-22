"""
scripts/generate_stage2_1_corrected_overlays.py
Renders high-resolution 4-panel corrected visual review packages for:
- Frame 15 (LHJ00001.jpg)
- Frame 21 (LR00140.jpg)
- Frame 22 (LR00286.jpg)
- Frame 24 (LR00578.jpg)
"""

import cv2, json, shutil
import numpy as np
from pathlib import Path

STAGE_DIR = Path("artifacts/annotation_staging/phase12_2")
OUTPUT_DIR = STAGE_DIR / "stage2_1_corrected_overlays"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_DIR = Path(r"C:\Users\Haker's pc\.gemini\antigravity\brain\e96a0217-11ef-4314-af5b-e8eaa45d475a\phase12_2_review")
BRAIN_DIR.mkdir(parents=True, exist_ok=True)

with open(STAGE_DIR / "candidates_153.json") as f:
    cands_153 = json.load(f)

with open("artifacts/backups/cvat_job6_stage0_pre_remediation_snapshot_20261118_205937.json") as f:
    cvat_snapshot = json.load(f)

# Load refined leaves generated in our analysis
with open(STAGE_DIR / "new_secondary_leaf_candidates.json") as f:
    new_leaves = json.load(f)

# Also refined F24 leaf from scripts/refine_f24_leaf.py
# Let's compute it inline to have complete precision
def get_f24_refined_pts():
    f24_leaf = next(c for c in cands_153 if c['candidate_id'] == 'CAND_F24_L_1462')
    pts = np.array(f24_leaf['polygon_coordinates']).reshape(-1, 2)
    mask = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    img = cv2.imread('data/detection_consolidated_100/images/LR00578.jpg')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    apical_roi = np.zeros((800, 800), dtype=np.uint8)
    apical_roi[0:120, 410:580] = 1
    plant_apical = ((hsv[:,:,0] >= 10) & (hsv[:,:,0] <= 90) & (hsv[:,:,1] >= 15)) | \
                   (((hsv[:,:,0] < 15) | (hsv[:,:,0] > 165)) & (hsv[:,:,1] >= 20))
    plant_apical = cv2.bitwise_and(plant_apical.astype(np.uint8) * 255, apical_roi * 255)
    combined = cv2.bitwise_or(mask, plant_apical)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest = max(cnts, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest, 0.003 * cv2.arcLength(largest, True), True)
    p = approx.reshape(-1, 2).astype(np.float32)
    p[:, 0] = np.clip(p[:, 0], 0.0, 800.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 800.0)
    return p

f24_refined_leaf_pts = get_f24_refined_pts()

frames_to_render = [15, 21, 22, 24]
frame_files = {
    15: ("LHJ00001.jpg", "Leaf Hopper Jassids"),
    21: ("LR00140.jpg", "Leaf Redding"),
    22: ("LR00286.jpg", "Leaf Redding"),
    24: ("LR00578.jpg", "Leaf Redding")
}

for f_idx in frames_to_render:
    fn, disease = frame_files[f_idx]
    img_path = Path("data/detection_consolidated_100/images") / fn
    orig = cv2.imread(str(img_path))
    h, w = orig.shape[:2]

    # Panel 1: Original Image with title
    p1 = orig.copy()
    cv2.putText(p1, f"Frame {f_idx:02d}: {fn} ({disease})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(p1, "Panel 1: Raw Original Image", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)

    # Panel 2: Baseline Annotations from CVAT Task #6 snapshot
    p2 = orig.copy()
    cv2.putText(p2, "Panel 2: Baseline CVAT Job #6", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    baseline_shapes = [s for s in cvat_snapshot['shapes'] if s['frame'] == f_idx]
    for s in baseline_shapes:
        pts = np.array(s['points']).reshape(-1, 2).astype(np.int32)
        if s['type'] == 'rectangle':
            x1, y1, x2, y2 = pts[0][0], pts[0][1], pts[1][0], pts[1][1]
            cv2.rectangle(p2, (x1, y1), (x2, y2), (0, 165, 255), 1)
        elif s['label_id'] == 5: # Leaf
            cv2.polylines(p2, [pts], True, (0, 255, 0), 2)
        else: # Lesion poly
            cv2.polylines(p2, [pts], True, (0, 0, 255), 1)
    cv2.putText(p2, f"Total Shapes: {len(baseline_shapes)}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Panel 3: Corrected Leaf Blade Contours
    p3 = orig.copy()
    cv2.putText(p3, "Panel 3: Corrected Leaf Boundaries", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    # Primary leaf
    if f_idx == 22:
        s1345 = next(s for s in cvat_snapshot['shapes'] if s['id'] == 1345)
        l_pts = np.array(s1345['points']).reshape(-1, 2).astype(np.int32)
        cv2.polylines(p3, [l_pts], True, (0, 255, 0), 2)
        cv2.putText(p3, "Leaf 1345 (Preserved Canopy Poly)", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    elif f_idx == 24:
        cv2.polylines(p3, [np.round(f24_refined_leaf_pts).astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(p3, "Refined Leaf 1462 (Includes Apical Lobe)", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    else:
        # F15, F21 primary leaves
        prim_leaf = next(c for c in cands_153 if c['frame_number'] == f_idx and c['annotation_label'] == 'Leaf')
        l_pts = np.array(prim_leaf['polygon_coordinates']).reshape(-1, 2).astype(np.int32)
        cv2.polylines(p3, [l_pts], True, (0, 255, 0), 2)
        cv2.putText(p3, f"Primary Leaf: {prim_leaf['candidate_id']}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Secondary leaves
        if f_idx == 15 and "CAND_F15_L_SECONDARY_01" in new_leaves:
            s_pts = np.array(new_leaves["CAND_F15_L_SECONDARY_01"]['polygon_coordinates']).reshape(-1, 2).astype(np.int32)
            cv2.polylines(p3, [s_pts], True, (255, 255, 0), 2)
            cv2.putText(p3, "Secondary Leaf: CAND_F15_L_SECONDARY_01", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        elif f_idx == 21 and "CAND_F21_L_BLADE2" in new_leaves:
            s_pts = np.array(new_leaves["CAND_F21_L_BLADE2"]['polygon_coordinates']).reshape(-1, 2).astype(np.int32)
            cv2.polylines(p3, [s_pts], True, (255, 255, 0), 2)
            cv2.putText(p3, "Top Blade 2: CAND_F21_L_BLADE2", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    # Panel 4: Adjudicated Lesions & Dispositions
    p4 = orig.copy()
    cv2.putText(p4, "Panel 4: Stage 2.1 Adjudicated Lesions", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    f_lesions = [c for c in cands_153 if c['frame_number'] == f_idx and c['annotation_label'] == 'Diseased_Region']
    
    app_cnt = 0
    rej_cnt = 0
    for c in f_lesions:
        pts = np.array(c['polygon_coordinates']).reshape(-1, 2).astype(np.int32)
        cid = c['replaced_shape_id']
        # Adjudication rules
        is_rejected = False
        if f_idx == 21 and cid in [1320, 1321, 1322, 1323, 1325, 1326, 1328, 1332, 1333, 1336, 1339]:
            is_rejected = True # Healthy parenchyma
        elif f_idx == 24 and cid == 1495:
            is_rejected = True # 96.5% healthy green tissue
        elif f_idx == 22 and cid in [1385, 1387, 1395, 1397, 1400, 1403, 1405, 1408]:
            is_rejected = True # >90% healthy green parenchyma
            
        if is_rejected:
            rej_cnt += 1
            cv2.polylines(p4, [pts], True, (0, 0, 255), 1) # Red for rejected
        else:
            app_cnt += 1
            cv2.polylines(p4, [pts], True, (0, 255, 255), 1) # Yellow for approved lesion
            
    cv2.putText(p4, f"Approved Lesions: {app_cnt} (Yellow)", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(p4, f"Rejected (Healthy/Margin): {rej_cnt} (Red)", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # Combine 4 panels: 2x2 grid (1600x1600)
    top_row = np.hstack([p1, p2])
    bot_row = np.hstack([p3, p4])
    combined_4panel = np.vstack([top_row, bot_row])

    out_fn = f"frame{f_idx:02d}_{Path(fn).stem}_stage2_1_corrected_overlay.png"
    out_path = OUTPUT_DIR / out_fn
    cv2.imwrite(str(out_path), combined_4panel)
    shutil.copy(str(out_path), str(BRAIN_DIR / out_fn))
    print(f"Generated and copied: {out_fn} ({combined_4panel.shape[1]}x{combined_4panel.shape[0]})")

print("\nAll 4-panel corrected overlays generated successfully.")
