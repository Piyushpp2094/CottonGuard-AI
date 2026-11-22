import cv2, json, numpy as np

img = cv2.imread('data/detection_consolidated_100/images/LR00140.jpg')

with open('artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json') as f:
    master_cands = json.load(f)

b1_cand = next(c for c in master_cands if c['candidate_id'] == 'CAND_F21_L_1289')
b1_pts = np.array(b1_cand['polygon_coordinates']).reshape(-1, 2).astype(np.int32)

b2_cand = next(c for c in master_cands if c['candidate_id'] == 'CAND_F21_L_BLADE2')
b2_pts = np.array(b2_cand['polygon_coordinates']).reshape(-1, 2).astype(np.int32)

for rep_id in [1327, 1334, 1337]:
    c = next(x for x in master_cands if x.get('replaced_shape_id') == rep_id and x['frame_number'] == 21)
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2).astype(np.int32)
    bx1, by1 = pts.min(axis=0)
    bx2, by2 = pts.max(axis=0)
    
    # Expand crop window by 30 px
    x1, y1 = max(0, bx1 - 30), max(0, by1 - 30)
    x2, y2 = min(800, bx2 + 30), min(800, by2 + 30)
    
    crop = img[y1:y2, x1:x2].copy()
    
    # Draw leaf contours (shifted to crop coordinates)
    cv2.polylines(crop, [b1_pts - [x1, y1]], True, (0, 255, 0), 2)
    cv2.polylines(crop, [b2_pts - [x1, y1]], True, (255, 255, 0), 2)
    # Draw lesion contour in cyan
    cv2.polylines(crop, [pts - [x1, y1]], True, (0, 255, 255), 2)
    
    out_p = f'artifacts/audit_phase11/f21_crop_{rep_id}.png'
    cv2.imwrite(out_p, crop)
    print(f"Saved {out_p}: bbox=[{bx1},{by1},{bx2},{by2}], crop=[{x1},{y1},{x2},{y2}]")
