import cv2, json, numpy as np

with open('artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json') as f:
    master_cands = json.load(f)

b1_cand = next(c for c in master_cands if c['candidate_id'] == 'CAND_F21_L_1289')
b1_pts = np.array(b1_cand['polygon_coordinates']).reshape(-1, 2)

# Check points of b1_pts near [300, 150]
dists = np.linalg.norm(b1_pts - np.array([313, 158]), axis=1)
nearby_indices = np.where(dists < 100)[0]
print("Blade 1 points near [313, 158]:")
for idx in nearby_indices:
    print(f"  Point {idx}: {b1_pts[idx]} (dist: {dists[idx]:.1f} px)")

# Check if 1334 is inside the raw UNet contour or adjacent blade
img = cv2.imread('data/detection_consolidated_100/images/LR00140.jpg')
roi = img[140:180, 295:335]
print("1334 mean BGR:", np.mean(roi, axis=(0, 1)))
