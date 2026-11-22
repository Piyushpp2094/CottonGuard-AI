import cv2, json, numpy as np

img = cv2.imread('data/detection_consolidated_100/images/LR00578.jpg')
with open('artifacts/annotation_staging/phase12_2/candidates_153.json') as f:
    cands = json.load(f)

f24_leaf = next(c for c in cands if c['candidate_id'] == 'CAND_F24_L_1462')
pts = np.array(f24_leaf['polygon_coordinates']).reshape(-1, 2)
print("F24 leaf top-most points (y < 120):")
top_pts = pts[pts[:, 1] < 120]
print(top_pts)

# Inspect ROI [0:180, 400:600]
roi = img[0:180, 400:600]
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
plant = ((hsv[:,:,0] >= 10) & (hsv[:,:,0] <= 85) & (hsv[:,:,1] >= 20)) | (((hsv[:,:,0] < 10) | (hsv[:,:,0] > 165)) & (hsv[:,:,1] >= 20))
print(f"Plant pixels in ROI [0:180, 400:600]: {np.sum(plant)} / {roi.shape[0]*roi.shape[1]} ({np.mean(plant)*100:.1f}%)")
