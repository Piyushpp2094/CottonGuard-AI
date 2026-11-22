"""
scripts/inspect_f15_border_foliage.py
"""
import cv2, json, numpy as np

img = cv2.imread("data/detection_consolidated_100/images/LHJ00001.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

with open("artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json") as f:
    master = json.load(f)

for rep_id in [1176, 1177, 1178, 1179]:
    c = next(x for x in master if x.get("replaced_shape_id") == rep_id and x["frame_number"] == 15)
    pts = np.array(c["polygon_coordinates"]).reshape(-1, 2)
    x1, y1 = np.floor(pts.min(axis=0)).astype(int)
    x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
    crop = img[y1:y2, x1:x2]
    crop_hsv = hsv[y1:y2, x1:x2]
    
    # Check green or chlorotic tissue in crop
    is_plant = ((crop_hsv[:,:,0] >= 15) & (crop_hsv[:,:,0] <= 85) & (crop_hsv[:,:,1] >= 20) & (crop_hsv[:,:,2] >= 30))
    plant_pct = np.sum(is_plant) / is_plant.size * 100
    
    # Check variance / Laplacian blur to see if it is in-focus foliage or blurred background
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    print(f"Lesion {rep_id}: bbox=[{x1},{y1},{x2},{y2}], area={c['area']:.1f} | Plant%={plant_pct:.1f}%, FocusVar={blur_score:.1f}")
