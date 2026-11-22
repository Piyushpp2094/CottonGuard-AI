"""
scripts/inspect_f21_all_leaf_contours.py
"""
import cv2, json, numpy as np

img = cv2.imread("data/detection_consolidated_100/images/LR00140.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Plant mask for the whole image
is_plant = ((hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 85) & (hsv[:,:,1] >= 25) & (hsv[:,:,2] >= 30)) | \
           (((hsv[:,:,0] < 15) | (hsv[:,:,0] > 165)) & (hsv[:,:,1] >= 25) & (hsv[:,:,2] >= 30)) | \
           ((hsv[:,:,0] >= 15) & (hsv[:,:,0] < 35) & (hsv[:,:,1] >= 25) & (hsv[:,:,2] >= 30))

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
plant_mask = cv2.morphologyEx(is_plant.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

# Check location of lesions 1329, 1330, 1331, 1335
with open("artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json") as f:
    master = json.load(f)

disputed = [c for c in master if c.get("replaced_shape_id") in [1329, 1330, 1331, 1335] and c["frame_number"] == 21]

print("Check if disputed lesions are on plant tissue:")
for c in disputed:
    pts = np.array(c["polygon_coordinates"]).reshape(-1, 2)
    l_mask = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_mask, [np.round(pts).astype(np.int32)], 1)
    tot = np.sum(l_mask == 1)
    on_plant = np.sum((plant_mask == 1) & (l_mask == 1))
    print(f"  Lesion {c['replaced_shape_id']}: on_plant = {on_plant}/{tot} ({on_plant/tot*100:.1f}%)")

# Now check: Is this right foliage genuinely part of the visible cotton leaf in LR00140.jpg?
# Or is it a separate leaf blade (Blade 3 or Right Lateral Blade)?
