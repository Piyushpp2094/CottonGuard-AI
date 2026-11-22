"""
scripts/inspect_f21_blade1_provenance.py
"""
import json, cv2, numpy as np

with open("artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json") as f:
    master = json.load(f)

lf1 = next(c for c in master if c["candidate_id"] == "CAND_F21_L_1289")
print("CAND_F21_L_1289 metadata:")
print("  Geometry source:", lf1.get("geometry_source"))
print("  Replaced shape id:", lf1.get("replaced_shape_id"))
print("  Vertex count:", lf1.get("vertex_count"))
print("  Area:", lf1.get("area"))

# Let's see what the image looks like and what leaf blades exist in LR00140.jpg
img = cv2.imread("data/detection_consolidated_100/images/LR00140.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Plant mask for the whole image
is_plant = ((hsv[:,:,0] >= 35) & (hsv[:,:,0] <= 85) & (hsv[:,:,1] >= 25) & (hsv[:,:,2] >= 30)) | \
           (((hsv[:,:,0] < 15) | (hsv[:,:,0] > 165)) & (hsv[:,:,1] >= 25) & (hsv[:,:,2] >= 30)) | \
           ((hsv[:,:,0] >= 15) & (hsv[:,:,0] < 35) & (hsv[:,:,1] >= 25) & (hsv[:,:,2] >= 30))

plant_mask = (is_plant * 255).astype(np.uint8)

# Find connected components of plant mask
cnts, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\nConnected plant components in LR00140.jpg: {len(cnts)}")
for i, cnt in enumerate(cnts):
    area = cv2.contourArea(cnt)
    if area > 10000:
        x, y, w, h = cv2.boundingRect(cnt)
        print(f"  Component {i}: area={area:.1f}, bbox=[x={x}, y={y}, w={w}, h={h}]")
