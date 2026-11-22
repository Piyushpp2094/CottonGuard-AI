"""
scripts/inspect_f21_right_canopy.py
"""
import cv2, json, numpy as np

img = cv2.imread("data/detection_consolidated_100/images/LR00140.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Inspect region x in [500, 800], y in [50, 400]
roi = img[50:400, 500:800]
roi_hsv = hsv[50:400, 500:800]

# Plant tissue: Green or Red or Chlorotic
# Green: H in [35, 85], S > 25, V > 30
# Red: H < 15 or H > 165, S > 25, V > 30
# Chlorotic: H in [15, 34], S > 25, V > 30
is_plant = ((roi_hsv[:,:,0] >= 35) & (roi_hsv[:,:,0] <= 85) & (roi_hsv[:,:,1] >= 25) & (roi_hsv[:,:,2] >= 30)) | \
           (((roi_hsv[:,:,0] < 15) | (roi_hsv[:,:,0] > 165)) & (roi_hsv[:,:,1] >= 25) & (roi_hsv[:,:,2] >= 30)) | \
           ((roi_hsv[:,:,0] >= 15) & (roi_hsv[:,:,0] < 35) & (roi_hsv[:,:,1] >= 25) & (roi_hsv[:,:,2] >= 30))

print("Plant percentage in right region x=[500,800], y=[50,400]:", np.sum(is_plant) / is_plant.size * 100)

# Check what the original CVAT shapes had in this region!
with open("artifacts/backups/cvat_job6_stage0_pre_remediation_snapshot_20261118_205937.json") as f:
    snap = json.load(f)

f21_shapes = [s for s in snap["shapes"] if s["frame"] == 21]
print("\nCVAT snapshot shapes in this region:")
for s in f21_shapes:
    pts = np.array(s["points"]).reshape(-1, 2)
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    if min_x >= 500 and min_y >= 50 and max_y <= 400:
        print(f"  Shape {s['id']}: type={s['type']}, label_id={s['label_id']}, bbox=[{min_x:.1f}, {min_y:.1f}, {max_x:.1f}, {max_y:.1f}]")
