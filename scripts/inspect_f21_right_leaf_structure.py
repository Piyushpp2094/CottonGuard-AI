"""
scripts/inspect_f21_right_leaf_structure.py
"""
import cv2, json, numpy as np

img = cv2.imread("data/detection_consolidated_100/images/LR00140.jpg")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

with open("artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json") as f:
    master = json.load(f)

lf1 = next(c for c in master if c["candidate_id"] == "CAND_F21_L_1289")
b1_pts = np.array(lf1["polygon_coordinates"]).reshape(-1, 2)

# Where does b1_pts cut inwards?
# Look at points 28 to 33:
# 28: [688, 411]
# 29: [628, 391]
# 30: [517, 301]
# 31: [477, 285]
# 32: [549, 142]
# 33: [572, 123]

# What is between x in [477, 750] and y in [140, 400]?
roi = img[140:400, 477:750]
print("ROI shape:", roi.shape)

# Let's inspect the edge map or gradient in that region
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 30, 100)
cv2.imwrite("artifacts/audit_phase11/f21_inspect/right_roi.png", roi)
cv2.imwrite("artifacts/audit_phase11/f21_inspect/right_edges.png", edges)

# Check if there is a distinct leaf blade or vein structure
# In LR00140.jpg (Leaf Redding), cotton leaves are palmate (typically 3 or 5 lobes)
# Blade 1 CAND_F21_L_1289 is the central/left lobe cluster.
# On the right, is it a right lateral lobe of the same palmate leaf, or a distinct leaf?
print("Saved right_roi.png and right_edges.png")
