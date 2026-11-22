import cv2, numpy as np

img = cv2.imread('data/detection_consolidated_100/images/LR00578.jpg')
# Crop region around apical lobe: x: [400, 600], y: [0, 200]
crop = img[0:200, 400:600]
cv2.imwrite('artifacts/audit_phase11/f24_apical_crop.png', crop)

# Let's inspect edges using Canny
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
print(f"Edges detected in apical crop: {np.sum(edges > 0)} pixels")
