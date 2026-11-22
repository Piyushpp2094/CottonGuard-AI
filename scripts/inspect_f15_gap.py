import cv2, numpy as np

img = cv2.imread("data/detection_consolidated_100/images/LHJ00001.jpg")
roi = img[350:550, 70:150]
print("Mean BGR in gap between blades:", np.mean(roi, axis=(0, 1)))
print("Min BGR:", np.min(roi, axis=(0, 1)))
print("Max BGR:", np.max(roi, axis=(0, 1)))

# Check if it's dark soil / shadow background
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
print("Mean brightness in gap:", np.mean(gray))
