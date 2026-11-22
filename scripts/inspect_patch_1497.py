import cv2, numpy as np

img = cv2.imread('data/detection_consolidated_100/images/LR00578.jpg')
roi = img[57:91, 488:570]
print("Mean BGR of 1497 roi:", np.mean(roi, axis=(0, 1)))
print("Min BGR:", np.min(roi, axis=(0, 1)))
print("Max BGR:", np.max(roi, axis=(0, 1)))

# Let's inspect the surrounding 100x100 patch
patch = img[30:130, 460:590]
cv2.imwrite('artifacts/audit_phase11/patch_1497.png', patch)
print("Saved patch_1497.png")
