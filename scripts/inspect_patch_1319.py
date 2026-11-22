import cv2, numpy as np

img = cv2.imread('data/detection_consolidated_100/images/LR00140.jpg')
patch = img[750:800, 550:660]
cv2.imwrite('artifacts/audit_phase11/patch_1319.png', patch)
print("Saved patch_1319.png. Patch shape:", patch.shape)
