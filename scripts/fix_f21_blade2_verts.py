import cv2, numpy as np

img21 = cv2.imread('data/detection_consolidated_100/images/LR00140.jpg')
hsv21 = cv2.cvtColor(img21, cv2.COLOR_BGR2HSV)

# Mask plant tissue in top stripe
mask21_top = np.zeros((800, 800), dtype=np.uint8)
tissue21 = ((hsv21[:,:,0] >= 10) & (hsv21[:,:,0] <= 90) & (hsv21[:,:,1] >= 15)) | \
           (((hsv21[:,:,0] < 15) | (hsv21[:,:,0] > 165)) & (hsv21[:,:,1] >= 20))
mask21_top[0:70, 270:650] = tissue21[0:70, 270:650]

cnts21, _ = cv2.findContours(mask21_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
best_cnt21 = max(cnts21, key=cv2.contourArea)
print(f"Top blade raw points: {len(best_cnt21)}, area: {cv2.contourArea(best_cnt21)}")

peri = cv2.arcLength(best_cnt21, True)
for eps in np.linspace(0.0005, 0.003, 20):
    approx = cv2.approxPolyDP(best_cnt21, eps * peri, True)
    print(f"eps={eps:.5f}: verts={len(approx)}")
