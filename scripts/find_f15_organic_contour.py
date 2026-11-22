import cv2, numpy as np

img = cv2.imread('data/detection_consolidated_100/images/LHJ00001.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Combine green, chlorotic yellow, and necrotic brown
mask_green = cv2.inRange(hsv, np.array([25, 20, 20]), np.array([85, 255, 255]))
mask_yellow = cv2.inRange(hsv, np.array([12, 25, 25]), np.array([25, 255, 255]))
mask_brown = cv2.inRange(hsv, np.array([0, 20, 20]), np.array([12, 255, 255]))
plant_mask = cv2.bitwise_or(cv2.bitwise_or(mask_green, mask_yellow), mask_brown)

# Restrict to quadrant x < 380, y > 260
quad_mask = np.zeros((800, 800), dtype=np.uint8)
quad_mask[260:800, 0:380] = 1
sec_plant = cv2.bitwise_and(plant_mask, plant_mask, mask=quad_mask)

# Morphological closing with a round kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
sec_plant = cv2.morphologyEx(sec_plant, cv2.MORPH_CLOSE, kernel, iterations=3)
sec_plant = cv2.morphologyEx(sec_plant, cv2.MORPH_OPEN, kernel, iterations=1)

cnts, _ = cv2.findContours(sec_plant, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
best_cnt = max(cnts, key=cv2.contourArea)
print(f"Raw contour points: {len(best_cnt)}, area: {cv2.contourArea(best_cnt)}")

peri = cv2.arcLength(best_cnt, True)
for eps in np.linspace(0.002, 0.01, 15):
    approx = cv2.approxPolyDP(best_cnt, eps * peri, True)
    print(f"eps={eps:.4f}: verts={len(approx)}")
