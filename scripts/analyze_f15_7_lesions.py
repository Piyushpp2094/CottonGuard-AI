import cv2, json, numpy as np

img = cv2.imread('data/detection_consolidated_100/images/LHJ00001.jpg')
with open('artifacts/annotation_staging/phase12_2/candidates_153.json') as f:
    cands = json.load(f)

f15_7 = [c for c in cands if c['frame_number'] == 15 and c['replaced_shape_id'] in range(1173, 1180)]
for c in f15_7:
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    bx1, by1 = np.min(pts, axis=0)
    bx2, by2 = np.max(pts, axis=0)
    roi = img[int(by1):int(by2), int(bx1):int(bx2)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    green = np.sum((h >= 25) & (h <= 85) & (s >= 30) & (v >= 30))
    yellow = np.sum((h >= 12) & (h < 25) & (s >= 30) & (v >= 50))
    brown = np.sum((h < 12) & (s >= 25))
    total = roi.shape[0] * roi.shape[1]
    print(f"{c['candidate_id']}: bbox=[{bx1:.0f}, {by1:.0f}, {bx2:.0f}, {by2:.0f}], size={roi.shape[1]}x{roi.shape[0]}, green={green/total*100:.1f}%, chlorotic={yellow/total*100:.1f}%, necrotic={brown/total*100:.1f}%")
