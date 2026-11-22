import cv2, json, numpy as np

img = cv2.imread("data/detection_consolidated_100/images/LHJ00001.jpg")
with open('artifacts/annotation_staging/phase12_2/candidates_153.json') as f:
    cands = json.load(f)

f15_7 = [c for c in cands if c['frame_number'] == 15 and c['replaced_shape_id'] in range(1173, 1180)]
for c in f15_7:
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    bx1, by1 = pts.min(axis=0)
    bx2, by2 = pts.max(axis=0)
    print(f"{c['candidate_id']}: bbox=[{bx1:.0f}, {by1:.0f}, {bx2:.0f}, {by2:.0f}]")
