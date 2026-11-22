import cv2, json, numpy as np

with open('artifacts/annotation_staging/phase12_2/candidates_153.json') as f:
    cands = json.load(f)

f15_leaf = next(c for c in cands if c['candidate_id'] == 'CAND_F15_L_1157')
pts = np.array(f15_leaf['polygon_coordinates']).reshape(-1, 2)
mask = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask, [pts.astype(np.int32)], 1)

f15_7 = [c for c in cands if c['frame_number'] == 15 and c['replaced_shape_id'] in range(1173, 1180)]
for c in f15_7:
    l_pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    l_mask = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_mask, [l_pts.astype(np.int32)], 1)
    overlap = np.sum((mask == 1) & (l_mask == 1))
    total = np.sum(l_mask == 1)
    ratio = overlap / total if total > 0 else 0
    print(f"{c['candidate_id']}: total_px={total}, overlap_with_f15_leaf={overlap} (ratio: {ratio:.4f})")
