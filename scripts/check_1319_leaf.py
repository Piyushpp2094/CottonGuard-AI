import cv2, json, numpy as np

with open('artifacts/annotation_staging/phase12_2/candidates_153.json') as f:
    cands = json.load(f)

f21_leaf = next(c for c in cands if c['candidate_id'] == 'CAND_F21_L_1289')
leaf_pts = np.array(f21_leaf['polygon_coordinates']).reshape(-1, 2)
leaf_mask = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(leaf_mask, [np.round(leaf_pts).astype(np.int32)], 1)

c1319 = next(c for c in cands if c['candidate_id'] == 'CAND_F21_LES_1319')
pts1319 = np.array(c1319['polygon_coordinates']).reshape(-1, 2)
m1319 = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(m1319, [np.round(pts1319).astype(np.int32)], 1)

print("1319 points:", pts1319)
print("Overlap with CAND_F21_L_1289:", np.sum((leaf_mask == 1) & (m1319 == 1)))
print("Distance from leaf boundary:")
# Find closest leaf point to 1319 center
c_1319 = np.mean(pts1319, axis=0)
dists = np.linalg.norm(leaf_pts - c_1319, axis=1)
min_idx = np.argmin(dists)
print(f"Closest leaf point: {leaf_pts[min_idx]} at distance {dists[min_idx]:.1f} px")
