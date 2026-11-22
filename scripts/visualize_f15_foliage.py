"""
scripts/visualize_f15_foliage.py
"""
import cv2, json, numpy as np

img = cv2.imread("data/detection_consolidated_100/images/LHJ00001.jpg")
vis = img.copy()

with open("artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json") as f:
    master = json.load(f)

# Primary leaf
lf1 = next(c for c in master if c["candidate_id"] == "CAND_F15_L_1157")
p_pts = np.array(lf1["polygon_coordinates"]).reshape(-1, 2).astype(np.int32)
cv2.polylines(vis, [p_pts], True, (0, 255, 0), 2)

# Secondary leaf
with open("artifacts/annotation_staging/phase12_2/new_secondary_leaf_candidates.json") as f:
    sec = json.load(f)
s_pts = np.array(sec["CAND_F15_L_SECONDARY_01"]["polygon_coordinates"]).reshape(-1, 2).astype(np.int32)
cv2.polylines(vis, [s_pts], True, (255, 255, 0), 2)

# All 15 lesions
f15_les = [c for c in master if c["frame_number"] == 15 and c["annotation_label"] == "Diseased_Region"]
for c in f15_les:
    pts = np.array(c["polygon_coordinates"]).reshape(-1, 2).astype(np.int32)
    rep_id = c["replaced_shape_id"]
    color = (0, 255, 255) if rep_id in [1173, 1174, 1175] else ((255, 100, 0) if rep_id >= 1180 else (0, 0, 255))
    cv2.polylines(vis, [pts], True, color, 2)
    cx, cy = np.mean(pts, axis=0).astype(int)
    cv2.putText(vis, str(rep_id), (cx - 10, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
    cv2.putText(vis, str(rep_id), (cx - 10, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

cv2.imwrite("artifacts/audit_phase11/f15_inspect/f15_full_visual.png", vis)
print("Saved artifacts/audit_phase11/f15_inspect/f15_full_visual.png")
