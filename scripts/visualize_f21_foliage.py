"""
scripts/visualize_f21_foliage.py
"""
import cv2, json, numpy as np

img = cv2.imread("data/detection_consolidated_100/images/LR00140.jpg")
vis = img.copy()

with open("artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json") as f:
    master = json.load(f)

# Draw Blade 1
lf1 = next(c for c in master if c["candidate_id"] == "CAND_F21_L_1289")
b1_pts = np.array(lf1["polygon_coordinates"]).reshape(-1, 2).astype(np.int32)
cv2.polylines(vis, [b1_pts], True, (0, 255, 0), 2)

# Draw Blade 2
with open("artifacts/annotation_staging/phase12_2/new_secondary_leaf_candidates.json") as f:
    sec = json.load(f)
b2_pts = np.array(sec["CAND_F21_L_BLADE2"]["polygon_coordinates"]).reshape(-1, 2).astype(np.int32)
cv2.polylines(vis, [b2_pts], True, (255, 255, 0), 2)

# Draw all 26 lesion candidates
f21_les = [c for c in master if c["frame_number"] == 21 and c["annotation_label"] == "Diseased_Region"]
for c in f21_les:
    pts = np.array(c["polygon_coordinates"]).reshape(-1, 2).astype(np.int32)
    rep_id = c["replaced_shape_id"]
    color = (0, 0, 255) if c["adjudication_decision"] == "REJECT" else (0, 255, 255)
    cv2.polylines(vis, [pts], True, color, 2)
    cx, cy = np.mean(pts, axis=0).astype(int)
    cv2.putText(vis, str(rep_id), (cx - 10, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
    cv2.putText(vis, str(rep_id), (cx - 10, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

cv2.imwrite("artifacts/audit_phase11/f21_inspect/f21_full_visual.png", vis)
print("Saved artifacts/audit_phase11/f21_inspect/f21_full_visual.png")
