"""
scripts/inspect_f21_disputed_focus.py
"""
import cv2, json, numpy as np

img = cv2.imread("data/detection_consolidated_100/images/LR00140.jpg")
with open("artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json") as f:
    master = json.load(f)

for rep_id in [1324, 1329, 1330, 1331, 1335, 1337, 1341]:
    c = next(x for x in master if x.get("replaced_shape_id") == rep_id and x["frame_number"] == 21)
    pts = np.array(c["polygon_coordinates"]).reshape(-1, 2)
    x1, y1 = np.floor(pts.min(axis=0)).astype(int)
    x2, y2 = np.ceil(pts.max(axis=0)).astype(int)
    crop = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Candidate {rep_id}: bbox=[{x1},{y1},{x2},{y2}], FocusVar={blur_score:.1f}, Area={c['area']:.1f}")
