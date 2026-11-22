import cv2, numpy as np

img = cv2.imread('data/detection_consolidated_100/images/LHJ00001.jpg')
# Check image dimensions
print("Image shape:", img.shape)

# Let's save a downsampled or annotated view showing where the 7 lesions and 8 lesions are
with open('artifacts/annotation_staging/phase12_2/candidates_153.json') as f:
    import json
    cands = json.load(f)

f15_cands = [c for c in cands if c['frame_number'] == 15]
vis = img.copy()
for c in f15_cands:
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2).astype(np.int32)
    color = (0, 0, 255) if c['replaced_shape_id'] in range(1173, 1180) else (255, 0, 0)
    cv2.polylines(vis, [pts], True, color, 2)
    cv2.putText(vis, str(c['replaced_shape_id']), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

cv2.imwrite('artifacts/audit_phase11/f15_all_annotations.png', vis)
print("Saved f15_all_annotations.png")
