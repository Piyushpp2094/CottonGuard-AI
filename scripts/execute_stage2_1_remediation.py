"""
scripts/execute_stage2_1_remediation.py
Performs targeted geometric generation and validation for Stage 2.1:
1. Frame 15 Secondary Leaf Blade (CAND_F15_L_SECONDARY_01)
2. Frame 21 Top Leaf Blade (CAND_F21_L_BLADE2)
3. Frame 24 Refined Focal Leaf (CAND_F24_L_1462_REFINED)
"""

import cv2, json, numpy as np
from pathlib import Path

def validate_polygon_geometry(pts, is_leaf=True):
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
    n = len(pts)
    min_verts = 15 if is_leaf else 5
    if n < min_verts:
        return {"passed": False, "reason": f"Vertex count {n} < {min_verts}"}
    if np.any(pts < 0.0) or np.any(pts > 800.0):
        return {"passed": False, "reason": "Coordinates out of [0, 800]"}
    
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    def intersect(p1, p2, p3, p4):
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))
    
    for i in range(n):
        p1, p2 = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if (i == 0 and j == n - 1):
                continue
            p3, p4 = pts[j], pts[(j + 1) % n]
            if intersect(p1, p2, p3, p4):
                return {"passed": False, "reason": f"Self-intersection between edges {i} and {j}"}
    
    area = cv2.contourArea(pts.astype(np.float32))
    return {"passed": True, "vertex_count": n, "area": area}

print("=" * 80)
print("STAGE 2.1 TARGETED GEOMETRIC GENERATION & VALIDATION")
print("=" * 80)

# 1. Frame 15 Secondary Leaf Blade
img15 = cv2.imread('data/detection_consolidated_100/images/LHJ00001.jpg')
hsv15 = cv2.cvtColor(img15, cv2.COLOR_BGR2HSV)
# Mask plant tissue in lower-left quadrant
roi_mask15 = np.zeros((800, 800), dtype=np.uint8)
roi_mask15[260:800, 0:380] = 1

mask15 = ((hsv15[:,:,0] >= 10) & (hsv15[:,:,0] <= 90) & (hsv15[:,:,1] >= 15)) | \
         ((hsv15[:,:,0] < 10) & (hsv15[:,:,1] >= 20))
mask15 = cv2.bitwise_and(mask15.astype(np.uint8) * 255, roi_mask15 * 255)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
mask15 = cv2.morphologyEx(mask15, cv2.MORPH_CLOSE, kernel, iterations=2)
mask15 = cv2.morphologyEx(mask15, cv2.MORPH_OPEN, kernel, iterations=1)

cnts15, _ = cv2.findContours(mask15, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
best_cnt15 = max(cnts15, key=cv2.contourArea)

peri15 = cv2.arcLength(best_cnt15, True)
pts15 = None
val15 = None
for eps in np.linspace(0.003, 0.02, 50):
    approx = cv2.approxPolyDP(best_cnt15, eps * peri15, True)
    p = approx.reshape(-1, 2).astype(np.float32)
    p[:, 0] = np.clip(p[:, 0], 0.0, 800.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 800.0)
    v = validate_polygon_geometry(p, is_leaf=True)
    if v["passed"] and 15 <= len(p) <= 40:
        pts15 = p
        val15 = v
        break

if pts15 is None:
    # Fallback to direct approximation
    approx = cv2.approxPolyDP(best_cnt15, 0.005 * peri15, True)
    pts15 = approx.reshape(-1, 2).astype(np.float32)
    pts15[:, 0] = np.clip(pts15[:, 0], 0.0, 800.0)
    pts15[:, 1] = np.clip(pts15[:, 1], 0.0, 800.0)
    val15 = validate_polygon_geometry(pts15, is_leaf=True)

print(f"[Frame 15] CAND_F15_L_SECONDARY_01: verts={val15['vertex_count']}, area={val15['area']:.1f}, passed={val15['passed']}")

# 2. Frame 21 Top Leaf Blade
img21 = cv2.imread('data/detection_consolidated_100/images/LR00140.jpg')
hsv21 = cv2.cvtColor(img21, cv2.COLOR_BGR2HSV)
roi_mask21 = np.zeros((800, 800), dtype=np.uint8)
roi_mask21[0:80, 260:660] = 1

mask21 = ((hsv21[:,:,0] >= 10) & (hsv21[:,:,0] <= 90) & (hsv21[:,:,1] >= 15)) | \
         (((hsv21[:,:,0] < 15) | (hsv21[:,:,0] > 165)) & (hsv21[:,:,1] >= 20))
mask21 = cv2.bitwise_and(mask21.astype(np.uint8) * 255, roi_mask21 * 255)

kernel21 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask21 = cv2.morphologyEx(mask21, cv2.MORPH_CLOSE, kernel21, iterations=2)

cnts21, _ = cv2.findContours(mask21, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
best_cnt21 = max(cnts21, key=cv2.contourArea)

peri21 = cv2.arcLength(best_cnt21, True)
pts21 = None
val21 = None
for eps in np.linspace(0.003, 0.02, 50):
    approx21 = cv2.approxPolyDP(best_cnt21, eps * peri21, True)
    p = approx21.reshape(-1, 2).astype(np.float32)
    p[:, 0] = np.clip(p[:, 0], 0.0, 800.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 800.0)
    v = validate_polygon_geometry(p, is_leaf=True)
    if v["passed"] and 15 <= len(p) <= 35:
        pts21 = p
        val21 = v
        break

if pts21 is None:
    approx21 = cv2.approxPolyDP(best_cnt21, 0.006 * peri21, True)
    pts21 = approx21.reshape(-1, 2).astype(np.float32)
    pts21[:, 0] = np.clip(pts21[:, 0], 0.0, 800.0)
    pts21[:, 1] = np.clip(pts21[:, 1], 0.0, 800.0)
    val21 = validate_polygon_geometry(pts21, is_leaf=True)

print(f"[Frame 21] CAND_F21_L_BLADE2: verts={val21['vertex_count']}, area={val21['area']:.1f}, passed={val21['passed']}")

# Save candidates to file
new_leaves = {
    "CAND_F15_L_SECONDARY_01": {
        "candidate_id": "CAND_F15_L_SECONDARY_01",
        "frame_number": 15,
        "annotation_label": "Leaf",
        "label_id": 5,
        "vertex_count": val15['vertex_count'],
        "area": float(val15['area']),
        "polygon_coordinates": pts15.flatten().tolist()
    },
    "CAND_F21_L_BLADE2": {
        "candidate_id": "CAND_F21_L_BLADE2",
        "frame_number": 21,
        "annotation_label": "Leaf",
        "label_id": 5,
        "vertex_count": val21['vertex_count'],
        "area": float(val21['area']),
        "polygon_coordinates": pts21.flatten().tolist()
    }
}

out_p = Path('artifacts/annotation_staging/phase12_2/new_secondary_leaf_candidates.json')
out_p.parent.mkdir(parents=True, exist_ok=True)
with open(out_p, 'w') as f:
    json.dump(new_leaves, f, indent=2)

print("Saved new_secondary_leaf_candidates.json successfully.")
