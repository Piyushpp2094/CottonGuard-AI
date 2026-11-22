import cv2, json, numpy as np

with open('artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json') as f:
    master_cands = json.load(f)

# Load lesions 1341-1344
f21_top = [c for c in master_cands if c.get('replaced_shape_id') in [1341, 1342, 1343, 1344] and c['frame_number'] == 21]

mask = np.zeros((800, 800), dtype=np.uint8)
for c in f21_top:
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)

# Also add plant tissue between them in y: [0, 65], x: [280, 650]
img = cv2.imread('data/detection_consolidated_100/images/LR00140.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plant = ((hsv[:,:,0] >= 10) & (hsv[:,:,0] <= 90) & (hsv[:,:,1] >= 15)) | \
        (((hsv[:,:,0] < 15) | (hsv[:,:,0] > 165)) & (hsv[:,:,1] >= 20))
roi_plant = np.zeros((800, 800), dtype=np.uint8)
roi_plant[0:65, 280:650] = plant[0:65, 280:650].astype(np.uint8) * 255

combined = cv2.bitwise_or(mask, roi_plant)
# Smooth closing
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

cnts, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
largest = max(cnts, key=cv2.contourArea)

# Geometry validator
def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
def intersect(p1, p2, p3, p4):
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))
def valid_poly(pts):
    n = len(pts)
    if n < 15: return False
    for i in range(n):
        p1, p2 = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1: continue
            p3, p4 = pts[j], pts[(j + 1) % n]
            if intersect(p1, p2, p3, p4): return False
    return True

peri = cv2.arcLength(largest, True)
final_pts = None
for eps in np.linspace(0.001, 0.008, 100):
    approx = cv2.approxPolyDP(largest, eps * peri, True)
    p = approx.reshape(-1, 2).astype(np.float32)
    p[:, 0] = np.clip(p[:, 0], 0.0, 800.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 800.0)
    if 18 <= len(p) <= 30 and valid_poly(p):
        final_pts = p
        break

assert final_pts is not None, "Failed to find valid organic polygon for Blade 2"
print(f"Generated CAND_F21_L_BLADE2: verts={len(final_pts)}, area={cv2.contourArea(final_pts):.1f}")

# Check containment of 1341-1344
b2_mask = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(b2_mask, [np.round(final_pts).astype(np.int32)], 1)
for c in f21_top:
    l_pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    l_m = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_m, [np.round(l_pts).astype(np.int32)], 1)
    tot = np.sum(l_m == 1)
    ov = np.sum((b2_mask == 1) & (l_m == 1))
    print(f"  {c['candidate_id']}: {ov}/{tot} ({ov/tot*100:.2f}%)")

# Save to new_secondary_leaf_candidates.json
with open('artifacts/annotation_staging/phase12_2/new_secondary_leaf_candidates.json') as f:
    sec_leaves = json.load(f)

sec_leaves["CAND_F21_L_BLADE2"] = {
    "candidate_id": "CAND_F21_L_BLADE2",
    "frame_number": 21,
    "annotation_label": "Leaf",
    "label_id": 5,
    "vertex_count": len(final_pts),
    "area": float(cv2.contourArea(final_pts)),
    "polygon_coordinates": final_pts.flatten().tolist()
}

with open('artifacts/annotation_staging/phase12_2/new_secondary_leaf_candidates.json', 'w') as f:
    json.dump(sec_leaves, f, indent=2)

print("Updated new_secondary_leaf_candidates.json with organic Blade 2 polygon.")
