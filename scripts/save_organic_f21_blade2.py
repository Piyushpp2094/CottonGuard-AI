import cv2, json, numpy as np

with open('artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json') as f:
    master_cands = json.load(f)

f21_top = [c for c in master_cands if c.get('replaced_shape_id') in [1341, 1342, 1343, 1344] and c['frame_number'] == 21]

mask = np.zeros((800, 800), dtype=np.uint8)
for c in f21_top:
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
dilated = cv2.dilate(mask, kernel, iterations=1)
# Ensure top border touches y=0
dilated[0:3, 270:650] = 255

cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
largest = max(cnts, key=cv2.contourArea)
peri = cv2.arcLength(largest, True)

approx = cv2.approxPolyDP(largest, 0.002 * peri, True)
pts = approx.reshape(-1, 2).astype(np.float32)
pts[:, 0] = np.clip(pts[:, 0], 0.0, 800.0)
pts[:, 1] = np.clip(pts[:, 1], 0.0, 800.0)

# Verify geometry
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

assert valid_poly(pts), "Blade 2 polygon failed geometry check"
area = float(cv2.contourArea(pts))
print(f"Generated valid organic CAND_F21_L_BLADE2: verts={len(pts)}, area={area:.1f}")

# Check containment
b2_mask = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(b2_mask, [np.round(pts).astype(np.int32)], 1)

for c in f21_top:
    l_pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    l_m = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_m, [np.round(l_pts).astype(np.int32)], 1)
    tot = np.sum(l_m == 1)
    ov = np.sum((b2_mask == 1) & (l_m == 1))
    print(f"  {c['candidate_id']}: {ov}/{tot} ({ov/tot*100:.2f}%)")
    assert ov == tot, f"Lesion {c['candidate_id']} not 100% contained!"

# Save to new_secondary_leaf_candidates.json
with open('artifacts/annotation_staging/phase12_2/new_secondary_leaf_candidates.json') as f:
    sec_leaves = json.load(f)

sec_leaves["CAND_F21_L_BLADE2"] = {
    "candidate_id": "CAND_F21_L_BLADE2",
    "frame_number": 21,
    "annotation_label": "Leaf",
    "label_id": 5,
    "vertex_count": len(pts),
    "area": area,
    "polygon_coordinates": pts.flatten().tolist()
}

with open('artifacts/annotation_staging/phase12_2/new_secondary_leaf_candidates.json', 'w') as f:
    json.dump(sec_leaves, f, indent=2)

# Also update candidates_stage2_2_master.json
for c in master_cands:
    if c.get("candidate_id") == "CAND_F21_L_BLADE2":
        c["vertex_count"] = len(pts)
        c["area"] = area
        c["polygon_coordinates"] = pts.flatten().tolist()
        break

with open('artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json', 'w') as f:
    json.dump(master_cands, f, indent=2)

print("Updated manifests with organic CAND_F21_L_BLADE2.")
