"""
scripts/reconcile_stage2_5_manifests.py
Updates:
1. candidates_adjudication_master.csv (143 rows: 94 ACCEPT, 49 REJECT)
2. candidates_stage2_2_master.json (155 entries: 106 ACCEPT, 49 REJECT)
3. precision_additions_manifest.json (106 additions: 12 Leaves, 94 Lesions)
4. precision_106_additions_manifest.json (alias)
"""

import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

STAGE_DIR = Path("artifacts/annotation_staging/phase12_2")
CSV_PATH = STAGE_DIR / "candidates_adjudication_master.csv"
MASTER_JSON_PATH = STAGE_DIR / "candidates_stage2_2_master.json"
OUTPUT_JSON = STAGE_DIR / "precision_additions_manifest.json"
OUTPUT_ALIAS = STAGE_DIR / "precision_106_additions_manifest.json"

df = pd.read_csv(CSV_PATH)

# Reject Frame 21 zero-containment candidates: 1329, 1330, 1331, 1335
f21_zero_ids = [1329, 1330, 1331, 1335]
for sid in f21_zero_ids:
    m = (df["frame"] == 21) & (df["original_shape_id"] == sid)
    df.loc[m, "final_decision"] = "REJECT"
    df.loc[m, "reason"] = "Out-of-focus background foliage in blurred depth plane; 0.00% containment in approved foreground leaf geometry"
    df.loc[m, "containment_percentage"] = 0.0
    df.loc[m, "human_status"] = "REJECTED"

# Reject Frame 15 zero-containment candidates: 1176, 1177, 1178, 1179
f15_zero_ids = [1176, 1177, 1178, 1179]
for sid in f15_zero_ids:
    m = (df["frame"] == 15) & (df["original_shape_id"] == sid)
    df.loc[m, "final_decision"] = "REJECT"
    if sid == 1178:
        df.loc[m, "reason"] = "Blurred background soil artifact (0.3% plant tissue); 0.00% containment in leaf geometry"
    elif sid == 1176:
        df.loc[m, "reason"] = "Inter-foliar shadow gap (73.6% non-plant); 0.00% containment in leaf geometry"
    else:
        df.loc[m, "reason"] = "Out-of-focus peripheral border foliage; 0.00% containment in approved leaf geometry"
    df.loc[m, "containment_percentage"] = 0.0
    df.loc[m, "human_status"] = "REJECTED"

# Save CSV
df.to_csv(CSV_PATH, index=False)
print(f"Updated {CSV_PATH}:")
print(f"  Total Rows: {len(df)}")
accept_cnt = sum(df["final_decision"] == "ACCEPT")
reject_cnt = sum(df["final_decision"] == "REJECT")
print(f"  ACCEPTED Lesions: {accept_cnt} (Expected: 94)")
print(f"  REJECTED Lesions: {reject_cnt} (Expected: 49)")
assert accept_cnt == 94, f"Expected 94 accepted, got {accept_cnt}"
assert reject_cnt == 49, f"Expected 49 rejected, got {reject_cnt}"

# Check rejections by frame
f15_rej = sum((df["frame"] == 15) & (df["final_decision"] == "REJECT"))
f21_rej = sum((df["frame"] == 21) & (df["final_decision"] == "REJECT"))
f22_rej = sum((df["frame"] == 22) & (df["final_decision"] == "REJECT"))
f24_rej = sum((df["frame"] == 24) & (df["final_decision"] == "REJECT"))
print(f"  Rejections breakdown: Frame 15: {f15_rej}, Frame 21: {f21_rej}, Frame 22: {f22_rej}, Frame 24: {f24_rej}")
assert f15_rej == 4, f"Expected 4 Frame 15 rejections, got {f15_rej}"
assert f21_rej == 20, f"Expected 20 Frame 21 rejections, got {f21_rej}"
assert f22_rej == 24, f"Expected 24 Frame 22 rejections, got {f22_rej}"
assert f24_rej == 1, f"Expected 1 Frame 24 rejections, got {f24_rej}"

# Update Master JSON
with open(MASTER_JSON_PATH) as f:
    master_cands = json.load(f)

for c in master_cands:
    f_num = c.get("frame_number")
    rep_id = c.get("replaced_shape_id")
    
    if f_num == 21 and rep_id in f21_zero_ids:
        c["adjudication_decision"] = "REJECT"
        c["human_review_status"] = "REJECTED"
        c["containment_percentage"] = 0.0
        c["adjudication_reason"] = "Out-of-focus background foliage in blurred depth plane; 0.00% containment in approved foreground leaf geometry"
        
    if f_num == 15 and rep_id in f15_zero_ids:
        c["adjudication_decision"] = "REJECT"
        c["human_review_status"] = "REJECTED"
        c["containment_percentage"] = 0.0
        if rep_id == 1178:
            c["adjudication_reason"] = "Blurred background soil artifact (0.3% plant tissue); 0.00% containment in leaf geometry"
        elif rep_id == 1176:
            c["adjudication_reason"] = "Inter-foliar shadow gap (73.6% non-plant); 0.00% containment in leaf geometry"
        else:
            c["adjudication_reason"] = "Out-of-focus peripheral border foliage; 0.00% containment in approved leaf geometry"

with open(MASTER_JSON_PATH, "w") as f:
    json.dump(master_cands, f, indent=2)

print(f"Updated {MASTER_JSON_PATH} with rejected decisions for 8 zero-containment candidates.")

# 3. Build Precision Additions Manifest (106 Shapes)
accepted_lesion_cids = set(df[df["final_decision"] == "ACCEPT"]["candidate_id"])
rejected_lesion_cids = set(df[df["final_decision"] == "REJECT"]["candidate_id"])

authoritative_meta = {
    0:  {"fn": "BBC00001.jpg", "gid": 50, "cls": "Bacterial Blight Cotton"},
    9:  {"fn": "CV00431.jpg",  "gid": 59, "cls": "Cotton Vein Enation Virus"},
    15: {"fn": "LHJ00001.jpg", "gid": 65, "cls": "Leaf Hopper Jassids"},
    17: {"fn": "LHJ00116.jpg", "gid": 67, "cls": "Leaf Hopper Jassids"},
    18: {"fn": "LHJ00172.jpg", "gid": 68, "cls": "Leaf Hopper Jassids"},
    19: {"fn": "LHJ00225.jpg", "gid": 69, "cls": "Leaf Hopper Jassids"},
    20: {"fn": "LR00001.jpg",  "gid": 70, "cls": "Leaf Redding"},
    21: {"fn": "LR00140.jpg",  "gid": 71, "cls": "Leaf Redding"},
    22: {"fn": "LR00286.jpg",  "gid": 72, "cls": "Leaf Redding"},
    23: {"fn": "LR00440.jpg",  "gid": 73, "cls": "Leaf Redding"},
    24: {"fn": "LR00578.jpg",  "gid": 74, "cls": "Leaf Redding"},
    26: {"fn": "LV00028.jpg",  "gid": 76, "cls": "Leaf Variegation"},
    28: {"fn": "LV00087.jpg",  "gid": 78, "cls": "Leaf Variegation"},
    31: {"fn": "HL00070.jpg",  "gid": 81, "cls": "Healthy Leaf"}
}

def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

def intersect(p1, p2, p3, p4):
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

def validate_poly(pts, is_leaf=True):
    pts = np.array(pts, dtype=np.float32).reshape(-1, 2)
    n = len(pts)
    min_v = 15 if is_leaf else 5
    if n < min_v:
        return False, f"Vertex count {n} < {min_v}"
    if np.any(pts < 0.0) or np.any(pts > 800.0):
        return False, "Coordinates out of bounds [0.0, 800.0]"
    for i in range(n):
        p1, p2 = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            p3, p4 = pts[j], pts[(j + 1) % n]
            if intersect(p1, p2, p3, p4):
                return False, f"Self-intersection between edge {i} and {j}"
    area = cv2.contourArea(pts)
    if area <= 0.0:
        return False, f"Non-positive area {area}"
    return True, f"OK (n={n}, area={area:.1f})"

manifest_106 = []
seen_cids = set()

# Add Leaves (12)
leaf_cands = [c for c in master_cands if c["annotation_label"] == "Leaf"]
print(f"\nProcessing {len(leaf_cands)} Leaf Candidates:")
assert len(leaf_cands) == 12, f"Expected 12 leaves, got {len(leaf_cands)}"

for lf in leaf_cands:
    cid = lf["candidate_id"]
    f_idx = lf["frame_number"]
    assert cid not in seen_cids, f"Duplicate CID: {cid}"
    seen_cids.add(cid)
    
    meta = authoritative_meta[f_idx]
    pts = np.array(lf["polygon_coordinates"], dtype=np.float32).reshape(-1, 2)
    ok, msg = validate_poly(pts, is_leaf=True)
    assert ok, f"Leaf {cid} failed validation: {msg}"
    
    clean_entry = {
        "candidate_id": cid,
        "frame_number": f_idx,
        "global_id": meta["gid"],
        "image_filename": meta["fn"],
        "disease_class": meta["cls"],
        "annotation_label": "Leaf",
        "label_id": 5,
        "type": "polygon",
        "replaced_shape_id": lf.get("replaced_shape_id"),
        "vertex_count": len(pts),
        "area": round(float(cv2.contourArea(pts)), 2),
        "polygon_coordinates": pts.flatten().tolist(),
        "geometry_source": lf.get("geometry_source", "UNet_ResNet34_MorphRefined"),
        "adjudication_decision": "ACCEPT",
        "human_review_status": "APPROVED",
        "adjudication_reason": lf.get("adjudication_reason", "Authoritative organic leaf contour")
    }
    manifest_106.append(clean_entry)
    print(f"  [LEAF] {cid} (Frame {f_idx:02d}, {meta['fn']}, {meta['cls']}): verts={len(pts)}, area={clean_entry['area']:.1f}")

# Add Lesions (94)
print(f"\nProcessing Approved Lesion Candidates:")
lesion_added_count = 0

for c in master_cands:
    if c["annotation_label"] != "Diseased_Region":
        continue
    cid = c["candidate_id"]
    f_idx = c["frame_number"]
    
    if cid in rejected_lesion_cids:
        continue
    if cid not in accepted_lesion_cids:
        continue
        
    assert cid not in seen_cids, f"Duplicate CID: {cid}"
    seen_cids.add(cid)
    
    meta = authoritative_meta[f_idx]
    pts = np.array(c["polygon_coordinates"], dtype=np.float32).reshape(-1, 2)
    ok, msg = validate_poly(pts, is_leaf=False)
    assert ok, f"Lesion {cid} failed validation: {msg}"
    
    clean_entry = {
        "candidate_id": cid,
        "frame_number": f_idx,
        "global_id": meta["gid"],
        "image_filename": meta["fn"],
        "disease_class": meta["cls"],
        "annotation_label": "Diseased_Region",
        "label_id": 6,
        "type": "polygon",
        "replaced_shape_id": c.get("replaced_shape_id"),
        "host_leaf_id": c.get("host_leaf_id"),
        "containment_percentage": c.get("containment_percentage"),
        "vertex_count": len(pts),
        "area": round(float(cv2.contourArea(pts)), 2),
        "polygon_coordinates": pts.flatten().tolist(),
        "geometry_source": c.get("geometry_source", "Otsu_MorphRefined"),
        "adjudication_decision": "ACCEPT",
        "human_review_status": "APPROVED",
        "adjudication_reason": c.get("adjudication_reason", "Authentic disease lesion contour")
    }
    manifest_106.append(clean_entry)
    lesion_added_count += 1

print(f"Added {lesion_added_count} accepted lesions.")
assert lesion_added_count == 94, f"Expected 94 lesions, got {lesion_added_count}"
assert len(manifest_106) == 106, f"Expected 106 total additions, got {len(manifest_106)}"

# Save 106 Manifests
with open(OUTPUT_JSON, "w") as f:
    json.dump(manifest_106, f, indent=2)
with open(OUTPUT_ALIAS, "w") as f:
    json.dump(manifest_106, f, indent=2)

print(f"\nSuccessfully wrote verified 106 additions manifest to:")
print(f"  {OUTPUT_JSON}")
print(f"  {OUTPUT_ALIAS}")

# Summary of additions by frame
print("\nFinal Additions Breakdown by Frame:")
frames_summary = {}
for item in manifest_106:
    f = item["frame_number"]
    frames_summary.setdefault(f, {"Leaf": 0, "Diseased_Region": 0, "Total": 0})
    frames_summary[f][item["annotation_label"]] += 1
    frames_summary[f]["Total"] += 1

for f in sorted(frames_summary.keys()):
    meta = authoritative_meta[f]
    info = frames_summary[f]
    print(f"  Frame {f:02d} (GID {meta['gid']}, {meta['fn']}, {meta['cls']}): {info['Total']} additions ({info['Leaf']} leaves, {info['Diseased_Region']} lesions)")

# 4. Verify Final Balance Sheet Arithmetic
baseline_count = 1170
authorized_deletions = 303
retained_baseline = baseline_count - authorized_deletions
final_cvat_count = retained_baseline + len(manifest_106)

print("\n" + "=" * 80)
print("FINAL RECONCILED TRANSACTION BALANCE SHEET (STAGE 2.5)")
print("=" * 80)
print(f"Live Baseline Shapes (Task #6 / Job #6):    {baseline_count}")
print(f"Authorized Stage 4 Deletions:             - {authorized_deletions}")
print(f"Retained Baseline Shapes:                 = {retained_baseline}")
print(f"  - Preserved DEC-01 Detection Rectangles:    37 (Frames 26 & 28)")
print(f"  - Retained Other Baseline Shapes:          830")
print(f"Verified Stage 3 Additions:               + {len(manifest_106)}")
print(f"  - Approved Leaf Polygons:                   12")
print(f"  - Approved Lesion Polygons:                 94")
print(f"Projected Final CVAT Task #6 Shape Count: = {final_cvat_count}")
print("=" * 80)
assert final_cvat_count == 973, f"Expected 973 final shapes, got {final_cvat_count}"
print("VERIFICATION SUCCESSFUL: Exact 973 final shapes confirmed!")
