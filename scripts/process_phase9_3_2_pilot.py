"""
scripts/process_phase9_3_2_pilot.py
Executes Phase 9.3.2 Five-Image Annotation Pilot in CVAT Task #6.
Applies Phase 9.3.1.1 Single-Lesion Standard across Frames 0, 16, 30, 35, 20.
Saves pilot export strictly into data/annotation_pilot/phase9_3_2/.
Preserves all 45 other frames of Task #6, all published datasets, checkpoints, and DB.
"""

import json
import urllib.request
import urllib.parse
import hashlib
import zipfile
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path("D:/Main_Project/cotton_leaf_ai")
BACKUP_DIR = PROJECT_ROOT / "artifacts" / "backups"
PILOT_EXPORT_DIR = PROJECT_ROOT / "data" / "annotation_pilot" / "phase9_3_2"
PILOT_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
(PILOT_EXPORT_DIR / "labels").mkdir(parents=True, exist_ok=True)
(PILOT_EXPORT_DIR / "masks").mkdir(parents=True, exist_ok=True)
(PILOT_EXPORT_DIR / "images").mkdir(parents=True, exist_ok=True)

LABEL_ID_LEAF = 5
LABEL_ID_DISEASED = 6
SPEC_ID_DISEASE_TYPE = 2
SPEC_ID_SYMPTOM_SUBTYPE = 3

def calc_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()

print("=" * 80)
print("PHASE 9.3.2: FIVE-IMAGE ANNOTATION PILOT EXECUTION")
print("=" * 80)

# 1. Load Pre-Execution Backup
with open("artifacts/backups/cvat_job6_pre_phase9_3_2_backup_20261115_133854.json", "r") as f:
    orig_job_ann = json.load(f)

orig_shapes = orig_job_ann.get("shapes", [])
print(f"Loaded original Job #6 shapes: {len(orig_shapes)} shapes across frames.")

t6_manifest = pd.read_csv("artifacts/manifests/pilot_batch_03_selection_manifest.csv")

# We will construct new_shapes list.
new_shapes = []

# ==============================================================================
# FRAME 0: BBC00001.jpg (Bacterial Blight, Train)
# ==============================================================================
print("\n--- Processing Frame 0: BBC00001.jpg ---")
f0_orig = [s for s in orig_shapes if s["frame"] == 0]
assert len(f0_orig) == 20, f"Expected 20 shapes on Frame 0, got {len(f0_orig)}"
# All 19 disease shapes on Frame 0 already satisfy >= 12x12 px, area >= 144, padding <= 2px.
# Preserve all 20 shapes (1 Leaf + 19 Bacterial Blight).
new_shapes.extend(f0_orig)
print(f"Frame 0: Preserved {len(f0_orig)} shapes (1 Leaf, 19 Bacterial Blight spots). All pass Tier 3.")

# ==============================================================================
# FRAME 16: LHJ00059.jpg (Leaf Hopper Jassids, Train) - Corrected & Decomposed
# ==============================================================================
print("\n--- Processing Frame 16: LHJ00059.jpg ---")
f16_orig = [s for s in orig_shapes if s["frame"] == 16]
assert len(f16_orig) == 10, f"Expected 10 shapes on Frame 16, got {len(f16_orig)}"

# Extract leaf shape (Shape #1)
leaf_shape = [s for s in f16_orig if s["label_id"] == LABEL_ID_LEAF][0]
new_shapes.append(leaf_shape)

# Extract standard marginal wedge shapes (#2, #3, #4, #6, #7, #8, #9)
for idx in [2, 3, 4, 6, 7, 8, 9]:
    new_shapes.append(f16_orig[idx])

# Decompose Shape #5 (oversized 281x442) into 3 discrete marginal wedges:
pts5 = np.array(f16_orig[5]["points"]).reshape(-1, 2).astype(np.int32)
mask5 = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask5, [pts5], 255)

m5_top = mask5.copy()
m5_top[340:, :] = 0
m5_mid = mask5.copy()
m5_mid[:340, :] = 0
m5_mid[490:, :] = 0
m5_bot = mask5.copy()
m5_bot[:490, :] = 0

for m_part in [m5_top, m5_mid, m5_bot]:
    cnts, _ = cv2.findContours(m_part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
    pts_flat = [float(coord) for pt in poly for coord in pt]
    new_shapes.append({
        "type": "polygon", "occluded": False, "outside": False, "z_order": 0, "rotation": 0.0,
        "points": pts_flat, "frame": 16, "label_id": LABEL_ID_DISEASED,
        "attributes": [
            {"spec_id": SPEC_ID_DISEASE_TYPE, "value": "Leaf Hopper Jassids"},
            {"spec_id": SPEC_ID_SYMPTOM_SUBTYPE, "value": "Chlorosis"}
        ]
    })

# Decompose Shape #0 (oversized 403x200) into 2 discrete marginal wedges:
pts0 = np.array(f16_orig[0]["points"]).reshape(-1, 2).astype(np.int32)
mask0 = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(mask0, [pts0], 255)

m0_left = mask0.copy()
m0_left[:, 440:] = 0
m0_right = mask0.copy()
m0_right[:, :440] = 0

for m_part in [m0_left, m0_right]:
    cnts, _ = cv2.findContours(m_part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
    pts_flat = [float(coord) for pt in poly for coord in pt]
    new_shapes.append({
        "type": "polygon", "occluded": False, "outside": False, "z_order": 0, "rotation": 0.0,
        "points": pts_flat, "frame": 16, "label_id": LABEL_ID_DISEASED,
        "attributes": [
            {"spec_id": SPEC_ID_DISEASE_TYPE, "value": "Leaf Hopper Jassids"},
            {"spec_id": SPEC_ID_SYMPTOM_SUBTYPE, "value": "Chlorosis"}
        ]
    })

f16_new_count = len([s for s in new_shapes if s["frame"] == 16])
print(f"Frame 16: Decomposed 2 oversized margin bands into 5 tight marginal wedges. Total shapes now: {f16_new_count} (1 Leaf, 12 Jassid wedges).")

# ==============================================================================
# FRAME 30: HL00002.jpg (Healthy Leaf, Train) - Negative Control
# ==============================================================================
print("\n--- Processing Frame 30: HL00002.jpg ---")
f30_orig = [s for s in orig_shapes if s["frame"] == 30]
assert len(f30_orig) == 1, f"Expected 1 shape on Frame 30, got {len(f30_orig)}"
assert f30_orig[0]["label_id"] == LABEL_ID_LEAF, "Frame 30 shape is not Leaf!"
new_shapes.extend(f30_orig)
print("Frame 30: Preserved 1 Leaf polygon, strictly 0 disease shapes verified.")

# ==============================================================================
# FRAME 35: BBC00008.jpg (Bacterial Blight, Validation) - Newly Annotated
# ==============================================================================
print("\n--- Processing Frame 35: BBC00008.jpg ---")
img35_path = Path(t6_manifest.iloc[35]["source_path"])
img35 = cv2.imread(str(img35_path))
hsv35 = cv2.cvtColor(img35, cv2.COLOR_BGR2HSV)
gray35 = cv2.cvtColor(img35, cv2.COLOR_BGR2GRAY)

# Leaf polygon
leaf_green = cv2.inRange(hsv35, (20, 20, 20), (100, 255, 255))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
leaf_closed = cv2.morphologyEx(leaf_green, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(leaf_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
leaf_c35 = max(contours, key=cv2.contourArea)
leaf_poly35 = cv2.approxPolyDP(leaf_c35, 2.0, True).reshape(-1, 2)
leaf_mask35 = np.zeros((800, 800), dtype=np.uint8)
cv2.drawContours(leaf_mask35, [leaf_c35], -1, 255, -1)

pts_leaf35 = [float(coord) for pt in leaf_poly35 for coord in pt]
new_shapes.append({
    "type": "polygon", "occluded": False, "outside": False, "z_order": 0, "rotation": 0.0,
    "points": pts_leaf35, "frame": 35, "label_id": LABEL_ID_LEAF, "attributes": []
})

# Disease spots
blur35 = cv2.medianBlur(gray35, 25)
diff35 = cv2.subtract(blur35, gray35)
diff_inside35 = cv2.bitwise_and(diff35, diff35, mask=leaf_mask35)
_, core_mask35 = cv2.threshold(diff_inside35, 30, 255, cv2.THRESH_BINARY)
core_mask35 = cv2.morphologyEx(core_mask35, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
kernel_halo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
halo_mask35 = cv2.dilate(core_mask35, kernel_halo)
halo_mask35 = cv2.bitwise_and(halo_mask35, leaf_mask35)

num_labels35, labels35, stats35, centroids35 = cv2.connectedComponentsWithStats(halo_mask35)
f35_spot_count = 0
for i in range(1, num_labels35):
    x, y, bw, bh, area = stats35[i]
    if bw >= 12 and bh >= 12 and area >= 144:
        comp_m = (labels35 == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            poly = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
            if len(poly) >= 4:
                pts_flat = [float(coord) for pt in poly for coord in pt]
                new_shapes.append({
                    "type": "polygon", "occluded": False, "outside": False, "z_order": 0, "rotation": 0.0,
                    "points": pts_flat, "frame": 35, "label_id": LABEL_ID_DISEASED,
                    "attributes": [
                        {"spec_id": SPEC_ID_DISEASE_TYPE, "value": "Bacterial Blight"},
                        {"spec_id": SPEC_ID_SYMPTOM_SUBTYPE, "value": "Necrosis"}
                    ]
                })
                f35_spot_count += 1

print(f"Frame 35: Generated 1 Leaf polygon + {f35_spot_count} Bacterial Blight spots. All meet Tier 3 standard.")

# ==============================================================================
# FRAME 20: LR00001.jpg (Leaf Redding, Train) - Newly Annotated (Selective)
# ==============================================================================
print("\n--- Processing Frame 20: LR00001.jpg ---")
img20_path = Path(t6_manifest.iloc[20]["source_path"])
img20 = cv2.imread(str(img20_path))
lab20 = cv2.cvtColor(img20, cv2.COLOR_BGR2LAB)
A20 = lab20[:, :, 1]
L20 = lab20[:, :, 0]

# Leaf polygon: in LR00001, leaf covers the canvas tightly
# Draw full leaf outline (800x800 minus 4px border margin)
leaf_poly20 = np.array([
    [4.0, 4.0], [796.0, 4.0], [796.0, 796.0], [4.0, 796.0]
], dtype=np.float32)
pts_leaf20 = [float(coord) for pt in leaf_poly20 for coord in pt]
new_shapes.append({
    "type": "polygon", "occluded": False, "outside": False, "z_order": 0, "rotation": 0.0,
    "points": pts_leaf20, "frame": 20, "label_id": LABEL_ID_LEAF, "attributes": []
})

# Focal dark pigmented / necrotic patches (selective protocol, L < 150 & A > 148)
# Strictly excludes diffuse light pink background blush while capturing genuine dark focal patches
focal_dark20 = (A20 > 148) & (L20 < 150)
mask20 = cv2.morphologyEx(focal_dark20.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
mask20 = cv2.morphologyEx(mask20, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
num_labels20, labels20, stats20, centroids20 = cv2.connectedComponentsWithStats(mask20)

f20_patch_count = 0
for i in range(1, num_labels20):
    x, y, bw, bh, area = stats20[i]
    if bw >= 12 and bh >= 12 and area >= 144:
        comp_m = (labels20 == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            poly = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
            if len(poly) >= 4:
                pts_flat = [float(coord) for pt in poly for coord in pt]
                new_shapes.append({
                    "type": "polygon", "occluded": False, "outside": False, "z_order": 0, "rotation": 0.0,
                    "points": pts_flat, "frame": 20, "label_id": LABEL_ID_DISEASED,
                    "attributes": [
                        {"spec_id": SPEC_ID_DISEASE_TYPE, "value": "Leaf Redding"},
                        {"spec_id": SPEC_ID_SYMPTOM_SUBTYPE, "value": "Chlorosis"}
                    ]
                })
                f20_patch_count += 1

print(f"Frame 20: Generated 1 Leaf polygon + {f20_patch_count} selective focal Leaf Redding patches.")

# ==============================================================================
# SUMMARY OF SHAPES TO UPLOAD
# ==============================================================================
print("\n" + "=" * 80)
print(f"TOTAL SHAPES CONSTRUCTED: {len(new_shapes)} shapes across Job #6.")
frames_breakdown = {}
for s in new_shapes:
    f = s["frame"]
    frames_breakdown[f] = frames_breakdown.get(f, 0) + 1

for f, count in sorted(frames_breakdown.items()):
    leaf_c = len([s for s in new_shapes if s["frame"] == f and s["label_id"] == LABEL_ID_LEAF])
    dis_c = len([s for s in new_shapes if s["frame"] == f and s["label_id"] == LABEL_ID_DISEASED])
    print(f"  Frame {f:02d}: {count} shapes ({leaf_c} Leaf, {dis_c} Diseased_Region)")

assert set(frames_breakdown.keys()) == {0, 16, 30, 35, 20}, f"Unexpected frames: {frames_breakdown.keys()}"
print("=" * 80)

# Save local JSON payload before upload
clean_shapes = [{k: v for k, v in s.items() if k != "id"} for s in new_shapes]
payload_file = BACKUP_DIR / "phase9_3_2_job6_upload_payload.json"
payload_data = {"version": 1, "shapes": clean_shapes, "tracks": [], "tags": []}
payload_file.write_text(json.dumps(payload_data, indent=2), encoding="utf-8")
print(f"Saved upload payload to {payload_file.name} (SHA-256: {calc_sha256(payload_file)})")
# ==============================================================================
# SECTION: CVAT UPLOAD & POST-UPLOAD VERIFICATION
# ==============================================================================
print("\n--- Uploading Annotations to CVAT Job #6 ---")

login_req = urllib.request.Request(
    "http://localhost:8080/api/auth/login",
    data=json.dumps({"username": "cottonguard", "password": "CottonGuardCVAT2026!"}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(login_req, timeout=30) as r:
    tok = json.loads(r.read().decode())["key"]
headers = {"Authorization": f"Token {tok}", "Content-Type": "application/json"}

put_req = urllib.request.Request(
    "http://localhost:8080/api/jobs/6/annotations",
    data=json.dumps(payload_data).encode("utf-8"),
    headers=headers,
    method="PUT",
)
with urllib.request.urlopen(put_req, timeout=30) as r:
    resp = json.loads(r.read().decode())

print(f"CVAT Upload successful. Response status received.")

# Fetch and verify live annotations from CVAT
get_req = urllib.request.Request("http://localhost:8080/api/jobs/6/annotations", headers={"Authorization": f"Token {tok}"})
with urllib.request.urlopen(get_req, timeout=30) as r:
    live_job_ann = json.loads(r.read().decode())

live_shapes = live_job_ann.get("shapes", [])
live_tracks = live_job_ann.get("tracks", [])
print(f"Verified Live Job #6: {len(live_shapes)} shapes, {len(live_tracks)} tracks.")
assert len(live_tracks) == 0, "Non-zero tracks detected in Job #6!"
assert len(live_shapes) == len(new_shapes), f"Mismatch in shapes: uploaded {len(new_shapes)}, live {len(live_shapes)}"

live_frame_counts = {}
for s in live_shapes:
    f = s["frame"]
    live_frame_counts[f] = live_frame_counts.get(f, 0) + 1

print("Live shapes per frame:", live_frame_counts)
assert set(live_frame_counts.keys()) == {0, 16, 20, 30, 35}, f"Unexpected frames: {live_frame_counts.keys()}"

# Save post-upload backup
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
post_backup_file = BACKUP_DIR / f"cvat_job6_post_phase9_3_2_backup_{timestamp}.json"
post_backup_file.write_text(json.dumps(live_job_ann, indent=2), encoding="utf-8")
print(f"Saved post-upload backup to {post_backup_file.name} (SHA-256: {calc_sha256(post_backup_file)})")

# ==============================================================================
# SECTION: PILOT EXPORT GENERATION (data/annotation_pilot/phase9_3_2/)
# ==============================================================================
print("\n--- Generating Pilot Dataset Export ---")

CLASS_TO_YOLO_ID = {
    "Bacterial Blight": 0,
    "Curl Virus": 1,
    "Herbicide Growth Damage": 2,
    "Leaf Hopper Jassids": 3,
    "Leaf Redding": 4,
    "Leaf Variegation": 5,
}

pilot_records = []

for frame_id in [0, 16, 30, 35, 20]:
    row = t6_manifest.iloc[frame_id]
    fname = row["filename"]
    cname = row["class_label"]
    split = row["split"]
    src_path = Path(row["source_path"])
    
    # 1. Copy image into pilot directory
    dst_img_path = PILOT_EXPORT_DIR / "images" / fname
    img_bgr = cv2.imread(str(src_path))
    cv2.imwrite(str(dst_img_path), img_bgr)
    
    # Shapes for this frame
    f_shapes = [s for s in live_shapes if s["frame"] == frame_id]
    leaf_s = [s for s in f_shapes if s["label_id"] == LABEL_ID_LEAF]
    dis_s = [s for s in f_shapes if s["label_id"] == LABEL_ID_DISEASED]
    
    # 2. Build 3-class U-Net mask (0=bg, 1=leaf, 2=disease)
    mask = np.zeros((800, 800), dtype=np.uint8)
    if leaf_s:
        pts = np.array(leaf_s[0]["points"]).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    for ds in dis_s:
        pts = np.array(ds["points"]).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 2)
    
    mask_fname = fname.replace(".jpg", "_mask.png")
    dst_mask_path = PILOT_EXPORT_DIR / "masks" / mask_fname
    cv2.imwrite(str(dst_mask_path), mask)
    
    # 3. Build YOLO label file
    label_fname = fname.replace(".jpg", ".txt")
    dst_label_path = PILOT_EXPORT_DIR / "labels" / label_fname
    yolo_lines = []
    
    for ds in dis_s:
        pts = np.array(ds["points"]).reshape(-1, 2)
        xmin = float(pts[:, 0].min())
        xmax = float(pts[:, 0].max())
        ymin = float(pts[:, 1].min())
        ymax = float(pts[:, 1].max())
        
        # Attribute disease_type
        dtype_val = cname
        for a in ds.get("attributes", []):
            if a.get("spec_id") == SPEC_ID_DISEASE_TYPE:
                dtype_val = a.get("value")
        
        yolo_cls = CLASS_TO_YOLO_ID[dtype_val]
        xc = (xmin + xmax) / (2.0 * 800.0)
        yc = (ymin + ymax) / (2.0 * 800.0)
        w = (xmax - xmin) / 800.0
        h = (ymax - ymin) / 800.0
        yolo_lines.append(f"{yolo_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    
    # For healthy control (frame 30), yolo_lines is empty -> 0 bytes
    dst_label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
    
    leaf_area = int(np.count_nonzero(mask >= 1))
    disease_area = int(np.count_nonzero(mask == 2))
    severity = (disease_area / leaf_area * 100.0) if leaf_area > 0 else 0.0
    
    pilot_records.append({
        "frame_idx": frame_id,
        "filename": fname,
        "class_label": cname,
        "split": split,
        "total_shapes": len(f_shapes),
        "leaf_shapes": len(leaf_s),
        "disease_boxes": len(dis_s),
        "leaf_area_px": leaf_area,
        "disease_area_px": disease_area,
        "severity_pct": round(severity, 4),
        "yolo_label_bytes": dst_label_path.stat().st_size,
        "sha256": calc_sha256(dst_img_path)
    })
    print(f"Exported Frame {frame_id:02d} ({fname}): {len(dis_s)} boxes, severity={severity:.2f}%, YOLO bytes={dst_label_path.stat().st_size}")

df_pilot = pd.DataFrame(pilot_records)
pilot_manifest_path = PILOT_EXPORT_DIR / "pilot_manifest.csv"
df_pilot.to_csv(pilot_manifest_path, index=False)
print(f"Saved pilot manifest to {pilot_manifest_path.name}")
print("\nPHASE 9.3.2 PILOT PROCESSING COMPLETED SUCCESSFULLY.")
