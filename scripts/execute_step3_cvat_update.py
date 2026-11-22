"""
scripts/execute_step3_cvat_update.py
Executes Phase 8.8 Step 3 CVAT update under strict authorization:
- Target: CottonGuard_Pilot_Batch_03_ShapesOnly (Task #6, Job #6)
- Frame 0 (BBC00001.jpg): 1 Leaf + 19 Diseased_Region polygons (Option B, P02/06/08/11/19 excluded)
- Frame 30 (HL00002.jpg): 1 Leaf (concave sinus corrected) + 0 Diseased_Region polygons
- Safeguards:
  1. Verifies original Phase 8.7 backup SHA-256 (never overwrites).
  2. Creates timestamped pre-update backup.
  3. Updates only frames 0 and 30 in Job #6.
  4. Preserves other 48 frames untouched.
  5. Exports updated CVAT XML to new backup zip.
  6. Re-imports and inspects exported XML.
  7. Verifies 5 excluded polygons are absent.
  8. Verifies unique rasterized disease pixels (6,477 px, 1.0702% severity).
  9. Compares before/after state for all 48 frames.
  10. Verifies all model checkpoints, database, and sealed test split remain untouched.
"""

import json
import time
import zipfile
import hashlib
import cv2
import numpy as np
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path("D:/Main_Project/cotton_leaf_ai")
backup_dir = PROJECT_ROOT / "artifacts" / "backups"
backup_dir.mkdir(parents=True, exist_ok=True)
corrections_dir = PROJECT_ROOT / "artifacts" / "pilot_corrections" / "phase8_8"
metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)

# Helper: calculate SHA-256
def calc_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().upper()

# Helper: unpack flat points
def unpack_points(flat_pts):
    pts = []
    for i in range(0, len(flat_pts), 2):
        pts.append([int(round(flat_pts[i])), int(round(flat_pts[i+1]))])
    return np.array(pts, dtype=np.int32)

print("="*80)
print("PHASE 8.8 STEP 3: CVAT TASK #6 UPDATE & VALIDATION")
print("="*80)

# --- 1. Verify Original Phase 8.7 Backup ---
orig_backup = backup_dir / "cvat_task6_two_image_pilot_backup.zip"
assert orig_backup.exists(), "Original Phase 8.7 backup zip missing!"
orig_hash = calc_sha256(orig_backup)
print(f"[SAFEGUARD 1] Original Phase 8.7 Backup SHA-256: {orig_hash}")
expected_orig_hash = "F4AB37F377CC0D753C36AB9190B0EE3F635CEB6E39C546269590B5791ED3E184"
assert orig_hash == expected_orig_hash, f"Hash mismatch! Expected {expected_orig_hash}, got {orig_hash}"
print("  -> Original backup verified and protected.")

# CVAT Auth
login_req = urllib.request.Request(
    "http://localhost:8080/api/auth/login",
    data=json.dumps({"username": "cottonguard", "password": "CottonGuardCVAT2026!"}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(login_req, timeout=30) as r:
    tok = json.loads(r.read().decode())["key"]
headers = {"Authorization": f"Token {tok}"}

def export_task6_xml(output_zip_path):
    export_url = "http://localhost:8080/api/tasks/6/dataset/export?" + urllib.parse.urlencode({
        "format": "CVAT for images 1.1",
        "save_images": "false"
    })
    exp_req = urllib.request.Request(export_url, headers=headers, method="POST")
    with urllib.request.urlopen(exp_req, timeout=30) as r:
        exp_data = json.loads(r.read().decode())
    rq_id = exp_data.get("rq_id")
    print(f"  Export enqueued, rq_id={rq_id}")
    
    result_url = None
    for step in range(40):
        time.sleep(1.0)
        stat_req = urllib.request.Request(f"http://localhost:8080/api/requests/{rq_id}", headers=headers)
        with urllib.request.urlopen(stat_req, timeout=30) as r:
            stat_data = json.loads(r.read().decode())
        st = stat_data.get("status")
        if st == "finished":
            result_url = stat_data.get("result_url")
            break
        if st == "failed":
            raise RuntimeError(f"Export failed: {stat_data.get('exc_info')}")
    if not result_url:
        raise TimeoutError("Export timed out waiting for result_url.")
        
    dl_req = urllib.request.Request(result_url, headers=headers)
    with urllib.request.urlopen(dl_req, timeout=60) as r:
        content = r.read()
    assert content[:2] == b"PK", "Downloaded content is not a valid zip archive!"
    output_zip_path.write_bytes(content)
    print(f"  Downloaded backup: {output_zip_path} ({len(content)} bytes)")
    return calc_sha256(output_zip_path)

# --- 2. Create Timestamped Pre-Update Backup ---
pre_update_zip = backup_dir / "cvat_task6_pre_step3_backup_20261114.zip"
print(f"\n[SAFEGUARD 2] Creating Pre-Update Backup -> {pre_update_zip.name}...")
pre_hash = export_task6_xml(pre_update_zip)
print(f"  Pre-Update Backup SHA-256: {pre_hash}")

# Inspect pre-update annotations
with zipfile.ZipFile(pre_update_zip, "r") as zf:
    pre_xml = zf.read("annotations.xml")
root_pre = ET.fromstring(pre_xml)
pre_images = {int(img.get("id")): img for img in root_pre.findall(".//image")}
pre_counts = {img_id: len(img.findall("polygon")) for img_id, img in pre_images.items()}
print(f"  Pre-update shapes: Frame 0 = {pre_counts.get(0, 0)}, Frame 30 = {pre_counts.get(30, 0)}, Other 48 frames sum = {sum(v for k, v in pre_counts.items() if k not in [0, 30])}")

# --- 3. Prepare Step 3 Payload (Option B for Frame 0, Corrected Leaf for Frame 30) ---
print("\n[SAFEGUARD 3] Preparing Step 3 Upload Payload...")
with open(corrections_dir / "bbc00001_option_b_19polygons.json") as f:
    bbc_cand = json.load(f)
with open(corrections_dir / "hl00002_corrected_polygon.json") as f:
    hl_cand = json.load(f)

# Leaf shape for Frame 0
bbc_leaf_shape = {
    "type": "polygon",
    "occluded": False,
    "outside": False,
    "z_order": 0,
    "rotation": 0.0,
    "points": bbc_cand["leaf_shape"]["points"],
    "frame": 0,
    "label_id": 5, # Leaf
    "attributes": []
}

# 19 Disease shapes for Frame 0
bbc_disease_shapes = []
for s in bbc_cand["disease_shapes"]:
    shape = {
        "type": "polygon",
        "occluded": False,
        "outside": False,
        "z_order": 0,
        "rotation": 0.0,
        "points": s["points"],
        "frame": 0,
        "label_id": 6, # Diseased_Region
        "attributes": [
            {"spec_id": 2, "value": "Bacterial Blight"},
            {"spec_id": 3, "value": "Necrosis"}
        ]
    }
    bbc_disease_shapes.append(shape)

# Leaf shape for Frame 30
hl_leaf_shape = {
    "type": "polygon",
    "occluded": False,
    "outside": False,
    "z_order": 0,
    "rotation": 0.0,
    "points": hl_cand["leaf_shape"]["points"],
    "frame": 30,
    "label_id": 5, # Leaf
    "attributes": []
}

step3_shapes = [bbc_leaf_shape] + bbc_disease_shapes + [hl_leaf_shape]
print(f"  Total shapes in upload payload: {len(step3_shapes)} (1 Leaf + 19 Diseased on Frame 0, 1 Leaf on Frame 30)")

# Save payload for reproducibility
payload_record = {
    "task_id": 6,
    "job_id": 6,
    "execution_phase": "Phase 8.8 Step 3",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "shapes": step3_shapes,
    "tracks": [],
    "tags": []
}
payload_file = corrections_dir / "step3_cvat_upload_payload.json"
with open(payload_file, "w") as f:
    json.dump(payload_record, f, indent=2)

# --- 4. Push Updated Annotations to Job #6 ---
print("\n[SAFEGUARD 4] Pushing Updated Annotations to CVAT Job #6...")
put_payload = {
    "shapes": step3_shapes,
    "tracks": [],
    "tags": []
}
put_req = urllib.request.Request(
    "http://localhost:8080/api/jobs/6/annotations",
    data=json.dumps(put_payload).encode("utf-8"),
    headers={"Content-Type": "application/json", "Authorization": f"Token {tok}"},
    method="PUT"
)
with urllib.request.urlopen(put_req, timeout=30) as r:
    put_resp = json.loads(r.read().decode())

saved_shapes = put_resp.get("shapes", [])
saved_tracks = put_resp.get("tracks", [])
print(f"  PUT Response: version={put_resp.get('version')}, saved_shapes={len(saved_shapes)}, saved_tracks={len(saved_tracks)}")
assert len(saved_shapes) == 21, f"Expected 21 shapes saved in Job #6, got {len(saved_shapes)}"
assert len(saved_tracks) == 0, f"Expected 0 tracks saved in Job #6, got {len(saved_tracks)}"

# --- 5. Export Updated CVAT XML Backup ---
updated_backup_zip = backup_dir / "cvat_task6_step3_updated_backup_20261114.zip"
print(f"\n[SAFEGUARD 5] Exporting Updated CVAT XML Backup -> {updated_backup_zip.name}...")
updated_hash = export_task6_xml(updated_backup_zip)
print(f"  Updated CVAT Backup SHA-256: {updated_hash}")

# --- 6. Re-import and Parse Exported XML for Strict Verification ---
print("\n[SAFEGUARD 6] Parsing Exported annotations.xml for Strict Verification...")
with zipfile.ZipFile(updated_backup_zip, "r") as zf:
    assert "annotations.xml" in zf.namelist(), "annotations.xml missing from updated export!"
    xml_data = zf.read("annotations.xml")

root_post = ET.fromstring(xml_data)

# Verify zero tracks
post_tracks = root_post.findall(".//track")
assert len(post_tracks) == 0, f"Found {len(post_tracks)} unexpected tracks in updated XML!"
print(f"  Tracks count: {len(post_tracks)} (PASSED)")

# Verify 50 images
post_images = root_post.findall(".//image")
assert len(post_images) == 50, f"Expected 50 images in XML, got {len(post_images)}"
post_images_by_id = {int(img.get("id")): img for img in post_images}
print(f"  Images count: {len(post_images)} (PASSED)")

# Verify Frame 0 (BBC00001.jpg)
img0 = post_images_by_id[0]
assert img0.get("name") == "BBC00001.jpg", f"Frame 0 name mismatch: {img0.get('name')}"
polys_0 = img0.findall("polygon")
leaf_0 = [p for p in polys_0 if p.get("label") == "Leaf"]
disease_0 = [p for p in polys_0 if p.get("label") == "Diseased_Region"]
print(f"  Frame 0: {len(leaf_0)} Leaf polygon, {len(disease_0)} Diseased_Region polygons")
assert len(leaf_0) == 1, f"Expected 1 Leaf polygon on Frame 0, got {len(leaf_0)}"
assert len(disease_0) == 19, f"Expected 19 Diseased_Region polygons on Frame 0, got {len(disease_0)}"

# Verify Frame 0 attributes
for d_poly in disease_0:
    attrs = {a.get("name"): a.text for a in d_poly.findall("attribute")}
    assert attrs.get("disease_type") == "Bacterial Blight", f"Attribute disease_type mismatch: {attrs.get('disease_type')}"
    assert attrs.get("symptom_subtype") == "Necrosis", f"Attribute symptom_subtype mismatch: {attrs.get('symptom_subtype')}"
print("  Frame 0 attributes verified: all 19 have disease_type='Bacterial Blight' and symptom_subtype='Necrosis' (PASSED)")

# --- 7. Verify the 5 Excluded Polygons are Absent ---
print("\n[SAFEGUARD 7] Verifying 5 Excluded Polygons are Absent from CVAT Upload...")
# Excluded centroids: P02=(420.8, 36.9), P06=(200.7, 183.1), P08=(224.5, 280.0), P11=(644.3, 328.8), P19=(258.3, 637.2)
excluded_centroids = [
    (420.8, 36.9),
    (200.7, 183.1),
    (224.5, 280.0),
    (644.3, 328.8),
    (258.3, 637.2)
]

uploaded_centroids = []
for d_poly in disease_0:
    pts_str = d_poly.get("points")
    pts_list = []
    for pair in pts_str.split(";"):
        x, y = pair.split(",")
        pts_list.append((float(x), float(y)))
    cx = sum(p[0] for p in pts_list) / len(pts_list)
    cy = sum(p[1] for p in pts_list) / len(pts_list)
    uploaded_centroids.append((cx, cy))

for ex_idx, (ex_x, ex_y) in enumerate(excluded_centroids, 1):
    min_dist = min(((ex_x - ux)**2 + (ex_y - uy)**2)**0.5 for ux, uy in uploaded_centroids)
    print(f"  Excluded candidate #{ex_idx} at ({ex_x:.1f}, {ex_y:.1f}) -> Closest uploaded centroid dist = {min_dist:.1f} px")
    assert min_dist > 15.0, f"Excluded candidate ({ex_x}, {ex_y}) unexpectedly found in upload (dist={min_dist})!"
print("  -> All 5 excluded polygons are confirmed completely ABSENT from CVAT upload (PASSED)")

# --- 8. Verify Disease Masks, Unique Rasterized Pixel Counts & Severity ---
print("\n[SAFEGUARD 8] Verifying Unique Rasterized Disease Pixels and Severity...")
# Rasterize Frame 0 Leaf and Disease from XML
h, w = int(img0.get("height", 800)), int(img0.get("width", 800))
leaf_pts_xml = []
for pair in leaf_0[0].get("points").split(";"):
    x, y = pair.split(",")
    leaf_pts_xml.append([int(round(float(x))), int(round(float(y)))])
leaf_mask_0 = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(leaf_mask_0, [np.array(leaf_pts_xml, dtype=np.int32)], 1)
total_leaf_0_px = int(np.count_nonzero(leaf_mask_0))

dis_mask_0 = np.zeros((h, w), dtype=np.uint8)
for d_poly in disease_0:
    d_pts = []
    for pair in d_poly.get("points").split(";"):
        x, y = pair.split(",")
        d_pts.append([int(round(float(x))), int(round(float(y)))])
    cv2.fillPoly(dis_mask_0, [np.array(d_pts, dtype=np.int32)], 1)
unique_dis_0_px = int(np.count_nonzero(dis_mask_0))
sev_0 = (unique_dis_0_px / total_leaf_0_px) * 100

print(f"  Frame 0 Leaf Lamina Pixels: {total_leaf_0_px} (expected 605221)")
print(f"  Frame 0 Unique Disease Pixels: {unique_dis_0_px} (expected 6477)")
print(f"  Frame 0 Standardized Severity: {sev_0:.4f}% (expected 1.0702%)")
assert total_leaf_0_px == 605221, f"Leaf pixel mismatch: {total_leaf_0_px}"
assert unique_dis_0_px == 6477, f"Disease pixel mismatch: {unique_dis_0_px}"
assert abs(sev_0 - 1.0702) < 0.0001, f"Severity mismatch: {sev_0}"

# Rasterize Frame 30 (HL00002.jpg)
img30 = post_images_by_id[30]
assert img30.get("name") == "HL00002.jpg", f"Frame 30 name mismatch: {img30.get('name')}"
polys_30 = img30.findall("polygon")
leaf_30 = [p for p in polys_30 if p.get("label") == "Leaf"]
disease_30 = [p for p in polys_30 if p.get("label") == "Diseased_Region"]
print(f"  Frame 30: {len(leaf_30)} Leaf polygon, {len(disease_30)} Diseased_Region polygons")
assert len(leaf_30) == 1, f"Expected 1 Leaf polygon on Frame 30, got {len(leaf_30)}"
assert len(disease_30) == 0, f"Expected 0 Diseased_Region polygons on Frame 30, got {len(disease_30)}"

leaf_pts_30 = []
for pair in leaf_30[0].get("points").split(";"):
    x, y = pair.split(",")
    leaf_pts_30.append([int(round(float(x))), int(round(float(y)))])
leaf_mask_30 = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(leaf_mask_30, [np.array(leaf_pts_30, dtype=np.int32)], 1)
total_leaf_30_px = int(np.count_nonzero(leaf_mask_30))
print(f"  Frame 30 Leaf Lamina Pixels: {total_leaf_30_px} (expected 515030)")
print(f"  Frame 30 Disease Pixels: 0 px, Severity: 0.0000%")
assert total_leaf_30_px == 515030, f"HL00002 leaf pixel mismatch: {total_leaf_30_px}"

# Verify sinus apex point (470, 238)
has_apex = any(pt[0] == 470 and pt[1] == 238 for pt in leaf_pts_30)
print(f"  Frame 30 Sinus Apex (470, 238) present: {has_apex} (PASSED)")
assert has_apex, "Sinus apex (470, 238) missing from Frame 30 Leaf polygon!"

# --- 9. Compare Before/After State for Other 48 Frames ---
print("\n[SAFEGUARD 9] Comparing Before/After State for all Other 48 Frames...")
post_counts = {img_id: len(img.findall("polygon")) for img_id, img in post_images_by_id.items()}
other_48_unchanged = True
for frame_id in range(50):
    if frame_id not in [0, 30]:
        pre_c = pre_counts.get(frame_id, 0)
        post_c = post_counts.get(frame_id, 0)
        if pre_c != 0 or post_c != 0:
            print(f"  Frame {frame_id} violation: pre={pre_c}, post={post_c}")
            other_48_unchanged = False
assert other_48_unchanged, "Non-target frame contamination detected!"
print("  -> All other 48 frames confirmed strictly 0 shapes before and 0 shapes after (PASSED)")

# --- 10. Verify Production Assets & Sealed Test Split Preservation ---
print("\n[SAFEGUARD 10] Verifying Checkpoints, Database, and Sealed Test Split...")
checkpoints_to_check = {
    "classification_best_model": (PROJECT_ROOT / "artifacts" / "checkpoints" / "classification" / "best_model.pt", "552BABA83A66AD4A8BBC1EEB7FE3EE8DD84DFABE68C1EF477930DA57A05FA90E"),
    "segmentation_best_unet": (PROJECT_ROOT / "artifacts" / "checkpoints" / "segmentation" / "best_unet.pt", "612FCA0A162B56B8253A81B9DED3D069C1E15666E505FEB38FC93ED6BE7C3A43"),
    "segmentation_exp2_768": (PROJECT_ROOT / "artifacts" / "checkpoints" / "segmentation" / "exp2_768_best_unet.pt", "C290A81D4CF0940DA641EF3684086812244F01E4A0372AE87853AB82B9844B34"),
    "segmentation_exp4b_w20": (PROJECT_ROOT / "artifacts" / "checkpoints" / "segmentation" / "exp4b_w20_best_unet.pt", "C6788F794869E362BCEBB8BF2A075A83EF0EACEBF6B9EF3D75FDFE57517F3558"),
    "database": (PROJECT_ROOT / "cotton_leaf_ai.db", "90AEF2B9865E4C871732E7EF33E63CF87D63DBE9F0463AAF9C0792D93782B5D2")
}

for name, (path, expected_h) in checkpoints_to_check.items():
    actual_h = calc_sha256(path)
    print(f"  {name}: {actual_h[:16]}... (matches expected)")
    assert actual_h == expected_h, f"Checkpoint hash mismatch for {name}!"

# Check test set lines
test_csv = PROJECT_ROOT / "artifacts" / "manifests" / "splits" / "test.csv"
with open(test_csv) as f:
    test_lines = f.readlines()
test_count = len(test_lines) - 1 # minus header
print(f"  Sealed test split: {test_count} images (expected 321)")
assert test_count == 321, f"Sealed test split altered! Expected 321, got {test_count}"

# Check original candidate JSON untouched
orig_bbc_json = corrections_dir / "bbc00001_corrected_polygons.json"
with open(orig_bbc_json) as f:
    orig_bbc_data = json.load(f)
assert orig_bbc_data["disease_polygons_count"] == 24, "Original candidate JSON was altered!"
print("  Original 24-polygon candidate JSON preserved untouched (PASSED)")

# Write execution validation report JSON
step3_validation_report = {
    "execution_phase": "Phase 8.8 Step 3",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "task_id": 6,
    "job_id": 6,
    "status": "COMPLETED_AND_VERIFIED",
    "cvat_shapes_before": {
        "frame_0": pre_counts.get(0, 0),
        "frame_30": pre_counts.get(30, 0),
        "other_48_frames": 0,
        "total": sum(pre_counts.values())
    },
    "cvat_shapes_after": {
        "frame_0": {
            "leaf_polygons": 1,
            "disease_polygons": 19,
            "total": 20,
            "unique_disease_px": 6477,
            "total_leaf_px": 605221,
            "severity_pct": 1.0702,
            "excluded_polygons_absent": [2, 6, 8, 11, 19]
        },
        "frame_30": {
            "leaf_polygons": 1,
            "disease_polygons": 0,
            "total": 1,
            "unique_disease_px": 0,
            "total_leaf_px": 515030,
            "severity_pct": 0.0,
            "sinus_apex": [470, 238]
        },
        "other_48_frames": {
            "shapes_count": 0,
            "status": "UNTOUCHED_AND_UNANNOTATED"
        },
        "total_shapes_in_task": 21
    },
    "backups": {
        "phase8_7_original_backup": {
            "path": str(orig_backup),
            "sha256": orig_hash,
            "status": "VERIFIED_UNTOUCHED"
        },
        "pre_update_backup": {
            "path": str(pre_update_zip),
            "sha256": pre_hash
        },
        "step3_updated_backup": {
            "path": str(updated_backup_zip),
            "sha256": updated_hash
        }
    },
    "preservation_invariants": {
        "all_checkpoints_verified": True,
        "database_verified": True,
        "sealed_test_set_verified": True,
        "zero_training_initiated": True,
        "other_48_images_unannotated": True
    }
}

val_file = metrics_dir / "phase8_8_step3_execution_validation.json"
with open(val_file, "w") as f:
    json.dump(step3_validation_report, f, indent=2)
print(f"\nSaved execution validation summary to: {val_file}")
print("="*80)
print("PHASE 8.8 STEP 3 CVAT UPDATE SUCCESSFULLY COMPLETED AND VERIFIED")
print("="*80)
