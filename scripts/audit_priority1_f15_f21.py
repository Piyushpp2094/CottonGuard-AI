"""
scripts/audit_priority1_f15_f21.py
Forensic inspection of Frames 15 (LHJ00001) and 21 (LR00140) for Stage 2.1.
Analyzes pixel properties, foliar components, and host-leaf associations.
"""

import json
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path("D:/Main_Project/cotton_leaf_ai")
IMG_DIR = PROJECT_ROOT / "data" / "detection_consolidated_100" / "images"
BASELINE_JSON = PROJECT_ROOT / "artifacts" / "backups" / "cvat_job6_fresh_pre_remediation_baseline_20261118.json"
CANDIDATES_JSON = PROJECT_ROOT / "artifacts" / "annotation_staging" / "phase12_2" / "candidates_153.json"

with open(BASELINE_JSON) as f:
    cvat_data = json.load(f)
with open(CANDIDATES_JSON) as f:
    cands = json.load(f)

# --- FRAME 15 ---
print("=" * 80)
print("FORENSIC AUDIT: FRAME 15 (LHJ00001.jpg)")
print("=" * 80)
img15 = cv2.imread(str(IMG_DIR / "LHJ00001.jpg"))
f15_lesions = [c for c in cands if c["frame_number"] == 15 and c["annotation_label"] == "Diseased_Region"]
f15_leaf = next(c for c in cands if c["frame_number"] == 15 and c["annotation_label"] == "Leaf")

print(f"Frame 15 total lesions: {len(f15_lesions)}")
print(f"Primary leaf candidate vertices: {f15_leaf['vertex_count']}, area: {f15_leaf['area']}")

# Check the 7 zero-containment lesions: 1173 to 1179
zero_cands = [c for c in f15_lesions if c["replaced_shape_id"] in range(1173, 1180)]
for c in zero_cands:
    pts = np.array(c["polygon_coordinates"]).reshape(-1, 2)
    bx1, by1 = np.min(pts, axis=0)
    bx2, by2 = np.max(pts, axis=0)
    roi = img15[int(by1):int(by2), int(bx1):int(bx2)]
    mean_bgr = np.mean(roi, axis=(0, 1))
    # In HSV: Hue for green foliage is ~35-85
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean_hsv = np.mean(hsv_roi, axis=(0, 1))
    print(f"Candidate {c['candidate_id']}: bbox=[{bx1:.1f}, {by1:.1f}, {bx2:.1f}, {by2:.1f}], Area={c['area']:.1f}, Mean BGR={mean_bgr.round(1)}, Mean HSV={mean_hsv.round(1)}")

# --- FRAME 21 ---
print("\n" + "=" * 80)
print("FORENSIC AUDIT: FRAME 21 (LR00140.jpg)")
print("=" * 80)
img21 = cv2.imread(str(IMG_DIR / "LR00140.jpg"))
f21_lesions = [c for c in cands if c["frame_number"] == 21 and c["annotation_label"] == "Diseased_Region"]
f21_leaf = next(c for c in cands if c["frame_number"] == 21 and c["annotation_label"] == "Leaf")

print(f"Frame 21 total lesions: {len(f21_lesions)}")
print(f"Primary leaf candidate bbox min/max: {np.min(np.array(f21_leaf['polygon_coordinates']).reshape(-1, 2), axis=0)} to {np.max(np.array(f21_leaf['polygon_coordinates']).reshape(-1, 2), axis=0)}")

# Cluster lesions by spatial position (x coordinate)
left_lesions = [c for c in f21_lesions if np.max(np.array(c["polygon_coordinates"]).reshape(-1, 2)[:, 0]) < 180]
right_lesions = [c for c in f21_lesions if np.min(np.array(c["polygon_coordinates"]).reshape(-1, 2)[:, 0]) >= 180]
border_lesions = [c for c in f21_lesions if c not in left_lesions and c not in right_lesions]

print(f"Lesions strictly on left (x < 180): {len(left_lesions)}")
print(f"Lesions on right (x >= 180): {len(right_lesions)}")
print(f"Lesions crossing boundary: {len(border_lesions)}")

for c in left_lesions[:5]:
    pts = np.array(c["polygon_coordinates"]).reshape(-1, 2)
    bx1, by1 = np.min(pts, axis=0)
    bx2, by2 = np.max(pts, axis=0)
    roi = img21[int(by1):int(by2), int(bx1):int(bx2)]
    print(f"Left Candidate {c['candidate_id']}: bbox=[{bx1:.1f}, {by1:.1f}, {bx2:.1f}, {by2:.1f}], Area={c['area']:.1f}, Mean BGR={np.mean(roi, axis=(0,1)).round(1)}")

for c in right_lesions:
    pts = np.array(c["polygon_coordinates"]).reshape(-1, 2)
    bx1, by1 = np.min(pts, axis=0)
    bx2, by2 = np.max(pts, axis=0)
    roi = img21[int(by1):int(by2), int(bx1):int(bx2)]
    print(f"Right Candidate {c['candidate_id']}: bbox=[{bx1:.1f}, {by1:.1f}, {bx2:.1f}, {by2:.1f}], Area={c['area']:.1f}, Containment={c['leaf_containment_ratio']*100:.1f}%")
