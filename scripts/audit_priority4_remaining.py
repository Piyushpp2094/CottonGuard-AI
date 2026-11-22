"""
scripts/audit_priority4_remaining.py
Forensic inspection of Frames 09, 17, 23, 26, 28 for Stage 2.1.
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

# --- FRAME 09 (CV00431.jpg) ---
print("=" * 80)
print("FORENSIC AUDIT: FRAME 09 (CV00431.jpg - Curl Virus)")
print("=" * 80)
f9_leaf = next(c for c in cands if c["frame_number"] == 9 and c["annotation_label"] == "Leaf")
print(f"Frame 09 Leaf Cand: {f9_leaf['candidate_id']}, Vertices: {f9_leaf['vertex_count']}, Area: {f9_leaf['area']}")
f9_shapes = [s for s in cvat_data["shapes"] if s["frame"] == 9]
print("Frame 09 shapes in baseline:", [(s["id"], s["label_id"], s["type"]) for s in f9_shapes])

# --- FRAME 17 (LHJ00116.jpg) ---
print("\n" + "=" * 80)
print("FORENSIC AUDIT: FRAME 17 (LHJ00116.jpg - Leaf Hopper Jassids)")
print("=" * 80)
f17_shapes = [s for s in cvat_data["shapes"] if s["frame"] == 17]
for s in f17_shapes:
    print(f"Shape {s['id']}: label={s['label_id']}, type={s['type']}, bbox={np.min(np.array(s['points']).reshape(-1,2), axis=0)} to {np.max(np.array(s['points']).reshape(-1,2), axis=0)}")

# --- FRAME 23 (LR00440.jpg) ---
print("\n" + "=" * 80)
print("FORENSIC AUDIT: FRAME 23 (LR00440.jpg - Leaf Redding)")
print("=" * 80)
f23_lesions = [c for c in cands if c["frame_number"] == 23 and c["annotation_label"] == "Diseased_Region"]
f23_shapes = [s for s in cvat_data["shapes"] if s["frame"] == 23]
f23_leaf = next(s for s in f23_shapes if s["label_id"] == 5)
leaf23_pts = np.array(f23_leaf["points"]).reshape(-1, 2)
print(f"Frame 23 baseline Leaf {f23_leaf['id']}: bbox={np.min(leaf23_pts, axis=0)} to {np.max(leaf23_pts, axis=0)}")
print(f"Frame 23 lesions count: {len(f23_lesions)}")
for c in f23_lesions[:5]:
    pts = np.array(c["polygon_coordinates"]).reshape(-1, 2)
    print(f"{c['candidate_id']}: bbox={np.min(pts, axis=0)} to {np.max(pts, axis=0)}, containment={c['leaf_containment_ratio']*100:.1f}%")

# --- FRAMES 26 & 28 (LV00028.jpg & LV00087.jpg - Leaf Variegation) ---
print("\n" + "=" * 80)
print("FORENSIC AUDIT: FRAMES 26 & 28 (Leaf Variegation)")
print("=" * 80)
f26_shapes = [s for s in cvat_data["shapes"] if s["frame"] == 26]
f28_shapes = [s for s in cvat_data["shapes"] if s["frame"] == 28]
print(f"Frame 26 total shapes: {len(f26_shapes)}, rectangles: {sum(1 for s in f26_shapes if s['type']=='rectangle')}, polygons: {sum(1 for s in f26_shapes if s['type']=='polygon')}")
print(f"Frame 28 total shapes: {len(f28_shapes)}, rectangles: {sum(1 for s in f28_shapes if s['type']=='rectangle')}, polygons: {sum(1 for s in f28_shapes if s['type']=='polygon')}")
