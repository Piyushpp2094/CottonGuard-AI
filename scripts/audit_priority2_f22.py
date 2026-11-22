"""
scripts/audit_priority2_f22.py
Forensic inspection of Frame 22 (LR00286.jpg) for Stage 2.1.
Examines all 35 lesion candidates against host leaf and image pixels.
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

img22 = cv2.imread(str(IMG_DIR / "LR00286.jpg"))

# Frame 22 in baseline: what is the leaf?
# In baseline, Frame 22 has organic polygon leaf shape
f22_shapes = [s for s in cvat_data["shapes"] if s["frame"] == 22]
f22_leaf = next(s for s in f22_shapes if s["label_id"] == 5)
leaf_pts = np.array(f22_leaf["points"], dtype=np.float32).reshape(-1, 2)

f22_lesions = [c for c in cands if c["frame_number"] == 22 and c["annotation_label"] == "Diseased_Region"]
print("=" * 80)
print(f"FORENSIC AUDIT: FRAME 22 (LR00286.jpg) — {len(f22_lesions)} LESIONS")
print("=" * 80)
print(f"Leaf shape ID: {f22_leaf['id']}, vertices: {len(leaf_pts)}, type: {f22_leaf['type']}")
print(f"Leaf bbox: min={np.min(leaf_pts, axis=0)}, max={np.max(leaf_pts, axis=0)}")

# Rasterize leaf
l_mask = np.zeros((800, 800), dtype=np.uint8)
cv2.fillPoly(l_mask, [np.round(leaf_pts).astype(np.int32).reshape(-1, 1, 2)], 1)

print("\n--- Auditing every lesion candidate in Frame 22 ---")
for idx, c in enumerate(f22_lesions):
    pts = np.array(c["polygon_coordinates"]).reshape(-1, 2)
    bx1, by1 = np.min(pts, axis=0)
    bx2, by2 = np.max(pts, axis=0)
    
    # Rasterize lesion
    les_mask = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(les_mask, [np.round(pts).astype(np.int32).reshape(-1, 1, 2)], 1)
    
    inter = np.logical_and(l_mask == 1, les_mask == 1).sum()
    total = np.sum(les_mask == 1)
    ratio = inter / total if total > 0 else 1.0
    
    # Check pixel color in original image ROI
    roi = img22[int(by1):int(by2), int(bx1):int(bx2)]
    mean_bgr = np.mean(roi, axis=(0,1)) if roi.size > 0 else [0,0,0]
    
    print(f"[{idx+1:02d}/35] {c['candidate_id']}: bbox=[{bx1:.1f}, {by1:.1f}, {bx2:.1f}, {by2:.1f}], Area={c['area']:.1f}, Verts={c['vertex_count']}, Containment={ratio*100:5.1f}%, Mean BGR={mean_bgr.round(1)}")
