"""
scripts/prepare_step3_option_b_payload.py
Prepares Option B payload for Phase 8.8 Step 3 CVAT push.
- Retains 19 confirmed genuine lesion polygons for Frame 0 (BBC00001.jpg).
- Excludes P02, P06, P08, P11, P19.
- Preserves original candidate JSON untouched.
- Verifies unique disease pixels (6,477), leaf pixels (605,221), and severity (1.0702%).
- Integrates approved Frame 30 (HL00002.jpg) Leaf polygon (0 disease).
- Prepares CVAT Job #6 PATCH/PUT payload.
"""

import json
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("D:/Main_Project/cotton_leaf_ai")
corrections_dir = PROJECT_ROOT / "artifacts" / "pilot_corrections" / "phase8_8"

# 1. Load original BBC00001 candidates
with open(corrections_dir / "bbc00001_corrected_polygons.json") as f:
    orig_bbc = json.load(f)

# 2. Load HL00002 candidates
with open(corrections_dir / "hl00002_corrected_polygon.json") as f:
    orig_hl = json.load(f)

def unpack_points(flat_pts):
    pts = []
    for i in range(0, len(flat_pts), 2):
        pts.append([int(round(flat_pts[i])), int(round(flat_pts[i+1]))])
    return np.array(pts, dtype=np.int32)

h, w = orig_bbc["height"], orig_bbc["width"]
leaf_pts = unpack_points(orig_bbc["leaf_shape"]["points"])
leaf_mask = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(leaf_mask, [leaf_pts], 1)
total_leaf_px = int(np.count_nonzero(leaf_mask))

# Exclude P02, P06, P08, P11, P19 (1-indexed)
excluded_ids = {2, 6, 8, 11, 19}
retained_shapes = []

for s in orig_bbc["disease_shapes"]:
    if s["polygon_id"] not in excluded_ids:
        retained_shapes.append(s)

print(f"Original disease shapes: {len(orig_bbc['disease_shapes'])}")
print(f"Retained disease shapes: {len(retained_shapes)}")
print(f"Excluded shape IDs: {sorted(list(excluded_ids))}")

# Check rasterized disease pixels
dis_mask = np.zeros((h, w), dtype=np.uint8)
for s in retained_shapes:
    pts = unpack_points(s["points"])
    cv2.fillPoly(dis_mask, [pts], 1)

unique_disease_px = int(np.count_nonzero(dis_mask))
severity_pct = (unique_disease_px / total_leaf_px) * 100

print(f"Unique rasterized disease pixels: {unique_disease_px} (expected 6477)")
print(f"Total leaf lamina pixels: {total_leaf_px} (expected 605221)")
print(f"Standardized severity: {severity_pct:.4f}% (expected 1.0702%)")

assert len(retained_shapes) == 19, f"Expected 19 shapes, got {len(retained_shapes)}"
assert unique_disease_px == 6477, f"Expected 6477 disease pixels, got {unique_disease_px}"
assert total_leaf_px == 605221, f"Expected 605221 leaf pixels, got {total_leaf_px}"
assert abs(severity_pct - 1.0702) < 0.0001, f"Expected 1.0702% severity, got {severity_pct:.4f}%"

# Renumber polygon IDs 1..19 for clean payload while preserving audit metadata
option_b_shapes = []
for new_id, s in enumerate(retained_shapes, 1):
    shape_copy = dict(s)
    shape_copy["original_candidate_id"] = s["polygon_id"]
    shape_copy["polygon_id"] = new_id
    option_b_shapes.append(shape_copy)

option_b_bbc = dict(orig_bbc)
option_b_bbc["status"] = "APPROVED_OPTION_B"
option_b_bbc["disease_polygons_count"] = 19
option_b_bbc["unique_disease_pixels"] = unique_disease_px
option_b_bbc["standardized_severity_pct"] = round(severity_pct, 4)
option_b_bbc["disease_shapes"] = option_b_shapes
option_b_bbc["excluded_polygon_ids"] = sorted(list(excluded_ids))

out_bbc_b = corrections_dir / "bbc00001_option_b_19polygons.json"
with open(out_bbc_b, "w") as f:
    json.dump(option_b_bbc, f, indent=2)
print(f"Saved Option B BBC00001 candidate to {out_bbc_b}")

# Also prepare unified CVAT shapes payload
# In CVAT API, shapes payload for a job/task is a list of shape objects.
# For Job #6:
# We need label IDs and attribute spec IDs as configured in CVAT Task #6.
# Let's inspect cvat_pilot_task6_setup.json or cvat_task6_verified_setup.json to get exact label_ids and attribute spec IDs.
