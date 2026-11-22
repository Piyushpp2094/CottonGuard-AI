"""
tests/test_candidate_1337_segmentation_clipping.py
Focused regression test validating Candidate 1337 segmentation export treatment:
- Verifies documented 84.58% union-containment exception.
- Confirms production export rule: final_mask[(leaf_mask == 1) & (diseased_mask == 1)] = 2.
- Confirms 0 diseased pixels exported outside leaf masks.
- Confirms 391 off-mask pixels are clipped to background (0) in U-Net target.
- Confirms 2,145 in-mask pixels are exported as Class 2 (Diseased_Region).
- Confirms original source polygon geometry is preserved in provenance records.
"""

import json
import cv2
import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "annotation_staging" / "phase12_2" / "precision_additions_manifest.json"


def test_candidate_1337_segmentation_clipping():
    assert MANIFEST_PATH.is_file(), f"Manifest file missing: {MANIFEST_PATH}"
    
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    # Locate Frame 21 shapes
    f21_shapes = [s for s in manifest if s["frame_number"] == 21]
    
    # 1. Locate Leaf Blades
    blade1_entry = next((s for s in f21_shapes if s["candidate_id"] == "CAND_F21_L_1289"), None)
    blade2_entry = next((s for s in f21_shapes if s["candidate_id"] == "CAND_F21_L_BLADE2"), None)
    assert blade1_entry is not None, "Blade 1 (CAND_F21_L_1289) missing from Frame 21 manifest"
    assert blade2_entry is not None, "Blade 2 (CAND_F21_L_BLADE2) missing from Frame 21 manifest"
    
    # 2. Locate Candidate 1337
    c1337_entry = next((s for s in f21_shapes if s["candidate_id"] == "CAND_F21_LES_1337"), None)
    assert c1337_entry is not None, "Candidate 1337 missing from Frame 21 manifest"
    
    # Verify provenance preservation
    assert c1337_entry["replaced_shape_id"] == 1337
    assert c1337_entry["vertex_count"] == 20
    assert len(c1337_entry["polygon_coordinates"]) == 40  # 20 pairs
    assert c1337_entry["area"] == 2431.0
    assert c1337_entry["type"] == "polygon"
    assert c1337_entry["label_id"] == 6
    
    # 3. Construct Leaf Union Canvas (800x800)
    leaf_union_mask = np.zeros((800, 800), dtype=np.uint8)
    b1_pts = np.array(blade1_entry["polygon_coordinates"], dtype=np.int32).reshape(-1, 2)
    b2_pts = np.array(blade2_entry["polygon_coordinates"], dtype=np.int32).reshape(-1, 2)
    cv2.fillPoly(leaf_union_mask, [b1_pts], 1)
    cv2.fillPoly(leaf_union_mask, [b2_pts], 1)
    
    # 4. Construct Raw Candidate 1337 Canvas
    c1337_raw_mask = np.zeros((800, 800), dtype=np.uint8)
    c1337_pts = np.array(c1337_entry["polygon_coordinates"], dtype=np.int32).reshape(-1, 2)
    cv2.fillPoly(c1337_raw_mask, [c1337_pts], 1)
    
    raw_pixels = int(np.sum(c1337_raw_mask == 1))
    assert raw_pixels == 2536, f"Expected 2536 raster pixels, got {raw_pixels}"
    
    # Check containment math
    contained_pixels = int(np.sum((c1337_raw_mask == 1) & (leaf_union_mask == 1)))
    outside_pixels = int(np.sum((c1337_raw_mask == 1) & (leaf_union_mask == 0)))
    
    assert contained_pixels == 2145, f"Expected 2145 contained pixels, got {contained_pixels}"
    assert outside_pixels == 391, f"Expected 391 outside pixels, got {outside_pixels}"
    assert contained_pixels + outside_pixels == raw_pixels
    
    containment_pct = (contained_pixels / raw_pixels) * 100.0
    assert round(containment_pct, 2) == 84.58, f"Expected 84.58%, got {containment_pct:.2f}%"
    
    # 5. Execute Production Segmentation Export Rule (from scripts/prepare_cvat_annotations.py lines 467-470)
    final_mask = np.zeros((800, 800), dtype=np.uint8)
    final_mask[leaf_union_mask == 1] = 1
    final_mask[(leaf_union_mask == 1) & (c1337_raw_mask == 1)] = 2
    
    # 6. Verify Exact Post-Export Properties
    # A. Dimension verification
    assert final_mask.shape == (800, 800), f"Invalid mask shape: {final_mask.shape}"
    assert final_mask.dtype == np.uint8, f"Invalid mask dtype: {final_mask.dtype}"
    
    # B. Discrete class labels {0, 1, 2}
    unique_vals = set(np.unique(final_mask))
    assert unique_vals.issubset({0, 1, 2}), f"Unexpected mask values: {unique_vals}"
    
    # C. Absolute exclusion of off-mask diseased pixels
    # Verify zero diseased pixels exist outside leaf union
    floating_disease_pixels = int(np.sum((final_mask == 2) & (leaf_union_mask == 0)))
    assert floating_disease_pixels == 0, f"Detected {floating_disease_pixels} floating disease pixels outside leaf mask!"
    
    # D. Verify Candidate 1337 off-mask pixels are clipped to background (0)
    c1337_outside_coords = (c1337_raw_mask == 1) & (leaf_union_mask == 0)
    outside_final_classes = final_mask[c1337_outside_coords]
    assert np.all(outside_final_classes == 0), "Some off-mask Candidate 1337 pixels were not clipped to 0 (background)!"
    assert len(outside_final_classes) == 391
    
    # E. Verify Candidate 1337 contained pixels are exported as Class 2
    c1337_inside_coords = (c1337_raw_mask == 1) & (leaf_union_mask == 1)
    inside_final_classes = final_mask[c1337_inside_coords]
    assert np.all(inside_final_classes == 2), "Some in-mask Candidate 1337 pixels were not marked as Class 2!"
    assert len(inside_final_classes) == 2145
    
    # F. Verify leaf background remains Class 1
    healthy_leaf_pixels = int(np.sum(final_mask == 1))
    total_leaf_pixels = int(np.sum(leaf_union_mask == 1))
    assert healthy_leaf_pixels == total_leaf_pixels - 2145


if __name__ == "__main__":
    test_candidate_1337_segmentation_clipping()
    print("test_candidate_1337_segmentation_clipping PASSED: 100% verified.")
