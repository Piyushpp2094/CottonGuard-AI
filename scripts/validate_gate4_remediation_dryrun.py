"""
scripts/validate_gate4_remediation_dryrun.py
CottonGuard AI — Gate 4 Zero-Mutation Dry-Run Validator.

Validates the planned 14-image CVAT Task #6 remediation transactions entirely offline.
Reads the authoritative baseline backup JSON and simulates all proposed additions,
deletions, and retentions in memory without making any HTTP requests or CVAT modifications.
"""

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Main_Project\cotton_leaf_ai")
BASELINE_BACKUP_PATH = PROJECT_ROOT / "artifacts" / "backups" / "cvat_job6_fresh_pre_remediation_baseline_20261118.json"
EXPECTED_BASELINE_SHA256 = "b516c238c330ee48b3a3f8492c694fac0165fc7e37c5ffafcedd89e8fff2e145"
EXPECTED_BASELINE_SHAPES = 1170
EXPECTED_BASELINE_FRAMES = 50

# The 14 targeted frames in Batch 03 (Global IDs 50 to 84)
REMEDIATION_FRAMES = {
    0:  {"global_id": 50, "filename": "BBC00001.jpg", "category": "Leaf-Only"},
    9:  {"global_id": 59, "filename": "CV00431.jpg",  "category": "Leaf-Only"},
    15: {"global_id": 65, "filename": "LHJ00001.jpg", "category": "Both"},
    17: {"global_id": 67, "filename": "LHJ00116.jpg", "category": "Both"},
    18: {"global_id": 68, "filename": "LHJ00172.jpg", "category": "Lesion-Only"},
    19: {"global_id": 69, "filename": "LHJ00225.jpg", "category": "Lesion-Only"},
    20: {"global_id": 70, "filename": "LR00001.jpg",  "category": "Leaf-Only"},
    21: {"global_id": 71, "filename": "LR00140.jpg",  "category": "Both"},
    22: {"global_id": 72, "filename": "LR00286.jpg",  "category": "Lesion-Only"},
    23: {"global_id": 73, "filename": "LR00440.jpg",  "category": "Lesion-Only"},
    24: {"global_id": 74, "filename": "LR00578.jpg",  "category": "Both"},
    26: {"global_id": 76, "filename": "LV00028.jpg",  "category": "Leaf-Only"},
    28: {"global_id": 78, "filename": "LV00087.jpg",  "category": "Leaf-Only"},
    31: {"global_id": 81, "filename": "HL00070.jpg",  "category": "Leaf-Only"},
}

# Explicit Deletion IDs per Frame
DELETIONS_BY_FRAME = {
    0:  {"leaf": [1039], "rect": [], "poly_4pt": []},
    9:  {"leaf": [1143], "rect": [], "poly_4pt": []},
    15: {"leaf": [1157], "rect": [1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187],
                         "poly_4pt": [1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172]},
    17: {"leaf": [1201], "rect": [1205, 1206, 1207], "poly_4pt": [1203, 1204]},
    18: {"leaf": [],     "rect": [1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244],
                         "poly_4pt": [1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226]},
    19: {"leaf": [],     "rect": [1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275],
                         "poly_4pt": [1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260]},
    20: {"leaf": [1276], "rect": [], "poly_4pt": []},
    21: {"leaf": [1289], "rect": list(range(1319, 1345)), "poly_4pt": list(range(1290, 1316))},
    22: {"leaf": [],     "rect": list(range(1384, 1419)), "poly_4pt": list(range(1346, 1381))},
    23: {"leaf": [],     "rect": list(range(1441, 1462)), "poly_4pt": list(range(1420, 1441))},
    24: {"leaf": [1462], "rect": list(range(1483, 1500)), "poly_4pt": list(range(1463, 1480))},
    26: {"leaf": [1567], "rect": [], "poly_4pt": []}, # Rectangles 1582-1595 kept under Policy A
    28: {"leaf": [1681], "rect": [], "poly_4pt": []}, # Rectangles 1705-1727 kept under Policy A
    31: {"leaf": [1756], "rect": [], "poly_4pt": []},
}

# Explicit Additions Count per Frame
ADDITIONS_BY_FRAME = {
    0:  {"leaf": 1, "lesion": 0},
    9:  {"leaf": 1, "lesion": 0},
    15: {"leaf": 1, "lesion": 15},
    17: {"leaf": 1, "lesion": 2},
    18: {"leaf": 0, "lesion": 17},
    19: {"leaf": 0, "lesion": 10},
    20: {"leaf": 1, "lesion": 0},
    21: {"leaf": 1, "lesion": 26},
    22: {"leaf": 0, "lesion": 35},
    23: {"leaf": 0, "lesion": 21},
    24: {"leaf": 1, "lesion": 17},
    26: {"leaf": 1, "lesion": 0},
    28: {"leaf": 1, "lesion": 0},
    31: {"leaf": 1, "lesion": 0},
}


def calc_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def run_dryrun():
    print("=" * 80)
    print("CottonGuard AI — Gate 4 Preflight Dry-Run Transaction Validator")
    print("MODE: STRICT READ-ONLY / ZERO-MUTATION SIMULATION")
    print("=" * 80)

    # 1. Verify Baseline Backup Hash
    actual_hash = calc_sha256(BASELINE_BACKUP_PATH)
    print(f"\n[STEP 1] Verifying Baseline Backup SHA-256:")
    print(f"  Path:     {BASELINE_BACKUP_PATH}")
    print(f"  Computed: {actual_hash}")
    print(f"  Expected: {EXPECTED_BASELINE_SHA256}")
    assert actual_hash == EXPECTED_BASELINE_SHA256, "Cryptographic hash mismatch on baseline backup!"
    print("  --> HASH VERIFICATION PASSED.")

    # 2. Parse Baseline Data
    with open(BASELINE_BACKUP_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    shapes = raw_data.get("shapes", [])
    print(f"\n[STEP 2] Verifying Baseline Shape Invariants:")
    print(f"  Total baseline shapes: {len(shapes)} (Expected: {EXPECTED_BASELINE_SHAPES})")
    assert len(shapes) == EXPECTED_BASELINE_SHAPES, "Baseline shape count mismatch!"

    shape_by_id = {s["id"]: s for s in shapes}
    assert len(shape_by_id) == len(shapes), "Duplicate shape IDs detected in baseline!"

    shapes_by_frame = defaultdict(list)
    for s in shapes:
        shapes_by_frame[s["frame"]].append(s)

    # 3. Validate Every Targeted Deletion
    print(f"\n[STEP 3] Validating Exact Target Shape IDs for Proposed Deletions:")
    all_deleted_ids = set()
    total_leaf_del = 0
    total_rect_del = 0
    total_poly4_del = 0

    for frame_idx, dels in sorted(DELETIONS_BY_FRAME.items()):
        meta = REMEDIATION_FRAMES[frame_idx]
        print(f"  Frame {frame_idx:02d} ({meta['filename']}, Global {meta['global_id']}):")

        # Validate Leaf deletions
        for sid in dels["leaf"]:
            assert sid in shape_by_id, f"Leaf shape ID {sid} not found in baseline!"
            s = shape_by_id[sid]
            assert s["frame"] == frame_idx, f"Shape {sid} frame mismatch: expected {frame_idx}, got {s['frame']}"
            assert s["label_id"] == 5, f"Shape {sid} is not a Leaf (label_id != 5)"
            all_deleted_ids.add(sid)
            total_leaf_del += 1
            print(f"    - Leaf DELETE ID={sid} (pts={len(s['points']) // 2}) verified.")

        # Validate Rect deletions
        for sid in dels["rect"]:
            assert sid in shape_by_id, f"Rectangle shape ID {sid} not found in baseline!"
            s = shape_by_id[sid]
            assert s["frame"] == frame_idx, f"Shape {sid} frame mismatch!"
            assert s["type"] == "rectangle", f"Shape {sid} is not a rectangle!"
            all_deleted_ids.add(sid)
            total_rect_del += 1

        if dels["rect"]:
            print(f"    - Rectangles DELETE: {len(dels['rect'])} shapes verified.")

        # Validate 4-pt poly deletions
        for sid in dels["poly_4pt"]:
            assert sid in shape_by_id, f"4-pt poly shape ID {sid} not found in baseline!"
            s = shape_by_id[sid]
            assert s["frame"] == frame_idx, f"Shape {sid} frame mismatch!"
            assert s["type"] == "polygon" and len(s["points"]) == 8, f"Shape {sid} is not an 8-coord 4-pt polygon!"
            all_deleted_ids.add(sid)
            total_poly4_del += 1

        if dels["poly_4pt"]:
            print(f"    - 4-pt Polygons DELETE: {len(dels['poly_4pt'])} shapes verified.")

    print(f"\n  Total Proposed Deletions (Policy A):")
    print(f"    Leaf Pseudo-Masks:      {total_leaf_del:3d}")
    print(f"    Explicit Rectangles:    {total_rect_del:3d}")
    print(f"    4-pt BBox Replicas:     {total_poly4_del:3d}")
    print(f"    TOTAL DELETIONS:        {len(all_deleted_ids):3d}")
    assert len(all_deleted_ids) == 303, f"Expected 303 deletions, got {len(all_deleted_ids)}"

    # 4. Validate Additions Arithmetic
    total_leaf_add = sum(a["leaf"] for a in ADDITIONS_BY_FRAME.values())
    total_lesion_add = sum(a["lesion"] for a in ADDITIONS_BY_FRAME.values())
    total_additions = total_leaf_add + total_lesion_add

    print(f"\n[STEP 4] Validating Proposed Additions Arithmetic:")
    print(f"    Replacement Organic Leaf Polygons:  {total_leaf_add:3d}")
    print(f"    New Organic Lesion Contours:        {total_lesion_add:3d}")
    print(f"    TOTAL ADDITIONS:                    {total_additions:3d}")
    assert total_leaf_add == 10, f"Expected 10 leaf additions, got {total_leaf_add}"
    assert total_lesion_add == 143, f"Expected 143 lesion additions, got {total_lesion_add}"
    assert total_additions == 153, f"Expected 153 total additions, got {total_additions}"

    # 5. In-Memory Post-Remediation Shape Simulation (Policy A)
    print(f"\n[STEP 5] In-Memory Transaction Execution (Zero Network / Zero CVAT Mutation):")
    simulated_shapes = [s for s in shapes if s["id"] not in all_deleted_ids]
    # Simulate addition shapes
    # 10 leaf shapes + 143 lesion shapes
    print(f"    Retained Baseline Shapes:           {len(simulated_shapes):4d}")
    print(f"    Plus Simulated Additions:           {total_additions:4d}")
    final_shape_count = len(simulated_shapes) + total_additions
    print(f"    EXPECTED FINAL GLOBAL SHAPES:       {final_shape_count:4d}")
    assert final_shape_count == 1020, f"Expected 1020 final shapes under Policy A, got {final_shape_count}"

    # Verify per-type final counts
    retained_rects = len([s for s in simulated_shapes if s["type"] == "rectangle"])
    retained_polys = len([s for s in simulated_shapes if s["type"] == "polygon"])
    final_rects = retained_rects # No rectangles added
    final_polys = retained_polys + total_additions # All additions are polygons

    print(f"\n    Breakdown by Shape Type (Policy A):")
    print(f"      Baseline Rectangles: {len([s for s in shapes if s['type'] == 'rectangle']):4d}  --> Final Rectangles: {final_rects:4d} (Delta: -{150})")
    print(f"      Baseline Polygons:   {len([s for s in shapes if s['type'] == 'polygon']):4d}  --> Final Polygons:   {final_polys:4d} (Delta: 0)")
    assert final_rects == 345, f"Expected 345 rectangles, got {final_rects}"
    assert final_polys == 675, f"Expected 675 polygons, got {final_polys}"

    # 6. Verify Holdout Isolation (Frames 35 to 49)
    print(f"\n[STEP 6] Verifying Strict Holdout Isolation:")
    holdout_shapes = [s for s in shapes if s["frame"] >= 35]
    holdout_deleted = [s for s in holdout_shapes if s["id"] in all_deleted_ids]
    print(f"    Holdout Frames: 35 to 49 (Global IDs 85 to 99)")
    print(f"    Total Holdout Shapes in Baseline: {len(holdout_shapes)} (Expected: 449)")
    print(f"    Holdout Shapes Targeted for Mutation: {len(holdout_deleted)}")
    assert len(holdout_shapes) == 449, f"Holdout shape count expected 449, got {len(holdout_shapes)}"
    assert len(holdout_deleted) == 0, "CRITICAL ERROR: Holdout shapes were targeted by remediation!"
    print("    --> ZERO-MUTATION HOLDOUT ISOLATION CONFIRMED.")

    print("\n" + "=" * 80)
    print("DRY-RUN VALIDATION COMPLETE: ALL 14 FRAMES & ARITHMETIC 100% VERIFIED.")
    print("ZERO LIVE REQUESTS SENT. ZERO MUTATIONS PERFORMED.")
    print("APPROVAL_GATE_4 REMAINS: PENDING_APPROVAL")
    print("=" * 80)


if __name__ == "__main__":
    run_dryrun()
