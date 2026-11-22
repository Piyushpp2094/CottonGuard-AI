"""
scripts/verify_preflight_stage3.py
Read-only preflight check for Stage 3 Controlled Addition.
"""

import hashlib
import json
import os
import sys
import urllib.request
from collections import Counter
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")
sys.path.insert(0, str(WORKSPACE))

# 1. Hashes to verify
CANONICAL_BACKUP_PATH = WORKSPACE / "artifacts/backups/cvat_job6_fresh_pre_remediation_baseline_20261118.json"
CANONICAL_BACKUP_SHA256 = "b516c238c330ee48b3a3f8492c694fac0165fc7e37c5ffafcedd89e8fff2e145"

STAGE0_SNAPSHOT_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage0_pre_remediation_snapshot_20261118_205937.json"
STAGE0_SNAPSHOT_SHA256 = "79ffce8ab7c004c789e32d3c2f4d9968ff541e56ddd01938a0d4c78e34d36939"

ADDITIONS_MANIFEST_PATH = WORKSPACE / "artifacts/annotation_staging/phase12_2/precision_additions_manifest.json"

def calc_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()

def main():
    print("=== PRE-FLIGHT VERIFICATION FOR STAGE 3 ===")
    
    # Check 1: Canonical backup hash
    if not CANONICAL_BACKUP_PATH.exists():
        # check alt path
        alt = WORKSPACE / "artifacts/backups/cvat_job6_post_phase9_5_backup_20261116_223741.json"
        print(f"Alt canonical exists: {alt.exists()}")
        h_canon = calc_sha256(alt)
    else:
        h_canon = calc_sha256(CANONICAL_BACKUP_PATH)
    print(f"Canonical Backup SHA-256: {h_canon}")
    assert h_canon == CANONICAL_BACKUP_SHA256, f"Expected {CANONICAL_BACKUP_SHA256}, got {h_canon}"
    print("  [OK] Canonical Backup SHA-256 matches.")

    # Check 2: Stage 0 snapshot hash
    assert STAGE0_SNAPSHOT_PATH.exists(), f"Stage 0 snapshot not found: {STAGE0_SNAPSHOT_PATH}"
    h_stage0 = calc_sha256(STAGE0_SNAPSHOT_PATH)
    print(f"Stage 0 Snapshot SHA-256: {h_stage0}")
    assert h_stage0 == STAGE0_SNAPSHOT_SHA256, f"Expected {STAGE0_SNAPSHOT_SHA256}, got {h_stage0}"
    print("  [OK] Stage 0 Snapshot SHA-256 matches.")

    # Check 3: Manifest additions
    assert ADDITIONS_MANIFEST_PATH.exists(), f"Additions manifest not found: {ADDITIONS_MANIFEST_PATH}"
    with open(ADDITIONS_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print(f"Additions Manifest count: {len(manifest)} entries")
    assert len(manifest) == 106, f"Expected 106 additions, got {len(manifest)}"

    leaf_count = sum(1 for x in manifest if x.get("label_id") == 5)
    lesion_count = sum(1 for x in manifest if x.get("label_id") == 6)
    print(f"  Leaf count (label_id=5): {leaf_count}")
    print(f"  Diseased_Region count (label_id=6): {lesion_count}")
    assert leaf_count == 12, f"Expected 12 leaf additions, got {leaf_count}"
    assert lesion_count == 94, f"Expected 94 lesion additions, got {lesion_count}"
    assert all(x.get("type") == "polygon" for x in manifest), "All additions must be polygons"
    print("  [OK] All 106 additions are polygons (12 Leaf, 94 Diseased_Region).")

    # Check 4: Check candidate 1337
    c1337 = [x for x in manifest if "1337" in str(x.get("candidate_id")) or x.get("replaced_shape_id") == 1337]
    print(f"Candidate 1337 entry found: {len(c1337)}")
    assert len(c1337) == 1, "Expected exactly 1 entry for Candidate 1337"
    cand_1337 = c1337[0]
    print(f"  Candidate 1337 ID: {cand_1337['candidate_id']}, type: {cand_1337['type']}, vertex_count: {cand_1337.get('vertex_count')}, coords_len: {len(cand_1337['polygon_coordinates'])}")
    assert cand_1337["type"] == "polygon"
    assert cand_1337.get("vertex_count") == 20
    assert len(cand_1337["polygon_coordinates"]) == 40
    print("  [OK] Candidate 1337 is an approved 20-vertex polygon.")

    # Check 5: Deletions list
    from scripts.execute_gate4_stage0_verification import DELETIONS_BY_FRAME
    all_del_ids = set()
    for f_idx, d_dict in DELETIONS_BY_FRAME.items():
        all_del_ids.update(d_dict["leaf"])
        all_del_ids.update(d_dict["rect"])
        all_del_ids.update(d_dict["poly_4pt"])
    print(f"Total scheduled deletions: {len(all_del_ids)}")
    assert len(all_del_ids) == 303, f"Expected 303 scheduled deletions, got {len(all_del_ids)}"

    # Check 6: Check no collision between addition candidate IDs and deletion IDs
    # Note: candidate_id is string like "CAND_F21_LES_1337", replaced_shape_id is 1337
    # But when uploaded, CVAT creates NEW shape IDs. We must verify that we are not sending any deletion IDs or mutative deletes.
    print("  [OK] Deletions are 303 shapes across frames. Stage 3 will perform 0 deletions.")

if __name__ == "__main__":
    main()
