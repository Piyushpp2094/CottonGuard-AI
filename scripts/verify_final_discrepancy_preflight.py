"""
scripts/verify_final_discrepancy_preflight.py
Performs comprehensive read-only checks for Phase 12.2 Final Stage 4 Discrepancy Resolution.
"""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")
sys.path.insert(0, str(WORKSPACE))

from scripts.restore_cvat_task6_baseline import CVATClient

CVAT_URL = "http://localhost:8080"
CVAT_USER = "cottonguard"
CVAT_PASS = "CottonGuardCVAT2026!"

client = CVATClient(CVAT_URL, CVAT_USER, CVAT_PASS)

# 1. Fetch live annotations from Job #6
live_ann = client.get_annotations(6)
live_shapes = live_ann.get("shapes", [])
print(f"1. Live CVAT Job #6 Shape Count: {len(live_shapes)} (Expected: 1,276)")
assert len(live_shapes) == 1276

live_shapes_by_id = {s["id"]: s for s in live_shapes}

# 2. Check 303 Deletion IDs
REGISTRY_PATH = WORKSPACE / "artifacts/audit_phase11/stage4_frozen_deletion_registry.json"
with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
    registry = json.load(f)
reg_ids = {r["cvat_shape_id"] for r in registry}
assert len(reg_ids) == 303

present_del_ids = reg_ids & set(live_shapes_by_id.keys())
print(f"2. Deletion IDs Present in Live CVAT: {len(present_del_ids)}/303")
assert len(present_del_ids) == 303

# 3. Check 106 Stage 3 Additions
stage3_ids = set(range(2224, 2330))
present_stage3 = stage3_ids & set(live_shapes_by_id.keys())
print(f"3. Stage 3 Additions Present in Live CVAT: {len(present_stage3)}/106")
assert len(present_stage3) == 106

# 4. Check 37 DEC-01 Rectangles
f26_rects = {s["id"] for s in live_shapes if s["frame"] == 26 and s["type"] == "rectangle"}
f28_rects = {s["id"] for s in live_shapes if s["frame"] == 28 and s["type"] == "rectangle"}
dec01_ids = f26_rects | f28_rects
print(f"4. DEC-01 Rectangles Preserved: {len(dec01_ids)}/37 (Frame 26: {len(f26_rects)}, Frame 28: {len(f28_rects)})")
assert len(f26_rects) == 14 and len(f28_rects) == 23 and len(dec01_ids) == 37
assert len(dec01_ids & reg_ids) == 0

# 5. Check 449 Holdout Shapes (Frames 35-49)
holdout_shapes = [s for s in live_shapes if 35 <= s["frame"] <= 49]
holdout_ids = {s["id"] for s in holdout_shapes}
print(f"5. Holdout Shapes Preserved: {len(holdout_ids)}/449 across {len(set(s['frame'] for s in holdout_shapes))} frames")
assert len(holdout_ids) == 449
assert len(holdout_ids & reg_ids) == 0

# 6. Check Normalized Annotation Hash
import urllib.request
def get_canonical_json_hash(ann):
    # Sort shapes by ID
    s_sorted = sorted(ann.get("shapes", []), key=lambda x: x["id"])
    clean_ann = {
        "version": ann.get("version", 0),
        "shapes": s_sorted,
        "tracks": ann.get("tracks", []),
        "tags": ann.get("tags", []),
        "intervals": ann.get("intervals", [])
    }
    dumped = json.dumps(clean_ann, sort_keys=True)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

live_hash = get_canonical_json_hash(live_ann)
with open(WORKSPACE / "artifacts/backups/cvat_job6_stage3_post_addition_snapshot.json", "r", encoding="utf-8") as f:
    snap_post = json.load(f)
snap_hash = get_canonical_json_hash(snap_post)

print(f"6. Live CVAT Normalized Hash: {live_hash}")
print(f"   Post-Stage-3 Snapshot Hash: {snap_hash}")
assert live_hash == snap_hash
print("-> 100% ZERO MUTATIONS CONFIRMED. Live CVAT is pristine.")
