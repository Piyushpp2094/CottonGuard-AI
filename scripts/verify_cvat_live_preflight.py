"""
scripts/verify_cvat_live_preflight.py
Verifies live CVAT Task #6 / Job #6 state before Stage 3 additions,
takes fresh pre-Stage-3 snapshot, and asserts all pre-conditions.
"""

import hashlib
import json
import os
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")
sys.path.insert(0, str(WORKSPACE))

from scripts.execute_gate4_stage0_verification import DELETIONS_BY_FRAME
from scripts.restore_cvat_task6_baseline import CVATClient

CVAT_URL = os.environ.get("CVAT_URL", "http://localhost:8080")
CVAT_USER = os.environ.get("CVAT_USERNAME", "cottonguard")
CVAT_PASS = os.environ.get("CVAT_PASSWORD", "CottonGuardCVAT2026!")

SNAPSHOT_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage3_pre_addition_snapshot.json"

def calc_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()

def main():
    print("=== LIVE CVAT PRE-STAGE 3 VERIFICATION ===")
    client = CVATClient(CVAT_URL, CVAT_USER, CVAT_PASS)
    print("  [OK] Successfully authenticated to CVAT.")

    # 1. Server info
    try:
        req = urllib.request.Request(f"{CVAT_URL}/api/server/about", headers=client._get_headers())
        with urllib.request.urlopen(req, timeout=10) as resp:
            about = json.loads(resp.read().decode("utf-8"))
            print(f"  CVAT Server Version: {about.get('version')} (Server time: {about.get('server', {}).get('time', 'unknown')})")
    except Exception as e:
        print("  Could not fetch /api/server/about:", e)

    # 2. Verify Task #6 and Job #6
    job = client.get_job(6)
    task_id = job.get("task_id")
    print(f"  Job #6 details: task_id={task_id}, status={job.get('status')}, stage={job.get('stage')}, state={job.get('state')}")
    assert task_id == 6, f"Job #6 belongs to task {task_id}, expected 6"

    # 3. Fetch live annotations
    ann = client.get_annotations(6)
    shapes = ann.get("shapes", [])
    total_shapes = len(shapes)
    print(f"  Live Shape Count: {total_shapes} (Expected: 1,170)")
    assert total_shapes == 1170, f"Expected 1,170 shapes, got {total_shapes}"

    # 4. Geometry and Label breakdown
    type_counts = Counter(s.get("type") for s in shapes)
    label_counts = Counter(s.get("label_id") for s in shapes)
    frame_counts = Counter(s.get("frame") for s in shapes)
    print(f"  Shape Types: {dict(type_counts)}")
    print(f"  Label IDs: {dict(label_counts)}")

    assert type_counts.get("rectangle") == 495, f"Expected 495 rectangles, got {type_counts.get('rectangle')}"
    assert type_counts.get("polygon") == 675, f"Expected 675 polygons, got {type_counts.get('polygon')}"
    assert label_counts.get(5) == 50, f"Expected 50 Leaf (id 5), got {label_counts.get(5)}"
    assert label_counts.get(6) == 1120, f"Expected 1,120 Diseased_Region (id 6), got {label_counts.get(6)}"

    # 5. DEC-01 37 rectangles in Frames 26 and 28
    f26_rects = [s for s in shapes if s.get("frame") == 26 and s.get("type") == "rectangle"]
    f28_rects = [s for s in shapes if s.get("frame") == 28 and s.get("type") == "rectangle"]
    print(f"  DEC-01 Rectangles: Frame 26={len(f26_rects)}, Frame 28={len(f28_rects)}, Total={len(f26_rects) + len(f28_rects)}")
    assert len(f26_rects) == 14, f"Expected 14 rects in Frame 26, got {len(f26_rects)}"
    assert len(f28_rects) == 23, f"Expected 23 rects in Frame 28, got {len(f28_rects)}"

    # 6. Holdouts (frames 35-49)
    holdout_shapes = [s for s in shapes if 35 <= s.get("frame", -1) <= 49]
    print(f"  Holdout Shapes (Frames 35-49): {len(holdout_shapes)} (Expected: 449)")
    assert len(holdout_shapes) == 449, f"Expected 449 holdout shapes, got {len(holdout_shapes)}"

    # 7. Check all 303 scheduled deletion shapes exist
    live_shape_ids = {s.get("id") for s in shapes}
    all_del_ids = set()
    for f_idx, d_dict in DELETIONS_BY_FRAME.items():
        all_del_ids.update(d_dict["leaf"])
        all_del_ids.update(d_dict["rect"])
        all_del_ids.update(d_dict["poly_4pt"])
    
    missing_deletions = all_del_ids - live_shape_ids
    print(f"  Verified 303 deletion IDs present in live: {len(all_del_ids - missing_deletions)}/303")
    assert len(missing_deletions) == 0, f"Missing deletion IDs in live CVAT: {missing_deletions}"

    # 8. Save pre-Stage-3 live snapshot
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=2)
    
    snapshot_hash = calc_sha256(SNAPSHOT_PATH)
    print(f"  Pre-Stage 3 Live Snapshot saved to: {SNAPSHOT_PATH}")
    print(f"  Snapshot SHA-256: {snapshot_hash}")
    print("=== LIVE PRE-STAGE 3 VERIFICATION PASSED ===")

if __name__ == "__main__":
    main()
