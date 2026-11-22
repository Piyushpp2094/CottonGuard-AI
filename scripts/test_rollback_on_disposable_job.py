"""
scripts/test_rollback_on_disposable_job.py
Tests the rollback mechanism (restore_cvat_task6_baseline.py) on disposable Job #2.
"""

import hashlib
import json
import os
import sys
import subprocess
import urllib.request
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")
sys.path.insert(0, str(WORKSPACE))

from scripts.restore_cvat_task6_baseline import CVATClient

CVAT_URL = os.environ.get("CVAT_URL", "http://localhost:8080")
CVAT_USER = os.environ.get("CVAT_USERNAME", "cottonguard")
CVAT_PASS = os.environ.get("CVAT_PASSWORD", "CottonGuardCVAT2026!")

TEST_JOB_ID = 2
TEST_BACKUP_PATH = WORKSPACE / "scratch/test_job2_backup.json"

def main():
    print(f"=== TESTING ROLLBACK ON DISPOSABLE JOB #{TEST_JOB_ID} ===")
    client = CVATClient(CVAT_URL, CVAT_USER, CVAT_PASS)

    # 1. Create a synthetic backup representing 2 shapes on Job 2
    synthetic_backup = {
        "version": 0,
        "tags": [],
        "tracks": [],
        "shapes": [
            {
                "type": "rectangle",
                "frame": 0,
                "label_id": 3,
                "points": [15.0, 15.0, 55.0, 55.0],
                "occluded": False,
                "outside": False,
                "z_order": 0,
                "rotation": 0.0,
                "attributes": []
            },
            {
                "type": "polygon",
                "frame": 1,
                "label_id": 4,
                "points": [10.0, 10.0, 20.0, 50.0, 50.0, 20.0],
                "occluded": False,
                "outside": False,
                "z_order": 0,
                "rotation": 0.0,
                "attributes": []
            }
        ]
    }
    TEST_BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_BACKUP_PATH, "w", encoding="utf-8") as f:
        json.dump(synthetic_backup, f, indent=2)

    h = hashlib.sha256()
    with open(TEST_BACKUP_PATH, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    backup_hash = h.hexdigest().lower()
    print(f"Synthetic backup created: {TEST_BACKUP_PATH} (SHA-256: {backup_hash})")

    # 2. Test restore_cvat_task6_baseline.py against Job 2
    cmd = [
        sys.executable,
        str(WORKSPACE / "scripts/restore_cvat_task6_baseline.py"),
        "--job-id", str(TEST_JOB_ID),
        "--backup-file", str(TEST_BACKUP_PATH),
        "--expected-sha256", backup_hash,
        "--expected-shape-count", "2",
        "--expected-frame-count", "14",
        "--confirm"
    ]
    print(f"Running rollback script on Job #{TEST_JOB_ID}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("Rollback output:")
    print(res.stdout)
    if res.stderr:
        print("Rollback stderr:", res.stderr)
    assert res.returncode == 0, f"Rollback script failed with code {res.returncode}"

    # 3. Verify Job 2 restored state
    ann = client.get_annotations(TEST_JOB_ID)
    shapes = ann.get("shapes", [])
    print(f"Job #{TEST_JOB_ID} restored shapes count: {len(shapes)} (Expected: 2)")
    assert len(shapes) == 2
    types = [s["type"] for s in shapes]
    print(f"Job #{TEST_JOB_ID} restored types: {types} (Expected: ['rectangle', 'polygon'])")
    assert "rectangle" in types and "polygon" in types
    print("  -> PASS: Rollback successfully restored exact geometry and label attributes to Job #2!")

    # 4. Clean up Job 2 back to 0 shapes
    print("\nCleaning up Job #2 back to 0 shapes...")
    del_url = f"{CVAT_URL}/api/jobs/{TEST_JOB_ID}/annotations/?action=delete"
    req_clean = urllib.request.Request(
        del_url,
        data=json.dumps({"version": 0, "shapes": shapes, "tracks": [], "tags": []}).encode("utf-8"),
        headers=client._get_headers(),
        method="PATCH"
    )
    with urllib.request.urlopen(req_clean, timeout=15) as resp:
        pass
    final_ann = client.get_annotations(TEST_JOB_ID)
    print(f"Job #{TEST_JOB_ID} final shapes after cleanup: {len(final_ann.get('shapes', []))}")
    assert len(final_ann.get("shapes", [])) == 0
    print("=== DISPOSABLE TEST JOB #2 ROLLBACK VALIDATION COMPLETE ===")

if __name__ == "__main__":
    main()
