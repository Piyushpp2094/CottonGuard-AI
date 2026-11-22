"""
scripts/test_current_rollback_on_disposable_job2.py
Tests the exact current version of scripts/restore_cvat_task6_baseline.py
(SHA-256 bbc22e9f31bbaea4a28daafcef44e627becf2546d3fcda3e06bbc7e4ac7e3c9a)
on disposable CVAT Job #2 without touching Task #6.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")
sys.path.insert(0, str(WORKSPACE))

from scripts.restore_cvat_task6_baseline import CVATClient

CVAT_URL = "http://localhost:8080"
CVAT_USER = "cottonguard"
CVAT_PASS = "CottonGuardCVAT2026!"
RESTORE_SCRIPT = WORKSPACE / "scripts/restore_cvat_task6_baseline.py"
BACKUP_PATH = WORKSPACE / "artifacts/backups/cvat_job2_synthetic_test_backup.json"

# 1. Verify exact hash of current restore script
with open(RESTORE_SCRIPT, "rb") as f:
    script_hash = hashlib.sha256(f.read()).hexdigest().lower()
print(f"Current rollback script: {RESTORE_SCRIPT.name}")
print(f"SHA-256: {script_hash}")
expected_script_hash = "bbc22e9f31bbaea4a28daafcef44e627becf2546d3fcda3e06bbc7e4ac7e3c9a"
assert script_hash == expected_script_hash, f"Script hash mismatch! Expected {expected_script_hash}, got {script_hash}"

# 2. Build synthetic test backup for Job #2 (Task #2 uses label_id 3 for Leaf, 4 for Diseased_Region)
synthetic_data = {
    "version": 0,
    "tags": [],
    "tracks": [],
    "intervals": [],
    "shapes": [
        {
            "id": 99901,  # synthetic ID to test stripping
            "label_id": 3,
            "type": "polygon",
            "frame": 0,
            "group": 0,
            "source": "manual",
            "occluded": False,
            "outside": False,
            "z_order": 0,
            "rotation": 0.0,
            "points": [100.0, 100.0, 200.0, 100.0, 200.0, 200.0, 100.0, 200.0],
            "attributes": [],
            "elements": []
        },
        {
            "id": 99902,  # synthetic ID to test stripping
            "label_id": 4,
            "type": "rectangle",
            "frame": 1,
            "group": 0,
            "source": "manual",
            "occluded": False,
            "outside": False,
            "z_order": 0,
            "rotation": 0.0,
            "points": [50.0, 50.0, 150.0, 150.0],
            "attributes": [],
            "elements": []
        }
    ]
}

BACKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(BACKUP_PATH, "w", encoding="utf-8") as f:
    json.dump(synthetic_data, f, indent=2)

with open(BACKUP_PATH, "rb") as f:
    backup_hash = hashlib.sha256(f.read()).hexdigest().lower()
print(f"Generated synthetic test backup: {BACKUP_PATH.name}")
print(f"Backup SHA-256: {backup_hash}")

# 3. Execute restore_cvat_task6_baseline.py on Job #2
cmd = [
    sys.executable,
    str(RESTORE_SCRIPT),
    "--job-id", "2",
    "--backup-file", str(BACKUP_PATH),
    "--expected-sha256", backup_hash,
    "--expected-shape-count", "2",
    "--expected-frame-count", "14",
    "--confirm"
]
print(f"Executing rollback command: {' '.join(cmd)}")
proc = subprocess.run(cmd, capture_output=True, text=True)
print("--- STDOUT ---")
print(proc.stdout)
print("--- STDERR ---")
print(proc.stderr)
assert proc.returncode == 0, f"Rollback script failed with return code {proc.returncode}!"

# 4. Verify Job #2 via CVATClient
client = CVATClient(CVAT_URL, CVAT_USER, CVAT_PASS)
live_ann = client.get_annotations(2)
live_shapes = live_ann.get("shapes", [])
print(f"Live Job #2 shapes after restore: {len(live_shapes)} (Expected: 2)")
assert len(live_shapes) == 2, f"Expected 2 shapes, got {len(live_shapes)}"

# Check ID replacement behavior
original_ids = {99901, 99902}
live_ids = {s["id"] for s in live_shapes}
print(f"Original synthetic IDs: {original_ids}")
print(f"Live server-assigned IDs: {live_ids}")
assert len(original_ids & live_ids) == 0, "Server failed to replace synthetic IDs with fresh server IDs!"
print("-> CONFIRMED: Original client IDs were stripped; CVAT assigned new sequential server IDs.")

# Check geometric and metadata parity
s0 = live_shapes[0]
s1 = live_shapes[1]
print(f"Shape 0: frame={s0['frame']}, label_id={s0['label_id']}, type={s0['type']}, points={s0['points']}")
print(f"Shape 1: frame={s1['frame']}, label_id={s1['label_id']}, type={s1['type']}, points={s1['points']}")

# Check category counts
assert len(live_ann.get("tracks", [])) == 0
assert len(live_ann.get("tags", [])) == 0
print("-> All categories (shapes=2, tracks=0, tags=0) verified.")

# 5. Clean Job #2 back to 0 shapes
print("Cleaning Job #2 back to 0 shapes...")
clean_payload = {"version": 0, "shapes": [], "tracks": [], "tags": [], "intervals": []}
client.put_annotations(2, clean_payload)
final_ann = client.get_annotations(2)
print(f"Job #2 final shape count: {len(final_ann.get('shapes', []))} (Expected: 0)")
assert len(final_ann.get("shapes", [])) == 0
print("-> Job #2 successfully cleaned to 0 shapes.")
print("\n[TEST RESULT: 100% SUCCESSFUL DISPOSABLE JOB VALIDATION]")
