"""
scripts/check_backup_hashes.py
Checks and verifies the cryptographic hashes and metadata of the two snapshots.
"""

import hashlib
import json
from pathlib import Path

f1 = Path("artifacts/backups/cvat_job6_fresh_pre_remediation_baseline_20261118.json")
f2 = Path("artifacts/backups/cvat_job6_stage0_pre_remediation_snapshot_20261118_205937.json")

def hash_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

print(f"File 1: {f1}")
print(f"  Exists: {f1.exists()}")
if f1.exists():
    h1 = hash_file(f1)
    print(f"  SHA-256: {h1}")
    with open(f1) as f:
        d1 = json.load(f)
    print(f"  Shape count: {len(d1.get('shapes', []))}")

print(f"\nFile 2: {f2}")
print(f"  Exists: {f2.exists()}")
if f2.exists():
    h2 = hash_file(f2)
    print(f"  SHA-256: {h2}")
    with open(f2) as f:
        d2 = json.load(f)
    print(f"  Shape count: {len(d2.get('shapes', []))}")
