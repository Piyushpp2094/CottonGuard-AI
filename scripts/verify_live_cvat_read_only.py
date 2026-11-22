"""
scripts/verify_live_cvat_read_only.py
Read-only verification of live CVAT Job #6 state.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")
sys.path.insert(0, str(WORKSPACE))

from scripts.restore_cvat_task6_baseline import CVATClient

CVAT_URL = os.environ.get("CVAT_URL", "http://localhost:8080")
CVAT_USER = os.environ.get("CVAT_USERNAME", "cottonguard")
CVAT_PASS = os.environ.get("CVAT_PASSWORD", "CottonGuardCVAT2026!")

client = CVATClient(CVAT_URL, CVAT_USER, CVAT_PASS)
ann = client.get_annotations(6)
shapes = ann.get("shapes", [])
print(f"Live CVAT Job #6 Total Shapes: {len(shapes)} (Expected: 1,276)")
assert len(shapes) == 1276

# Check label and type breakdowns
from collections import Counter
type_counts = Counter(s["type"] for s in shapes)
label_counts = Counter(s["label_id"] for s in shapes)
print(f"Live Types: {dict(type_counts)} (Expected: 495 rect, 781 poly)")
print(f"Live Labels: {dict(label_counts)} (Expected: 62 Leaf [id 5], 1214 Diseased [id 6])")
assert type_counts["rectangle"] == 495
assert type_counts["polygon"] == 781
assert label_counts[5] == 62
assert label_counts[6] == 1214

# Check SHA-256 of live payload
h = hashlib.sha256(json.dumps(ann, indent=2).encode("utf-8")).hexdigest().lower()
print(f"Live Annotations Normalized SHA-256: {h}")

POST_SNAPSHOT_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage3_post_addition_snapshot.json"
with open(POST_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
    snap = json.load(f)
h_snap = hashlib.sha256(json.dumps(snap, indent=2).encode("utf-8")).hexdigest().lower()
print(f"Post-Snapshot Normalized SHA-256:    {h_snap}")
assert h == h_snap, "Live annotations do not match post-Stage-3 snapshot!"
print("Live CVAT Job #6 perfectly matches post-Stage-3 snapshot (1,276 shapes).")
