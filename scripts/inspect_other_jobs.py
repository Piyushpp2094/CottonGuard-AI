"""
scripts/inspect_other_jobs.py
Inspects annotations in jobs 2, 3, 4, 5 to see if any is an empty or disposable test job.
"""

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

for jid in [2, 3, 4, 5, 6]:
    ann = client.get_annotations(jid)
    shapes = ann.get("shapes", [])
    print(f"Job #{jid}: {len(shapes)} shapes")
