"""
scripts/list_cvat_tasks_jobs.py
Lists all tasks and jobs in CVAT to see if a disposable test task/job exists.
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

import urllib.request

client = CVATClient(CVAT_URL, CVAT_USER, CVAT_PASS)

# List tasks
req = urllib.request.Request(f"{CVAT_URL}/api/tasks", headers=client._get_headers())
with urllib.request.urlopen(req, timeout=10) as resp:
    tasks_data = json.loads(resp.read().decode("utf-8"))
    tasks = tasks_data.get("results", [])
    print(f"Total Tasks in CVAT: {len(tasks)}")
    for t in tasks:
        print(f"  Task ID {t['id']}: name='{t['name']}', size={t.get('size')}, status='{t.get('status')}'")

# List jobs
req = urllib.request.Request(f"{CVAT_URL}/api/jobs", headers=client._get_headers())
with urllib.request.urlopen(req, timeout=10) as resp:
    jobs_data = json.loads(resp.read().decode("utf-8"))
    jobs = jobs_data.get("results", [])
    print(f"\nTotal Jobs in CVAT: {len(jobs)}")
    for j in jobs:
        print(f"  Job ID {j['id']}: task_id={j.get('task_id')}, stage='{j.get('stage')}', state='{j.get('state')}'")
