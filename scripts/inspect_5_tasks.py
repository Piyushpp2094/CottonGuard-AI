"""
scripts/inspect_5_tasks.py
Prints summary table rows for the 5 tasks.
"""

from pathlib import Path

tracker_path = Path("reports/MASTER_PROJECT_TASK_TRACKER.md")
content = tracker_path.read_text(encoding="utf-8")

targets = {
    "TASK-PHASE12-2-STAGE2-3-FINAL-PRECISION-PREFLIGHT",
    "TASK-PHASE12-2-STAGE2-4-MANIFEST-INTEGRITY-RECONCILIATION",
    "TASK-FULL-TEST-VERIFY",
    "TASK-WINDOWS-DEPLOY-SCRIPTS",
    "TASK-PROJECT-REVIEW-PACKAGE"
}

for line in content.splitlines():
    for t in targets:
        if t in line:
            print(line)
