"""
scripts/check_tracker_tasks.py
Finds all task IDs in summary table vs detailed sections in reports/MASTER_PROJECT_TASK_TRACKER.md.
"""

import re
from pathlib import Path

tracker_path = Path("reports/MASTER_PROJECT_TASK_TRACKER.md")
content = tracker_path.read_text(encoding="utf-8")

# Extract task IDs from summary table
table_tasks = []
in_table = False
for line in content.splitlines():
    if line.startswith("| `TASK-"):
        m = re.match(r"\|\s*`(TASK-[^`]+)`", line)
        if m:
            table_tasks.append(m.group(1))

# Extract task IDs from detailed headings: ### TASK-...
detail_tasks = []
for line in content.splitlines():
    if line.startswith("### TASK-"):
        m = re.match(r"###\s*(TASK-[^:]+):?", line)
        if m:
            detail_tasks.append(m.group(1).strip())

print(f"Total Summary Table Tasks: {len(table_tasks)}")
print(f"Total Detailed Tasks:      {len(detail_tasks)}")

table_set = set(table_tasks)
detail_set = set(detail_tasks)

summary_only = [t for t in table_tasks if t not in detail_set]
detail_only = [t for t in detail_tasks if t not in table_set]

print(f"\nTasks in Summary Table but NOT in Detailed Sections ({len(summary_only)}):")
for t in summary_only:
    print(f"  - {t}")

if detail_only:
    print(f"\nTasks in Detailed Sections but NOT in Summary Table ({len(detail_only)}):")
    for t in detail_only:
        print(f"  - {t}")
