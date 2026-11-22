"""
scripts/inspect_tracker_table_lines.py
Inspects duplicate rows in the summary table.
"""

from collections import Counter
import re
from pathlib import Path

content = Path("reports/MASTER_PROJECT_TASK_TRACKER.md").read_text(encoding="utf-8")
lines = content.splitlines()

tasks_with_line = []
for idx, line in enumerate(lines):
    if line.startswith("| `TASK-"):
        m = re.match(r"\|\s*`(TASK-[^`]+)`", line)
        if m:
            tasks_with_line.append((idx + 1, m.group(1), line))

c = Counter(t[1] for t in tasks_with_line)
print(f"Total rows starting with | `TASK-: {len(tasks_with_line)}")
print(f"Unique task IDs: {len(c)}")

duplicates = [task for task, count in c.items() if count > 1]
print(f"\nDuplicate task IDs in table ({len(duplicates)}):")
for d in duplicates:
    print(f"  Task: {d} (count: {c[d]})")
    for line_no, tid, l in tasks_with_line:
        if tid == d:
            print(f"    Line {line_no}: {l}")
