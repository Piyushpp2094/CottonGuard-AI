"""
scripts/sync_brain_tracker.py
Copies reports/MASTER_PROJECT_TASK_TRACKER.md to the Brain artifact location.
"""

import shutil
from pathlib import Path

src = Path("reports/MASTER_PROJECT_TASK_TRACKER.md")
dst = Path(r"C:\Users\Haker's pc\.gemini\antigravity\brain\e96a0217-11ef-4314-af5b-e8eaa45d475a\MASTER_PROJECT_TASK_TRACKER.md")

shutil.copy2(src, dst)
print(f"Copied {src} to {dst} ({dst.stat().st_size} bytes)")
