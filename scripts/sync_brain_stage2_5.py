"""
scripts/sync_brain_stage2_5.py
Copies phase12_2_stage2_5_host_leaf_and_label_integrity.md to Brain artifact directory.
"""

import shutil
from pathlib import Path

src = Path("artifacts/audit_phase11/phase12_2_stage2_5_host_leaf_and_label_integrity.md")
dst = Path(r"C:\Users\Haker's pc\.gemini\antigravity\brain\e96a0217-11ef-4314-af5b-e8eaa45d475a\phase12_2_stage2_5_host_leaf_and_label_integrity.md")

shutil.copy2(src, dst)
print(f"Copied {src} to {dst} ({dst.stat().st_size} bytes)")
