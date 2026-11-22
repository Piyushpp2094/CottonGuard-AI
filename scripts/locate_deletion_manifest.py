"""
scripts/locate_deletion_manifest.py
Searches artifacts, data, reports, and scripts for deletion manifests.
"""

import json
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")

def main():
    target_dirs = ["artifacts", "data", "reports", "scripts"]
    found_files = []
    for d in target_dirs:
        for p in (WORKSPACE / d).rglob("*"):
            try:
                if p.is_file() and p.suffix in [".json", ".csv"]:
                    name_lower = p.name.lower()
                    if "delet" in name_lower or "303" in name_lower or "obsolete" in name_lower or "manifest" in name_lower:
                        found_files.append(p)
            except Exception:
                pass

    print(f"Candidate manifest files found ({len(found_files)}):")
    for f in found_files:
        print(f"  {f.relative_to(WORKSPACE)}")

if __name__ == "__main__":
    main()
