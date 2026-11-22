"""
scripts/find_unet_train.py
Finds U-Net dataset / training scripts.
"""

from pathlib import Path

for base in [Path("src"), Path("scripts")]:
    for p in base.rglob("*.py"):
        if "unet" in p.name.lower() or "segmentation" in p.name.lower():
            print(f"Match: {p}")
