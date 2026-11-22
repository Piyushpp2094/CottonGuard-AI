"""
scripts/find_mask_code.py
Finds all occurrences of mask generation / rasterization in src and scripts.
"""

from pathlib import Path

for base in [Path("src"), Path("scripts")]:
    for p in base.rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            if any(term in txt.lower() for term in ["cv2.fillpoly", "leaf_union", "diseased_mask", "generate_mask", "rasterize", "unet_dataset"]):
                print(f"Match: {p}")
        except Exception as e:
            pass
