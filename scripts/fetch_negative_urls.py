"""
Fetches and caches download URLs for negative samples:
- PlantVillage non-cotton leaves (Potato, Tomato, Pepper, Corn, Apple)
- ImageNet objects/animals/vehicles from EliSchwartz/imagenet-sample-images
"""

import json
from pathlib import Path
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

categories = [
    "Potato___Early_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Pepper__bell___healthy",
    "Corn_(maize)___healthy",
    "Apple___healthy",
]

urls_by_cat = {}
for cat in categories:
    url = f"https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/{cat}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read())
            dl_urls = [x["download_url"] for x in data if x["name"].lower().endswith((".jpg", ".jpeg", ".png"))]
            urls_by_cat[cat] = dl_urls
            rem = resp.headers.get("x-ratelimit-remaining")
            print(f"{cat}: {len(dl_urls)} URLs (Remaining API rate limit: {rem})", flush=True)
    except Exception as e:
        print(f"{cat}: failed ({e})", flush=True)

out_file = DATA_DIR / "plantvillage_urls.json"
with open(out_file, "w") as f:
    json.dump(urls_by_cat, f, indent=2)

print(f"Saved {out_file} successfully! Total categories: {len(urls_by_cat)}", flush=True)
