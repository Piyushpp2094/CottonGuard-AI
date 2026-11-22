"""
scripts/inspect_candidates_manifest_ids.py
Examines replaced_shape_id and paired_poly_id in candidates_manifest.csv and compares with DELETIONS_BY_FRAME.
"""

import sys
from pathlib import Path
import pandas as pd

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")
sys.path.insert(0, str(WORKSPACE))

from scripts.execute_gate4_stage0_verification import DELETIONS_BY_FRAME

p_man = WORKSPACE / "artifacts/annotation_staging/phase12_2/candidates_manifest.csv"
df = pd.read_csv(p_man)

rep_ids = set(df["replaced_shape_id"].dropna().astype(int))
paired_ids = set(df["paired_poly_id"].dropna().astype(int))
combined_ids = rep_ids | paired_ids

all_303 = set()
for f, d in DELETIONS_BY_FRAME.items():
    all_303.update(d["leaf"] + d["rect"] + d["poly_4pt"])

diff = all_303 - combined_ids
print(f"Total in DELETIONS_BY_FRAME: {len(all_303)}")
print(f"Total in candidates_manifest (rep + paired): {len(combined_ids)}")
print(f"Difference (in 303 but not in candidates_manifest): {len(diff)}")

POST_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage3_post_addition_snapshot.json"
import json
with open(POST_PATH, "r", encoding="utf-8") as f:
    snap = json.load(f)
shapes_by_id = {s["id"]: s for s in snap["shapes"]}

for sid in sorted(diff):
    s = shapes_by_id[sid]
    print(f"  ID {sid}: Frame {s['frame']:02d}, Type={s['type']}, Label={s['label_id']}, Points={s['points']}")
