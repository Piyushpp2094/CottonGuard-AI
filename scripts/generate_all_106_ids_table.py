"""
scripts/generate_all_106_ids_table.py
Generates the complete 106-row audit table for all new CVAT shape IDs (2224..2329).
"""

import json
import sys
from collections import Counter
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")
sys.path.insert(0, str(WORKSPACE))

POST_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage3_post_addition_snapshot.json"
MANIFEST_PATH = WORKSPACE / "artifacts/annotation_staging/phase12_2/precision_additions_manifest.json"
MAPPING_PATH = WORKSPACE / "artifacts/annotation_staging/phase12_2/stage3_added_shapes_mapping.json"

with open(POST_PATH, "r", encoding="utf-8") as f:
    post_data = json.load(f)
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest = json.load(f)
with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    mapping = json.load(f)

post_shapes = {s["id"]: s for s in post_data["shapes"]}
mapping_by_id = {m["new_cvat_shape_id"]: m for m in mapping}
manifest_by_cand = {m["candidate_id"]: m for m in manifest}

new_ids = sorted([s["id"] for s in post_shapes.values() if 2224 <= s["id"] <= 2329])
print(f"Total new IDs in range: {len(new_ids)}")

rows = []
for nid in new_ids:
    s = post_shapes[nid]
    m_rec = mapping_by_id[nid]
    cand_id = m_rec["candidate_id"]
    man_item = manifest_by_cand[cand_id]
    
    row = {
        "cvat_id": nid,
        "frame": s["frame"],
        "global_id": m_rec.get("global_id"),
        "filename": m_rec.get("image_filename"),
        "candidate_id": cand_id,
        "label": m_rec.get("label"),
        "label_id": s["label_id"],
        "replaced_id": m_rec.get("replaced_shape_id"),
        "vertex_count": len(s["points"]) // 2,
        "type": s["type"]
    }
    rows.append(row)

print("First 15 rows:")
for r in rows[:15]:
    print(r)

print("\nLast 15 rows:")
for r in rows[-15:]:
    print(r)

# Verify counts
leaves = [r for r in rows if r["label_id"] == 5]
lesions = [r for r in rows if r["label_id"] == 6]
print(f"\nTotal Leaves: {len(leaves)}")
print(f"Total Lesions: {len(lesions)}")
assert len(leaves) == 12
assert len(lesions) == 94
