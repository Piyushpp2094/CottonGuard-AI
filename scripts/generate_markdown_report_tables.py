"""
scripts/generate_markdown_report_tables.py
Generates markdown table strings for the reconciliation report.
"""

import json
import sys
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

with open("scratch/all_106_table.md", "w", encoding="utf-8") as out:
    out.write("| # | CVAT ID | Frame | Global ID | Filename | Candidate ID | Label | Label ID | Replaced ID | Vertices | Type |\n")
    out.write("| :---: | :---: | :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
    for idx, nid in enumerate(new_ids, start=1):
        s = post_shapes[nid]
        m = mapping_by_id[nid]
        cand_id = m["candidate_id"]
        man = manifest_by_cand[cand_id]
        rep = str(m.get("replaced_shape_id")) if m.get("replaced_shape_id") is not None else "—"
        out.write(f"| {idx} | **{nid}** | {s['frame']:02d} | {m.get('global_id')} | `{m.get('image_filename')}` | `{cand_id}` | {m.get('label')} | {s['label_id']} | {rep} | {len(s['points']) // 2} | {s['type']} |\n")

print(f"Written scratch/all_106_table.md ({len(new_ids)} rows)")
