"""
scripts/deep_inspect_stage4_deletions.py
Comprehensive analysis of the 303 scheduled deletion shapes against:
- Authoritative baseline snapshot
- Post-Stage-3 snapshot
- Live CVAT annotations (read-only)
"""

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")
sys.path.insert(0, str(WORKSPACE))

from scripts.execute_gate4_stage0_verification import DELETIONS_BY_FRAME

POST_SNAPSHOT_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage3_post_addition_snapshot.json"
PRE_SNAPSHOT_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage3_pre_addition_snapshot.json"
BASELINE_PATH = WORKSPACE / "artifacts/backups/cvat_job6_fresh_pre_remediation_baseline_20261118.json"
ADDITIONS_MANIFEST_PATH = WORKSPACE / "artifacts/annotation_staging/phase12_2/precision_additions_manifest.json"
MAPPING_PATH = WORKSPACE / "artifacts/annotation_staging/phase12_2/stage3_added_shapes_mapping.json"

FRAME_METADATA = {
    0:  {"global_id": 50, "filename": "BBC00001.jpg", "disease": "Bacterial Blight Cotton"},
    9:  {"global_id": 59, "filename": "CV00431.jpg",  "disease": "Cotton Vein Enation Virus"},
    15: {"global_id": 65, "filename": "LHJ00001.jpg", "disease": "Leaf Hopper and Jassids"},
    17: {"global_id": 67, "filename": "LHJ00116.jpg", "disease": "Leaf Hopper and Jassids"},
    18: {"global_id": 68, "filename": "LHJ00172.jpg", "disease": "Leaf Hopper and Jassids"},
    19: {"global_id": 69, "filename": "LHJ00225.jpg", "disease": "Leaf Hopper and Jassids"},
    20: {"global_id": 70, "filename": "LR00001.jpg",  "disease": "Leaf Reddening"},
    21: {"global_id": 71, "filename": "LR00140.jpg",  "disease": "Leaf Reddening"},
    22: {"global_id": 72, "filename": "LR00286.jpg",  "disease": "Leaf Reddening"},
    23: {"global_id": 73, "filename": "LR00440.jpg",  "disease": "Leaf Reddening"},
    24: {"global_id": 74, "filename": "LR00578.jpg",  "disease": "Leaf Reddening"},
    26: {"global_id": 76, "filename": "LV00028.jpg",  "disease": "Leaf Variegation"},
    28: {"global_id": 78, "filename": "LV00087.jpg",  "disease": "Leaf Variegation"},
    31: {"global_id": 81, "filename": "HL00070.jpg",  "disease": "Healthy Leaf"},
}

def calc_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()

def main():
    print("=== STAGE 4 PRE-DELETION VALIDATION AUDIT ===")

    # 1. Inspect Hashes
    print(f"Baseline Snapshot SHA-256:     {calc_sha256(BASELINE_PATH)}")
    print(f"Pre-Stage-3 Snapshot SHA-256:  {calc_sha256(PRE_SNAPSHOT_PATH)}")
    print(f"Post-Stage-3 Snapshot SHA-256: {calc_sha256(POST_SNAPSHOT_PATH)}")
    print(f"Additions Manifest SHA-256:    {calc_sha256(ADDITIONS_MANIFEST_PATH)}")

    # 2. Load Snapshots
    with open(POST_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        post_data = json.load(f)
    post_shapes = {s["id"]: s for s in post_data["shapes"]}
    print(f"\nPost-Stage-3 Total Shapes: {len(post_shapes)} (Expected: 1,276)")
    assert len(post_shapes) == 1276

    # 3. Analyze Deletions roster in DELETIONS_BY_FRAME
    leaf_del_ids = []
    rect_del_ids = []
    poly_del_ids = []
    all_del_ids_list = []

    for f_idx, d_dict in sorted(DELETIONS_BY_FRAME.items()):
        leaf_del_ids.extend(d_dict["leaf"])
        rect_del_ids.extend(d_dict["rect"])
        poly_del_ids.extend(d_dict["poly_4pt"])
        all_del_ids_list.extend(d_dict["leaf"] + d_dict["rect"] + d_dict["poly_4pt"])

    del_set = set(all_del_ids_list)
    print(f"\nTotal Deletion IDs in roster: {len(all_del_ids_list)}")
    print(f"Unique Deletion IDs in roster: {len(del_set)}")
    assert len(all_del_ids_list) == 303
    assert len(del_set) == 303
    print("  -> PASS: Exactly 303 unique deletion IDs in roster.")

    # Breakdown by dictionary keys
    print(f"\nRoster Key Breakdown:")
    print(f"  'leaf':     {len(leaf_del_ids)}")
    print(f"  'rect':     {len(rect_del_ids)}")
    print(f"  'poly_4pt': {len(poly_del_ids)}")
    print(f"  Total:      {len(leaf_del_ids) + len(rect_del_ids) + len(poly_del_ids)}")

    # 4. Now inspect the ACTUAL live shape metadata for each of the 303 IDs!
    del_shapes = []
    missing_from_post = []
    for sid in all_del_ids_list:
        s = post_shapes.get(sid)
        if s is None:
            missing_from_post.append(sid)
        else:
            del_shapes.append(s)

    print(f"\nActual Shape Presence in Live/Post-Snapshot: {len(del_shapes)}/303")
    assert len(missing_from_post) == 0, f"Missing deletion IDs: {missing_from_post}"
    print("  -> PASS: All 303 deletion shapes exist in the live snapshot.")

    # 5. Extract actual metadata for each deletion shape
    del_labels = Counter(s.get("label_id") for s in del_shapes)
    del_types = Counter(s.get("type") for s in del_shapes)
    del_frames = Counter(s.get("frame") for s in del_shapes)

    print(f"\n--- ACTUAL SHAPE METADATA BREAKDOWN (From live CVAT shape objects) ---")
    print(f"By Label ID:  {dict(del_labels)}")
    print(f"By Geometry:  {dict(del_types)}")
    print(f"By Frame:     {dict(sorted(del_frames.items()))}")

    # Inspect polygons vertex counts
    poly_shapes = [s for s in del_shapes if s.get("type") == "polygon"]
    poly_v_counts = Counter(len(s.get("points", [])) // 2 for s in poly_shapes)
    print(f"Polygon vertex counts: {dict(poly_v_counts)}")

    # Inspect rectangles points
    rect_shapes = [s for s in del_shapes if s.get("type") == "rectangle"]
    rect_pts_counts = Counter(len(s.get("points", [])) for s in rect_shapes)
    print(f"Rectangle points lengths: {dict(rect_pts_counts)}")

    # Check leaves among deletions
    leaf_del_shapes = [s for s in del_shapes if s.get("label_id") == 5]
    print(f"\nLeaf Deletion Shapes ({len(leaf_del_shapes)}):")
    for s in leaf_del_shapes:
        v_cnt = len(s.get("points", [])) // 2 if s.get("type") == "polygon" else len(s.get("points", []))
        print(f"  ID {s['id']}: Frame {s['frame']}, Type={s['type']}, Points/Vertices={v_cnt}, Points={s.get('points')}")

    # Check lesions among deletions
    lesion_del_shapes = [s for s in del_shapes if s.get("label_id") == 6]
    lesion_del_types = Counter(s.get("type") for s in lesion_del_shapes)
    print(f"\nLesion Deletion Shapes ({len(lesion_del_shapes)}):")
    print(f"  Types: {dict(lesion_del_types)}")

    # Check cross-tabulation of Label x Type
    cross_tab = Counter((s.get("label_id"), s.get("type")) for s in del_shapes)
    print(f"\nCross-tabulation (Label ID, Geometry Type):")
    for (lid, gtype), cnt in sorted(cross_tab.items()):
        lname = "Leaf" if lid == 5 else "Diseased_Region"
        print(f"  {lname} (id {lid}) + {gtype}: {cnt}")

    # Check the Discrepancy between Stage 2.6 and Stage 3:
    print(f"\n--- DISCREPANCY FORENSICS ---")
    print("Stage 2.6 description:")
    print("  - 10 invalid Leaf pseudo-masks (polygons)")
    print("  - 150 lesion rectangles")
    print("  - 143 four-point polygon replicas")
    print("Stage 3 report text:")
    print("  - 10 bounding-box leaves")
    print("  - 140 detection rectangles")
    print("  - 153 synthetic four-point polygons")
    print(f"Actual observed:")
    print(f"  - Leaves: {len(leaf_del_shapes)} (Type: {Counter(s['type'] for s in leaf_del_shapes)})")
    print(f"  - Rectangles: {len(rect_shapes)} (Labels: {Counter(s['label_id'] for s in rect_shapes)})")
    print(f"  - Polygons: {len(poly_shapes)} (Labels: {Counter(s['label_id'] for s in poly_shapes)})")

if __name__ == "__main__":
    main()
