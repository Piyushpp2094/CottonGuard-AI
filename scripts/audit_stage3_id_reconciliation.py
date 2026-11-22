"""
scripts/audit_stage3_id_reconciliation.py
Strictly READ-ONLY audit to reconcile Stage 3 shape IDs between
precision_additions_manifest.json, pre-snapshot, post-snapshot, and stage3_added_shapes_mapping.json.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")

PRE_SNAPSHOT_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage3_pre_addition_snapshot.json"
POST_SNAPSHOT_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage3_post_addition_snapshot.json"
MANIFEST_PATH = WORKSPACE / "artifacts/annotation_staging/phase12_2/precision_additions_manifest.json"
MAPPING_PATH = WORKSPACE / "artifacts/annotation_staging/phase12_2/stage3_added_shapes_mapping.json"

def main():
    print("=== STAGE 3 READ-ONLY ID RECONCILIATION AUDIT ===")

    # 1. Load snapshots
    with open(PRE_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        pre_data = json.load(f)
    with open(POST_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        post_data = json.load(f)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    pre_shapes = pre_data.get("shapes", [])
    post_shapes = post_data.get("shapes", [])
    print(f"Pre-Stage-3 Shape Count: {len(pre_shapes)}")
    print(f"Post-Stage-3 Shape Count: {len(post_shapes)}")
    print(f"Difference: {len(post_shapes) - len(pre_shapes)}")

    pre_ids = {s["id"] for s in pre_shapes}
    post_ids = {s["id"] for s in post_shapes}

    assert len(pre_ids) == len(pre_shapes), "Duplicate IDs in pre-snapshot"
    assert len(post_ids) == len(post_shapes), "Duplicate IDs in post-snapshot"

    new_ids = post_ids - pre_ids
    deleted_ids = pre_ids - post_ids

    print(f"Deleted IDs Count: {len(deleted_ids)}")
    print(f"New IDs Count: {len(new_ids)}")
    assert len(deleted_ids) == 0, f"Deletions detected in Stage 3: {deleted_ids}"
    assert len(new_ids) == 106, f"Expected 106 new IDs, got {len(new_ids)}"

    new_ids_sorted = sorted(list(new_ids))
    print(f"Min New ID: {new_ids_sorted[0]}, Max New ID: {new_ids_sorted[-1]}")
    expected_range = list(range(2224, 2330))
    print(f"Are new IDs exactly range(2224, 2330)? {new_ids_sorted == expected_range}")
    if new_ids_sorted != expected_range:
        missing = set(expected_range) - set(new_ids_sorted)
        extra = set(new_ids_sorted) - set(expected_range)
        print(f"Missing from 2224..2329: {missing}")
        print(f"Extra outside 2224..2329: {extra}")

    # Index new shapes from post_data
    post_shapes_by_id = {s["id"]: s for s in post_shapes}
    new_shapes = [post_shapes_by_id[sid] for sid in new_ids_sorted]

    # Inspect ID 2245 specifically
    s_2245 = post_shapes_by_id.get(2245)
    print("\n--- Inspecting Shape ID 2245 in post-snapshot ---")
    if s_2245:
        print(f"ID 2245 details:")
        print(f"  Frame: {s_2245.get('frame')}")
        print(f"  Label ID: {s_2245.get('label_id')}")
        print(f"  Type: {s_2245.get('type')}")
        print(f"  Points Count: {len(s_2245.get('points', [])) // 2} vertices ({len(s_2245.get('points', []))} coordinates)")
        print(f"  Points Sample: {s_2245.get('points', [])[:8]}...")
    else:
        print("ID 2245 NOT FOUND in post_shapes!")

    # Check mapping entry for 2245
    m_2245 = [x for x in mapping if x.get("new_cvat_shape_id") == 2245]
    print(f"Mapping entries for 2245: {len(m_2245)}")
    if m_2245:
        print(f"  Candidate ID: {m_2245[0].get('candidate_id')}")
        print(f"  Frame: {m_2245[0].get('frame')}")
        print(f"  Label: {m_2245[0].get('label')} (ID {m_2245[0].get('label_id')})")
        print(f"  Replaced Shape ID: {m_2245[0].get('replaced_shape_id')}")

    # Inspect Frame 15 shapes and IDs
    f15_new = [s for s in new_shapes if s.get("frame") == 15]
    print(f"\n--- Frame 15 New Shapes ({len(f15_new)}) ---")
    for s in f15_new:
        print(f"  ID: {s['id']}, Label: {s['label_id']}, Type: {s['type']}, Vertices: {len(s['points']) // 2}")

    # Inspect Frame 21 shapes and IDs
    f21_new = [s for s in new_shapes if s.get("frame") == 21]
    print(f"\n--- Frame 21 New Shapes ({len(f21_new)}) ---")
    for s in f21_new:
        print(f"  ID: {s['id']}, Label: {s['label_id']}, Type: {s['type']}, Vertices: {len(s['points']) // 2}")

    # Per frame counts of new shapes
    print("\n--- Per-Frame Breakdown of New Shapes ---")
    by_frame = defaultdict(list)
    for s in new_shapes:
        by_frame[s["frame"]].append(s)

    for f_idx in sorted(by_frame.keys()):
        shps = by_frame[f_idx]
        l_cnt = sum(1 for s in shps if s["label_id"] == 5)
        d_cnt = sum(1 for s in shps if s["label_id"] == 6)
        ids = [s["id"] for s in shps]
        print(f"Frame {f_idx:02d}: Total={len(shps)} (Leaves={l_cnt}, Lesions={d_cnt}), IDs={ids}")

if __name__ == "__main__":
    main()
