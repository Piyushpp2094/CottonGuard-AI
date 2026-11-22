"""
scripts/verify_manifest_to_live_mapping.py
Comprehensive cross-verification between:
1. precision_additions_manifest.json (106 entries)
2. stage3_added_shapes_mapping.json (106 entries)
3. cvat_job6_stage3_post_addition_snapshot.json (1,276 shapes)
4. cvat_job6_stage3_pre_addition_snapshot.json (1,170 shapes)
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")
sys.path.insert(0, str(WORKSPACE))

PRE_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage3_pre_addition_snapshot.json"
POST_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage3_post_addition_snapshot.json"
MANIFEST_PATH = WORKSPACE / "artifacts/annotation_staging/phase12_2/precision_additions_manifest.json"
MAPPING_PATH = WORKSPACE / "artifacts/annotation_staging/phase12_2/stage3_added_shapes_mapping.json"

def main():
    with open(PRE_PATH, "r", encoding="utf-8") as f:
        pre_data = json.load(f)
    with open(POST_PATH, "r", encoding="utf-8") as f:
        post_data = json.load(f)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    pre_shapes = {s["id"]: s for s in pre_data["shapes"]}
    post_shapes = {s["id"]: s for s in post_data["shapes"]}

    print(f"Total Pre Shapes: {len(pre_shapes)}")
    print(f"Total Post Shapes: {len(post_shapes)}")
    print(f"Total Manifest Items: {len(manifest)}")
    print(f"Total Mapping Items: {len(mapping)}")

    # 1. Check all 106 new IDs
    new_ids = sorted(list(set(post_shapes.keys()) - set(pre_shapes.keys())))
    print(f"\nExact count of new IDs: {len(new_ids)}")
    assert len(new_ids) == 106
    assert new_ids == list(range(2224, 2330))
    print("New IDs are strictly contiguous from 2224 to 2329 inclusive.")

    # 2. Check 1,170 baseline shapes unchanged
    unaltered_shapes = 0
    altered_shapes = []
    for sid, s_pre in pre_shapes.items():
        s_post = post_shapes.get(sid)
        if not s_post:
            altered_shapes.append((sid, "missing"))
            continue
        # check fields
        if (s_pre.get("type") != s_post.get("type") or
            s_pre.get("frame") != s_post.get("frame") or
            s_pre.get("label_id") != s_post.get("label_id") or
            s_pre.get("points") != s_post.get("points")):
            altered_shapes.append((sid, "modified"))
        else:
            unaltered_shapes += 1

    print(f"\nBaseline shapes preserved exactly: {unaltered_shapes}/1170")
    assert unaltered_shapes == 1170, f"Baseline shapes modified or missing: {altered_shapes}"

    # 3. Check 303 scheduled deletions in post
    from scripts.execute_gate4_stage0_verification import DELETIONS_BY_FRAME
    all_del_ids = set()
    for f_idx, d_dict in DELETIONS_BY_FRAME.items():
        all_del_ids.update(d_dict["leaf"])
        all_del_ids.update(d_dict["rect"])
        all_del_ids.update(d_dict["poly_4pt"])
    assert len(all_del_ids) == 303
    del_in_post = {sid for sid in all_del_ids if sid in post_shapes}
    print(f"Scheduled 303 deletion shapes present in post-snapshot: {len(del_in_post)}/303")
    assert len(del_in_post) == 303

    # 4. Check 37 DEC-01 rectangles in post
    f26_rects = [s for s in post_shapes.values() if s.get("frame") == 26 and s.get("type") == "rectangle"]
    f28_rects = [s for s in post_shapes.values() if s.get("frame") == 28 and s.get("type") == "rectangle"]
    print(f"DEC-01 Rectangles in post-snapshot: Frame 26={len(f26_rects)}, Frame 28={len(f28_rects)}, Total={len(f26_rects)+len(f28_rects)}")
    assert len(f26_rects) == 14
    assert len(f28_rects) == 23

    # 5. Check holdout shapes (Frames 35-49)
    holdout_shapes = [s for s in post_shapes.values() if 35 <= s.get("frame", -1) <= 49]
    print(f"Holdout Shapes in post-snapshot: {len(holdout_shapes)} (Expected: 449)")
    assert len(holdout_shapes) == 449

    # 6. Detailed manifest vs live comparison
    # Let's map each new shape to manifest item
    manifest_by_cand = {m["candidate_id"]: m for m in manifest}
    mapping_by_id = {m["new_cvat_shape_id"]: m for m in mapping}
    mapping_by_cand = {m["candidate_id"]: m for m in mapping}

    print("\n--- Detailed Audit of All 106 New Shapes ---")
    leaf_ids = []
    lesion_ids = []
    frame_shape_map = defaultdict(lambda: {"leaves": [], "lesions": []})

    for nid in new_ids:
        s = post_shapes[nid]
        m_rec = mapping_by_id.get(nid)
        assert m_rec is not None, f"No mapping record for new ID {nid}"
        cand_id = m_rec["candidate_id"]
        man_item = manifest_by_cand.get(cand_id)
        assert man_item is not None, f"No manifest item for candidate {cand_id}"

        # verify consistency between live shape, mapping, and manifest
        assert s["frame"] == m_rec["frame"] == man_item["frame_number"], f"Frame mismatch on ID {nid}"
        assert s["label_id"] == m_rec["label_id"] == man_item["label_id"], f"Label ID mismatch on ID {nid}"
        assert s["type"] == m_rec["type"] == man_item["type"] == "polygon", f"Type mismatch on ID {nid}"
        pts_live = [round(p, 1) for p in s["points"]]
        pts_man = [round(float(p), 1) for p in man_item["polygon_coordinates"]]
        assert pts_live == pts_man, f"Coordinates mismatch on ID {nid}"

        f_num = s["frame"]
        if s["label_id"] == 5:
            leaf_ids.append(nid)
            frame_shape_map[f_num]["leaves"].append(nid)
        elif s["label_id"] == 6:
            lesion_ids.append(nid)
            frame_shape_map[f_num]["lesions"].append(nid)
        else:
            raise ValueError(f"Unexpected label_id {s['label_id']} on ID {nid}")

    print(f"Total Leaves (label_id=5): {len(leaf_ids)} (Expected: 12)")
    print(f"Total Lesions (label_id=6): {len(lesion_ids)} (Expected: 94)")
    assert len(leaf_ids) == 12
    assert len(lesion_ids) == 94

    print(f"\nAll 12 Leaf IDs: {sorted(leaf_ids)}")
    print(f"All 94 Lesion IDs: {sorted(lesion_ids)}")

    print("\n--- Correct Per-Frame Breakdown Table ---")
    print(f"{'Frame':<6} | {'Leaves Added':<14} | {'Lesions Added':<14} | {'Total':<6} | {'Leaf IDs':<20} | {'Lesion IDs Range / List'}")
    print("-" * 105)
    for f_idx in sorted(frame_shape_map.keys()):
        leaves = frame_shape_map[f_idx]["leaves"]
        lesions = frame_shape_map[f_idx]["lesions"]
        tot = len(leaves) + len(lesions)
        leaves_str = ", ".join(map(str, sorted(leaves))) if leaves else "-"
        if len(lesions) > 2 and sorted(lesions) == list(range(min(lesions), max(lesions) + 1)):
            lesions_str = f"{min(lesions)}–{max(lesions)} ({len(lesions)})"
        elif lesions:
            lesions_str = ", ".join(map(str, sorted(lesions)))
        else:
            lesions_str = "-"
        print(f"{f_idx:<6} | {len(leaves):<14} | {len(lesions):<14} | {tot:<6} | {leaves_str:<20} | {lesions_str}")
    print("-" * 105)
    print(f"{'TOTAL':<6} | {len(leaf_ids):<14} | {len(lesion_ids):<14} | {len(new_ids):<6} | {'12 IDs':<20} | 94 IDs")

    # Specifically analyze Frame 15 and Frame 21
    print("\n--- Frame 15 Forensic Analysis ---")
    f15_leaves = frame_shape_map[15]["leaves"]
    f15_lesions = frame_shape_map[15]["lesions"]
    print(f"Frame 15 Leaves ({len(f15_leaves)}):")
    for lid in f15_leaves:
        m = mapping_by_id[lid]
        print(f"  ID {lid}: Cand={m['candidate_id']}, Replaced={m['replaced_shape_id']}, Label={m['label']}")
    print(f"Frame 15 Lesions ({len(f15_lesions)}):")
    for lid in f15_lesions:
        m = mapping_by_id[lid]
        print(f"  ID {lid}: Cand={m['candidate_id']}, Replaced={m['replaced_shape_id']}, Label={m['label']}")

    print("\n--- Manifest Ordering & ID Allocation ---")
    for i, m_item in enumerate(manifest[:25]):
        cand_id = m_item["candidate_id"]
        cvat_id = mapping_by_cand[cand_id]["new_cvat_shape_id"]
        print(f"Index {i:03d}: Cand={cand_id:<26} Frame={m_item['frame_number']:02d} Label={m_item['annotation_label']:<15} -> CVAT ID={cvat_id}")
    f21_leaves = frame_shape_map[21]["leaves"]
    f21_lesions = frame_shape_map[21]["lesions"]
    print(f"Frame 21 Leaves ({len(f21_leaves)}):")
    for lid in f21_leaves:
        m = mapping_by_id[lid]
        print(f"  ID {lid}: Cand={m['candidate_id']}, Replaced={m['replaced_shape_id']}, Label={m['label']}")
    print(f"Frame 21 Lesions ({len(f21_lesions)}):")
    for lid in f21_lesions:
        m = mapping_by_id[lid]
        print(f"  ID {lid}: Cand={m['candidate_id']}, Replaced={m['replaced_shape_id']}, Label={m['label']}")

if __name__ == "__main__":
    main()
