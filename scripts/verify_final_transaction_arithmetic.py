"""
scripts/verify_final_transaction_arithmetic.py
Independently verifies the complete final transaction balance sheet from source records:
- Baseline CVAT Job #6 snapshot (1,170 shapes)
- Authorized 303 deletion IDs
- Precision 106 additions manifest
"""

import json
from collections import Counter
from pathlib import Path

# 1. Load Baseline
SNAPSHOT_PATH = Path("artifacts/backups/cvat_job6_stage0_pre_remediation_snapshot_20261118_205937.json")
with open(SNAPSHOT_PATH) as f:
    snap = json.load(f)
baseline_shapes = snap["shapes"]
assert len(baseline_shapes) == 1170, f"Expected 1170 baseline shapes, got {len(baseline_shapes)}"

# 2. Load Deletions
DELETIONS_BY_FRAME = {
    0:  {"leaf": [1039], "rect": [], "poly_4pt": []},
    9:  {"leaf": [1143], "rect": [], "poly_4pt": []},
    15: {"leaf": [1157], "rect": [1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187],
                         "poly_4pt": [1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172]},
    17: {"leaf": [1201], "rect": [1205, 1206, 1207], "poly_4pt": [1203, 1204]},
    18: {"leaf": [],     "rect": [1227, 1228, 1229, 1230, 1231, 1232, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1242, 1243, 1244],
                         "poly_4pt": [1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226]},
    19: {"leaf": [],     "rect": [1261, 1262, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275],
                         "poly_4pt": [1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260]},
    20: {"leaf": [1276], "rect": [], "poly_4pt": []},
    21: {"leaf": [1289], "rect": list(range(1319, 1345)), "poly_4pt": list(range(1290, 1316))},
    22: {"leaf": [],     "rect": list(range(1384, 1419)), "poly_4pt": list(range(1346, 1381))},
    23: {"leaf": [],     "rect": list(range(1441, 1462)), "poly_4pt": list(range(1420, 1441))},
    24: {"leaf": [1462], "rect": list(range(1483, 1500)), "poly_4pt": list(range(1463, 1480))},
    26: {"leaf": [1567], "rect": [], "poly_4pt": []},
    28: {"leaf": [1681], "rect": [], "poly_4pt": []},
    31: {"leaf": [1756], "rect": [], "poly_4pt": []},
}

all_del_ids = set()
for f, d in DELETIONS_BY_FRAME.items():
    all_del_ids.update(d["leaf"])
    all_del_ids.update(d["rect"])
    all_del_ids.update(d["poly_4pt"])
assert len(all_del_ids) == 303, f"Expected 303 deletions, got {len(all_del_ids)}"

# 3. Load Additions Manifest
ADDITIONS_PATH = Path("artifacts/annotation_staging/phase12_2/precision_additions_manifest.json")
with open(ADDITIONS_PATH) as f:
    adds_106 = json.load(f)
assert len(adds_106) == 106, f"Expected 106 additions, got {len(adds_106)}"

# Inspect additions
add_leaves = [a for a in adds_106 if a["annotation_label"] == "Leaf"]
add_lesions = [a for a in adds_106 if a["annotation_label"] == "Diseased_Region"]
add_rects = [a for a in adds_106 if a["type"] == "rectangle"]
add_polys = [a for a in adds_106 if a["type"] == "polygon"]

assert len(add_leaves) == 12, f"Expected 12 leaf additions, got {len(add_leaves)}"
assert len(add_lesions) == 94, f"Expected 94 lesion additions, got {len(add_lesions)}"
assert len(add_rects) == 0, f"Expected 0 rectangle additions, got {len(add_rects)}"
assert len(add_polys) == 106, f"Expected 106 polygon additions, got {len(add_polys)}"

# Inspect baseline shapes
base_by_id = {s["id"]: s for s in baseline_shapes}
assert len(base_by_id) == 1170
assert all_del_ids.issubset(base_by_id.keys()), "Some deletion IDs are not in baseline!"

del_shapes = [base_by_id[sid] for sid in all_del_ids]
del_leaves = [s for s in del_shapes if s["label_id"] == 5]
del_lesions = [s for s in del_shapes if s["label_id"] == 6]
del_rects = [s for s in del_shapes if s["type"] == "rectangle"]
del_polys = [s for s in del_shapes if s["type"] == "polygon"]

assert len(del_leaves) == 10
assert len(del_lesions) == 293
assert len(del_rects) == 150
assert len(del_polys) == 153

# Inspect retained shapes
ret_shapes = [s for s in baseline_shapes if s["id"] not in all_del_ids]
ret_leaves = [s for s in ret_shapes if s["label_id"] == 5]
ret_lesions = [s for s in ret_shapes if s["label_id"] == 6]
ret_rects = [s for s in ret_shapes if s["type"] == "rectangle"]
ret_polys = [s for s in ret_shapes if s["type"] == "polygon"]

assert len(ret_shapes) == 867
assert len(ret_leaves) == 40
assert len(ret_lesions) == 827
assert len(ret_rects) == 345
assert len(ret_polys) == 522

# Compute simulated final CVAT shapes
final_total = len(ret_shapes) + len(adds_106)
final_leaves = len(ret_leaves) + len(add_leaves)
final_lesions = len(ret_lesions) + len(add_lesions)
final_rects = len(ret_rects) + len(add_rects)
final_polys = len(ret_polys) + len(add_polys)

print("=" * 80)
print("INDEPENDENT FINAL TRANSACTION ARITHMETIC VERIFICATION")
print("=" * 80)
print(f"Baseline Shapes:    {len(baseline_shapes)} (Leaf: {sum(1 for s in baseline_shapes if s['label_id']==5)}, Dis: {sum(1 for s in baseline_shapes if s['label_id']==6)}, Rect: {sum(1 for s in baseline_shapes if s['type']=='rectangle')}, Poly: {sum(1 for s in baseline_shapes if s['type']=='polygon')})")
print(f"Deletions:        - {len(del_shapes)} (Leaf: {len(del_leaves)}, Dis: {len(del_lesions)}, Rect: {len(del_rects)}, Poly: {len(del_polys)})")
print(f"Retained:         = {len(ret_shapes)} (Leaf: {len(ret_leaves)}, Dis: {len(ret_lesions)}, Rect: {len(ret_rects)}, Poly: {len(ret_polys)})")
print(f"Additions:        + {len(adds_106)} (Leaf: {len(add_leaves)}, Dis: {len(add_lesions)}, Rect: {len(add_rects)}, Poly: {len(add_polys)})")
print("-" * 80)
print(f"Expected Final:   = {final_total} (Leaf: {final_leaves}, Dis: {final_lesions}, Rect: {final_rects}, Poly: {final_polys})")
print("=" * 80)

assert final_total == 973, f"Expected 973, got {final_total}"
assert final_leaves == 52, f"Expected 52 leaves, got {final_leaves}"
assert final_lesions == 921, f"Expected 921 lesions, got {final_lesions}"
assert final_rects == 345, f"Expected 345 rectangles, got {final_rects}"
assert final_polys == 628, f"Expected 628 polygons, got {final_polys}"
assert final_leaves + final_lesions == 973
assert final_rects + final_polys == 973

print("ALL TRANSACTION ARITHMETIC VERIFIED EXACTLY AGAINST SOURCE RECORDS.")
