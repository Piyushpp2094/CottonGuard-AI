"""
scripts/reconcile_label_counts.py
Reconciles shape counts by label_id, shape type, and U-Net class mapping.
"""

import json
from collections import Counter
from pathlib import Path

# Load baseline snapshot
with open("artifacts/backups/cvat_job6_stage0_pre_remediation_snapshot_20261118_205937.json") as f:
    snap = json.load(f)
shapes = snap["shapes"]

# 303 Deletion IDs
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
assert len(all_del_ids) == 303

# 1. Baseline Breakdown
shape_by_id = {s["id"]: s for s in shapes}
del_shapes = [shape_by_id[sid] for sid in all_del_ids]
retained_shapes = [s for s in shapes if s["id"] not in all_del_ids]

print("=" * 80)
print("LABEL AND SHAPE-TYPE INVENTORY RECONCILIATION")
print("=" * 80)

print("\n1. BASELINE CVAT JOB #6 (1,170 shapes):")
print("  By Shape Type:", Counter(s["type"] for s in shapes))
print("  By Label ID:  ", Counter(s["label_id"] for s in shapes), "(5=Leaf, 6=Diseased_Region)")

print("\n2. AUTHORIZED DELETIONS (303 shapes):")
print("  By Shape Type:", Counter(s["type"] for s in del_shapes))
print("  By Label ID:  ", Counter(s["label_id"] for s in del_shapes), "(5=Leaf, 6=Diseased_Region)")

print("\n3. RETAINED BASELINE SHAPES (867 shapes):")
print("  By Shape Type:", Counter(s["type"] for s in retained_shapes))
print("  By Label ID:  ", Counter(s["label_id"] for s in retained_shapes), "(5=Leaf, 6=Diseased_Region)")

# 4. Check Provisional 114 Addition Manifest (Stage 2.4 / 981 count)
with open("artifacts/annotation_staging/phase12_2/precision_additions_manifest.json") as f:
    adds_114 = json.load(f)

print("\n4. PROVISIONAL 114 ADDITIONS (Stage 2.4):")
print("  By Shape Type:", Counter(a["type"] for a in adds_114))
print("  By Label ID:  ", Counter(a["label_id"] for a in adds_114), "(5=Leaf, 6=Diseased_Region)")

leaf_981 = Counter(s["label_id"] for s in retained_shapes)[5] + Counter(a["label_id"] for a in adds_114)[5]
dis_981 = Counter(s["label_id"] for s in retained_shapes)[6] + Counter(a["label_id"] for a in adds_114)[6]
rect_981 = Counter(s["type"] for s in retained_shapes)["rectangle"]
poly_981 = Counter(s["type"] for s in retained_shapes)["polygon"] + len(adds_114)

print(f"\nPROVISIONAL 981 TOTALS:")
print(f"  Leaf Shapes (label_id=5):           {leaf_981} (Expected: 52)")
print(f"  Diseased Region Shapes (label_id=6): {dis_981} (Expected: 929)")
print(f"  Rectangles:                          {rect_981} (Expected: 345)")
print(f"  Polygons:                            {poly_981} (Expected: 636)")
print(f"  Total Shapes:                        {rect_981 + poly_981} (Expected: 981)")
