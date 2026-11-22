"""
scripts/reproduce_retained_decomposition.py
Reproducible calculation generating the exact disjoint shape categories for the 867 retained shapes.
"""

import json
from collections import Counter
from pathlib import Path

# Load snapshot
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

remediation_frames = set(DELETIONS_BY_FRAME.keys())
dev_non_remediation_frames = set(f for f in range(35) if f not in remediation_frames)
holdout_frames = set(range(35, 50))

assert len(remediation_frames) == 14
assert len(dev_non_remediation_frames) == 21
assert len(holdout_frames) == 15
assert len(remediation_frames | dev_non_remediation_frames | holdout_frames) == 50

# Partition all 867 retained shapes into clean, mutually exclusive, collectively exhaustive categories:
retained_shapes = [s for s in shapes if s["id"] not in all_del_ids]
assert len(retained_shapes) == 867

# Categories:
# Cat A: Preserved DEC-01 Detection Rectangles in Remediation Set (Frames 26 & 28)
cat_a = [s for s in retained_shapes if s["frame"] in (26, 28) and s["type"] == "rectangle"]

# Cat B: Retained Baseline Polygons in Remediation Set (14 frames)
cat_b = [s for s in retained_shapes if s["frame"] in remediation_frames and s["type"] == "polygon"]

# Cat C: Retained Non-Remediation Development Set Shapes (21 frames)
cat_c = [s for s in retained_shapes if s["frame"] in dev_non_remediation_frames]

# Cat D: Sealed Holdout Set Shapes (15 frames: 35-49)
cat_d = [s for s in retained_shapes if s["frame"] in holdout_frames]

categories = [
    ("A. Preserved DEC-01 Detection Rectangles (Frames 26 & 28)", cat_a),
    ("B. Retained Baseline Polygons in Remediation Set (14 frames)", cat_b),
    ("C. Retained Non-Remediation Dev Set Shapes (21 frames)", cat_c),
    ("D. Sealed Holdout Set Shapes (15 frames: 35-49)", cat_d)
]

print("=" * 95)
print(f"{'Category':<58} {'Total':<7} {'Leaf':<6} {'Dis':<6} {'Rect':<6} {'Poly':<6}")
print("=" * 95)

tot_s = 0
tot_leaf = 0
tot_dis = 0
tot_rect = 0
tot_poly = 0

seen_ids = set()

for name, cat in categories:
    t = len(cat)
    leaf = sum(1 for s in cat if s["label_id"] == 5)
    dis = sum(1 for s in cat if s["label_id"] == 6)
    rect = sum(1 for s in cat if s["type"] == "rectangle")
    poly = sum(1 for s in cat if s["type"] == "polygon")
    
    # Check disjoint
    cat_ids = set(s["id"] for s in cat)
    assert not (cat_ids & seen_ids), f"Overlapping IDs found in {name}!"
    seen_ids.update(cat_ids)
    
    print(f"{name:<58} {t:<7} {leaf:<6} {dis:<6} {rect:<6} {poly:<6}")
    tot_s += t
    tot_leaf += leaf
    tot_dis += dis
    tot_rect += rect
    tot_poly += poly

print("-" * 95)
print(f"{'TOTAL RETAINED BASELINE SHAPES':<58} {tot_s:<7} {tot_leaf:<6} {tot_dis:<6} {tot_rect:<6} {tot_poly:<6}")
print("=" * 95)

# Verification assertions
assert seen_ids == set(s["id"] for s in retained_shapes), "Mismatch in retained shape set!"
assert tot_s == 867, f"Total shapes {tot_s} != 867"
assert tot_leaf == 40, f"Leaf shapes {tot_leaf} != 40"
assert tot_dis == 827, f"Diseased shapes {tot_dis} != 827"
assert tot_rect == 345, f"Rectangles {tot_rect} != 345"
assert tot_poly == 522, f"Polygons {tot_poly} != 522"

print("\nALL VERIFICATION CRITERIA PASSED EXACTLY:")
print("1. Total equals 867: PASSED")
print("2. Leaf labels sum to 40: PASSED")
print("3. Diseased region labels sum to 827: PASSED")
print("4. Rectangle types sum to 345: PASSED")
print("5. Polygon types sum to 522: PASSED")
print("6. No shape counted twice (strictly disjoint): PASSED")
print("7. No retained shape omitted (collectively exhaustive): PASSED")
