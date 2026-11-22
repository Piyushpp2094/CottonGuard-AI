"""
scripts/verify_transaction_balance_sheet.py
Verifies the complete transaction balance sheet for Gate 4 Stage 2.3 Preflight.
"""

import json

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

all_del_ids = []
for f, d in DELETIONS_BY_FRAME.items():
    all_del_ids.extend(d["leaf"])
    all_del_ids.extend(d["rect"])
    all_del_ids.extend(d["poly_4pt"])

del_set = set(all_del_ids)
print(f"Total deletion count: {len(all_del_ids)}, Unique: {len(del_set)}")
assert len(all_del_ids) == 303
assert len(del_set) == 303

with open("artifacts/backups/cvat_job6_stage0_pre_remediation_snapshot_20261118_205937.json") as f:
    snap = json.load(f)
shapes = snap["shapes"]
snap_ids = {s["id"] for s in shapes}

# Verify all exist
missing = del_set - snap_ids
print(f"Missing deletion IDs in snapshot: {len(missing)}")
assert len(missing) == 0

# Check retained rectangles in the 14 remediation frames (Frames 26 & 28)
dev_14_frames = set(DELETIONS_BY_FRAME.keys())
retained_rects_dev14 = [s for s in shapes if s["frame"] in dev_14_frames and s["type"] == "rectangle" and s["id"] not in del_set]
print(f"Retained rectangles in 14 remediation frames: {len(retained_rects_dev14)}")
assert len(retained_rects_dev14) == 37, f"Expected 37 retained rectangles, got {len(retained_rects_dev14)}"

r26 = [s for s in retained_rects_dev14 if s["frame"] == 26]
r28 = [s for s in retained_rects_dev14 if s["frame"] == 28]
print(f"  Frame 26 (LV00028.jpg, GID 76): {len(r26)} detection rectangles: {[s['id'] for s in r26]}")
print(f"  Frame 28 (LV00087.jpg, GID 78): {len(r28)} detection rectangles: {[s['id'] for s in r28]}")

# Check holdout frames (35-49)
holdout_shapes = [s for s in shapes if 35 <= s["frame"] <= 49]
print(f"Holdout shapes in baseline: {len(holdout_shapes)}")
assert len(holdout_shapes) == 449
holdout_del_overlap = del_set.intersection({s["id"] for s in holdout_shapes})
print(f"Holdout shapes in deletions: {len(holdout_del_overlap)}")
assert len(holdout_del_overlap) == 0

# Reconciled math
baseline_total = len(shapes)
deletions = len(del_set)
retained_baseline = baseline_total - deletions

with open("artifacts/annotation_staging/phase12_2/precision_116_additions_manifest.json") as f:
    adds = json.load(f)
additions = len(adds)
leaf_adds = sum(1 for a in adds if a["annotation_label"] == "Leaf")
lesion_adds = sum(1 for a in adds if a["annotation_label"] == "Diseased_Region")
final_total = retained_baseline + additions

print("\n================ TRANSACTION BALANCE SHEET ================")
print(f"Live Baseline Shapes:               {baseline_total}")
print(f"Authorized Deletions:             - {deletions}")
print(f"Baseline Retained:                = {retained_baseline}")
print(f"  - Retained Dev-14 Detection Rectangles: {len(retained_rects_dev14)}")
print(f"  - Retained Other Shapes in Baseline:    {retained_baseline - len(retained_rects_dev14)}")
print(f"Precision Additions:              + {additions}")
print(f"  - Leaf Polygons:                    {leaf_adds}")
print(f"  - Lesion Polygons:                  {lesion_adds}")
print(f"Projected Final CVAT Shapes:      = {final_total}")
print("===========================================================")
assert final_total == 983
print("BALANCE SHEET AUDIT PASSED: Exact 983 projected final shapes!")
