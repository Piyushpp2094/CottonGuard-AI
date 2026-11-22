"""
scripts/analyze_retained_baseline.py
Performs exact forensic decomposition of the 867 retained shapes.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

# Load baseline snapshot
with open("artifacts/backups/cvat_job6_stage0_pre_remediation_snapshot_20261118_205937.json") as f:
    snap = json.load(f)
shapes = snap["shapes"]

# Load 303 deletions
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

retained = [s for s in shapes if s["id"] not in all_del_ids]
assert len(retained) == 867

# Let's inspect which frames the retained shapes belong to.
# Recall the frames:
# Total 50 frames:
# Dev set frames: 0 to 34 (35 frames: IDs 50 to 84).
# Holdout set frames: 35 to 49 (15 frames: IDs 85 to 99).
# Wait! Let's check which frames are in the CVAT task!
frames_all = sorted(set(s["frame"] for s in shapes))
print(f"Total frames in snapshot: {len(frames_all)} (min={min(frames_all)}, max={max(frames_all)})")

# Let's categorize frames:
# 1. Ready development frames vs Remediation development frames vs Holdout frames?
# Wait! What are the 11 ready frames, 8 remediation frames, 2 leaf-only frames, etc.?
# Let's check the frames in the 14 remediation images:
remediation_frames = set(DELETIONS_BY_FRAME.keys())
print(f"Remediation frames ({len(remediation_frames)}):", sorted(remediation_frames))

dev_frames = [f for f in frames_all if f < 35]
holdout_frames = [f for f in frames_all if f >= 35]
print(f"Dev frames ({len(dev_frames)}):", dev_frames)
print(f"Holdout frames ({len(holdout_frames)}):", holdout_frames)

# In retained shapes:
retained_remediation = [s for s in retained if s["frame"] in remediation_frames]
retained_non_remediation_dev = [s for s in retained if s["frame"] in dev_frames and s["frame"] not in remediation_frames]
retained_holdout = [s for s in retained if s["frame"] in holdout_frames]

print("\n--- RETAINED BY HIGH-LEVEL SUBSET ---")
print(f"1. Retained in Remediation Frames ({len(remediation_frames)} frames): {len(retained_remediation)}")
print(f"   Types: {Counter(s['type'] for s in retained_remediation)}")
print(f"   Labels: {Counter(s['label_id'] for s in retained_remediation)}")

print(f"2. Retained in Non-Remediation Dev Frames ({len(retained_non_remediation_dev)} frames): {len(retained_non_remediation_dev)}")
print(f"   Types: {Counter(s['type'] for s in retained_non_remediation_dev)}")
print(f"   Labels: {Counter(s['label_id'] for s in retained_non_remediation_dev)}")

print(f"3. Retained in Sealed Holdout Frames ({len(retained_holdout)} frames): {len(retained_holdout)}")
print(f"   Types: {Counter(s['type'] for s in retained_holdout)}")
print(f"   Labels: {Counter(s['label_id'] for s in retained_holdout)}")

# Total check:
assert len(retained_remediation) + len(retained_non_remediation_dev) + len(retained_holdout) == 867

# Let's check DEC-01 rectangles specifically:
dec01_rects = [s for s in retained if s["type"] == "rectangle" and s["frame"] in (26, 28)]
print(f"\nDEC-01 Rectangles in frames 26 & 28: {len(dec01_rects)}")
print(f"  Frame 26: {sum(1 for s in dec01_rects if s['frame'] == 26)}")
print(f"  Frame 28: {sum(1 for s in dec01_rects if s['frame'] == 28)}")
