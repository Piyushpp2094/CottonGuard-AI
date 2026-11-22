"""
scripts/inspect_shape_1337.py
Inspects baseline shape 1337 in the baseline backup vs Candidate 1337 in the staged manifest.
"""

import json
from pathlib import Path

# Load baseline snapshot
with open("artifacts/backups/cvat_job6_fresh_pre_remediation_baseline_20261118.json") as f:
    d1 = json.load(f)

shape_1337 = next((s for s in d1["shapes"] if s["id"] == 1337), None)
print("=== BASELINE SHAPE 1337 ===")
print("Found:", shape_1337 is not None)
if shape_1337:
    print("ID:", shape_1337["id"])
    print("Frame:", shape_1337["frame"])
    print("Label ID:", shape_1337["label_id"])
    print("Type:", shape_1337["type"])
    print("Points:", shape_1337["points"])
    print("Attributes:", shape_1337.get("attributes", []))

# Load Candidate 1337
with open("artifacts/annotation_staging/phase12_2/precision_additions_manifest.json") as f:
    adds = json.load(f)

cand_1337 = next((c for c in adds if c.get("replaced_shape_id") == 1337 or c.get("candidate_id") == "CAND_F21_LES_1337"), None)
print("\n=== STAGED CANDIDATE 1337 ===")
print("Found:", cand_1337 is not None)
if cand_1337:
    print("Candidate ID:", cand_1337["candidate_id"])
    print("Replaced Shape ID:", cand_1337["replaced_shape_id"])
    print("Frame:", cand_1337["frame_number"])
    print("Label ID:", cand_1337["label_id"])
    print("Type:", cand_1337["type"])
    print("Vertex count:", cand_1337["vertex_count"])
    print("Area:", cand_1337["area"])
    print("Decision:", cand_1337["adjudication_decision"])
    print("Points (first 4 coords):", cand_1337["polygon_coordinates"][:8])
