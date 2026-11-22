"""
scripts/audit_stage2_4_frame31.py
Investigates Frame 31 source mapping, image filename, class label, and shape 1756.
"""

import json
import pandas as pd
from pathlib import Path

# 1. Check consolidated 100 manifest
df = pd.read_csv("artifacts/manifests/consolidated_100_annotation_manifest.csv")
print("Columns in manifest:", df.columns.tolist())

# Look for Frame 31 or Global ID 81
print("\nSearching manifest for Frame 31 / Global ID 81:")
m_gid81 = df[df["global_id"] == 81] if "global_id" in df.columns else pd.DataFrame()
print("Matches for global_id == 81:\n", m_gid81)

if "cvat_frame" in df.columns:
    m_f31 = df[df["cvat_frame"] == 31]
    print("\nMatches for cvat_frame == 31:\n", m_f31)

# Check row 31 (0-indexed) or around it
print("\nManifest row 31 (iloc[31]):\n", df.iloc[31])

# Check where RD00001.jpg and HL00070.jpg are in manifest
rd_match = df[df["filename"].str.contains("RD00001", na=False)]
print("\nRD00001.jpg in manifest:\n", rd_match)

hl_match = df[df["filename"].str.contains("HL00070", na=False)]
print("\nHL00070.jpg in manifest:\n", hl_match)

# 2. Check CVAT task data / snapshot
with open("artifacts/backups/cvat_job6_stage0_pre_remediation_snapshot_20261118_205937.json") as f:
    snap = json.load(f)

f31_shapes = [s for s in snap["shapes"] if s["frame"] == 31]
print(f"\nCVAT Job #6 snapshot shapes for frame 31: (Total: {len(f31_shapes)})")
for s in f31_shapes:
    print(f"  ID {s['id']}: type={s['type']}, label_id={s['label_id']}, frame={s['frame']}, points={s['points'][:8]}...")

# 3. Check staged candidate for Frame 31
with open("artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json") as f:
    cands = json.load(f)

f31_cands = [c for c in cands if c["frame_number"] == 31]
print(f"\nStaged candidates for frame 31: (Total: {len(f31_cands)})")
for c in f31_cands:
    print(f"  CID: {c.get('candidate_id')}, label={c.get('annotation_label')}, replaced_id={c.get('replaced_shape_id')}, img={c.get('image_filename')}")

# 4. Check images directory on disk
img_dir = Path("data/detection_consolidated_100/images")
print(f"\nDoes RD00001.jpg exist on disk? {(img_dir / 'RD00001.jpg').exists()}")
print(f"Does HL00070.jpg exist on disk? {(img_dir / 'HL00070.jpg').exists()}")
