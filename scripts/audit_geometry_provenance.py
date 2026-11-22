"""
scripts/audit_geometry_provenance.py
Calculates SHA-256 hashes, vertex counts, areas, and full provenance for revised geometries.
"""

import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest().lower()

# 1. Source images
img21_p = Path("data/detection_consolidated_100/images/LR00140.jpg")
img24_p = Path("data/detection_consolidated_100/images/LR00578.jpg")
img15_p = Path("data/detection_consolidated_100/images/LHJ00001.jpg")
img31_p = Path("data/detection_consolidated_100/images/HL00070.jpg")

print("Source Image SHA-256 Hashes:")
print(f"  Frame 15 (LHJ00001.jpg): {sha256_file(img15_p)}")
print(f"  Frame 21 (LR00140.jpg):  {sha256_file(img21_p)}")
print(f"  Frame 24 (LR00578.jpg):  {sha256_file(img24_p)}")
print(f"  Frame 31 (HL00070.jpg):  {sha256_file(img31_p)}")

# 2. Check CAND_F21_L_BLADE2
with open("artifacts/annotation_staging/phase12_2/new_secondary_leaf_candidates.json") as f:
    sec = json.load(f)

b2 = sec["CAND_F21_L_BLADE2"]
b2_pts = np.array(b2["polygon_coordinates"]).reshape(-1, 2)
b2_area = cv2.contourArea(b2_pts.astype(np.float32))
b2_bytes = json.dumps(b2["polygon_coordinates"]).encode("utf-8")
print("\nCAND_F21_L_BLADE2 Provenance:")
print(f"  Version: Authoritative Organic v2.0")
print(f"  Vertex Count: {len(b2_pts)}")
print(f"  Contour Area: {b2_area:.1f} px (Reported: {b2['area']})")
print(f"  Coordinate SHA-256: {sha256_bytes(b2_bytes)}")
print(f"  File SHA-256: {sha256_file('artifacts/annotation_staging/phase12_2/new_secondary_leaf_candidates.json')}")

# 3. Check CAND_F24_L_1462_REFINED
with open("artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json") as f:
    master = json.load(f)

f24_lf = next(c for c in master if c["candidate_id"] == "CAND_F24_L_1462_REFINED")
f24_pts = np.array(f24_lf["polygon_coordinates"]).reshape(-1, 2)
f24_area = cv2.contourArea(f24_pts.astype(np.float32))
f24_bytes = json.dumps(f24_lf["polygon_coordinates"]).encode("utf-8")

print("\nCAND_F24_L_1462_REFINED Provenance:")
print(f"  Version: Authoritative Sinus-Refined v2.0")
print(f"  Vertex Count: {len(f24_pts)}")
print(f"  Contour Area: {f24_area:.1f} px (Reported: {f24_lf['area']})")
print(f"  Coordinate SHA-256: {sha256_bytes(f24_bytes)}")

# 4. Check CAND_F15_L_SECONDARY_01
f15_sec = sec["CAND_F15_L_SECONDARY_01"]
f15_sec_pts = np.array(f15_sec["polygon_coordinates"]).reshape(-1, 2)
f15_sec_area = cv2.contourArea(f15_sec_pts.astype(np.float32))
f15_sec_bytes = json.dumps(f15_sec["polygon_coordinates"]).encode("utf-8")

print("\nCAND_F15_L_SECONDARY_01 Provenance:")
print(f"  Version: Authoritative Secondary Blade v1.0")
print(f"  Vertex Count: {len(f15_sec_pts)}")
print(f"  Contour Area: {f15_sec_area:.1f} px (Reported: {f15_sec['area']})")
print(f"  Coordinate SHA-256: {sha256_bytes(f15_sec_bytes)}")
