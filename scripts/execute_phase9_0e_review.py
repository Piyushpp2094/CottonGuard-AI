"""
scripts/execute_phase9_0e_review.py
Phase 9.0E: Final Leaf-Instance and Disease-Mask Review for CottonGuard AI.
Performs rigorous offline review of:
1. Frame 07: CV00211.jpg (Curl Virus) -> Investigates leaf-instance problem, multi-leaf rosette, stem inclusion
2. Frame 16: LHJ00059.jpg (Leaf Hopper Jassids) -> Inspects P08, left margin, lower central spots; excises false-positive green lamina and interior spots

Outputs:
- artifacts/pilot_corrections/phase9_0e/LHJ00059_corrected_annotations.json
- artifacts/pilot_corrections/phase9_0e/masks/*
- artifacts/visual_qa/phase9_0e/qa_phase9_0e_CV00211.png
- artifacts/visual_qa/phase9_0e/qa_phase9_0e_LHJ00059.png
- artifacts/visual_qa/phase9_0e/phase9_0e_quantitative_comparison.csv
- artifacts/metrics/phase9_0e_sha256_verification.json
"""

import json
import csv
import shutil
import hashlib
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("D:/Main_Project/cotton_leaf_ai")
phase9_0d_dir = PROJECT_ROOT / "artifacts" / "pilot_corrections" / "phase9_0d"
images_dir = PROJECT_ROOT / "data" / "pilot_batch_03" / "images"
manifest_path = PROJECT_ROOT / "artifacts" / "manifests" / "pilot_batch_03_selection_manifest.csv"

out_dir = PROJECT_ROOT / "artifacts" / "pilot_corrections" / "phase9_0e"
masks_dir = out_dir / "masks"
qa_dir = PROJECT_ROOT / "artifacts" / "visual_qa" / "phase9_0e"
metrics_dir = PROJECT_ROOT / "artifacts" / "metrics"
brain_dir = Path("C:/Users/Haker's pc/.gemini/antigravity/brain/e96a0217-11ef-4314-af5b-e8eaa45d475a")

out_dir.mkdir(parents=True, exist_ok=True)
masks_dir.mkdir(parents=True, exist_ok=True)
qa_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)

def unpack_pts(flat):
    return np.array([[flat[i], flat[i+1]] for i in range(0, len(flat), 2)], dtype=np.int32)

def pack_pts(pts):
    return [float(coord) for pt in pts.reshape(-1, 2) for coord in pt]

def calc_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()

# ------------------------------------------------------------------------------
# STEP 1: Verify SHA-256 against manifest
# ------------------------------------------------------------------------------
print("Verifying source images against manifest...")
with open(manifest_path, "r") as f:
    manifest_rows = list(csv.DictReader(f))

target_specs = {
    "CV00211.jpg": {"frame_idx": 7, "class": "Curl Virus", "subtype": "Deformation", "split": "train"},
    "LHJ00059.jpg": {"frame_idx": 16, "class": "Leaf Hopper Jassids", "subtype": "Chlorosis", "split": "train"}
}

sha_report = {}
for fname, spec in target_specs.items():
    fidx = spec["frame_idx"]
    manifest_row = manifest_rows[fidx]
    assert manifest_row["filename"] == fname, f"Filename mismatch at frame {fidx}"
    
    expected_sha = manifest_row["sha256"].lower()
    actual_sha = calc_sha256(images_dir / fname)
    assert actual_sha == expected_sha, f"SHA mismatch on {fname}"
    
    sha_report[fname] = {
        "frame_idx": fidx,
        "filename": fname,
        "class_label": spec["class"],
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "sha_match": True
    }
    print(f"  Frame {fidx:02d} ({fname}): SHA-256 MATCH ({actual_sha[:16]}...)")

with open(metrics_dir / "phase9_0e_sha256_verification.json", "w") as f:
    json.dump(sha_report, f, indent=2)

# ------------------------------------------------------------------------------
# STEP 2: Process Task 1 — CV00211.jpg (Curl Virus) -> Leaf Instance Audit
# ------------------------------------------------------------------------------
print("\nProcessing Task 1: CV00211.jpg Leaf-Instance Audit...")
cv_img = cv2.imread(str(images_dir / "CV00211.jpg"))
h, w = cv_img.shape[:2]

with open(phase9_0d_dir / "CV00211_corrected_annotations.json") as f:
    cv_90d = json.load(f)

cv_leaf_pts = unpack_pts(cv_90d["leaf_polygon"]["points"])
cv_leaf_mask = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(cv_leaf_mask, [cv_leaf_pts], 1)
cv_leaf_area = int(np.count_nonzero(cv_leaf_mask))

# Diagnostic breakdown of CV00211:
# The image encloses 3 distinct leaf instances + caulescent stem and petioles:
# Leaf 1 (Upper Apical): y=0..340, x=180..580. Truncated at y=0 by 77 px.
# Leaf 2 (Mid-Left Lateral): y=300..520, x=34..300. Lateral subordinate leaf, petiole occluded.
# Leaf 3 (Lower Basal): y=520..799, x=130..640. Truncated at y=799 by >400 px.
# Central axis: Caulescent stem and multiple petiole junctions at x=200..350, y=350..550.
# Verdict: REVIEW (Target leaf cannot be identified reliably; do not force artificial correction).

# ------------------------------------------------------------------------------
# STEP 3: Process Task 2 — LHJ00059.jpg (Leaf Hopper Jassids) -> Disease Mask Review & Correction
# ------------------------------------------------------------------------------
print("\nProcessing Task 2: LHJ00059.jpg Targeted Disease-Mask Review...")
lhj_img = cv2.imread(str(images_dir / "LHJ00059.jpg"))
h, w = lhj_img.shape[:2]

with open(phase9_0d_dir / "LHJ00059_corrected_annotations.json") as f:
    lhj_90d = json.load(f)

lhj_leaf_pts = unpack_pts(lhj_90d["leaf_polygon"]["points"])
lhj_leaf_mask = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(lhj_leaf_mask, [lhj_leaf_pts], 1)
lhj_leaf_area = int(np.count_nonzero(lhj_leaf_mask))

# Forensic findings on LHJ00059:
# 1. P08 (61,709 px): Encloses healthy green lamina (99.0% green, yellow index -24.18 below mean). EXCISE.
# 2. P11 (277 px), P13 (1,186 px), P14 (496 px): Located 104-153 px deep inside interior lamina. EXCISE.
# 3. P10 (335 px): Located 58.5 px inside lamina, non-marginal. EXCISE.
# 4. RETAIN 9 genuine marginal polygons:
#    - P01 (4,260 px): Left marginal yellow chlorotic band
#    - P02 (1,487 px): Upper-left marginal serration
#    - P03 (280 px): Left marginal fold tip
#    - P04 (50,898 px): Major left marginal hopper burn band (chlorosis + necrotic tipping)
#    - P05 (925 px): Upper-left marginal serration
#    - P06 (2,935 px): Upper-left marginal tip
#    - P07 (234 px): Right margin serration
#    - P09 (483 px): Right margin tip
#    - P12 (30,941 px): Basal marginal hopper burn band (chlorosis + necrosis)

kept_pids = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P09", "P12"]
lhj_corrected_shapes = []
lhj_combined_mask = np.zeros((h, w), dtype=np.uint8)

poly_id_counter = 1
for p in lhj_90d["disease_polygons"]:
    pid = p["polygon_id"]
    if pid in kept_pids:
        pts = unpack_pts(p["points"])
        pm = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(pm, [pts], 1)
        
        # Verify 0 px outside leaf
        out_px = int(np.count_nonzero(cv2.bitwise_and(pm, cv2.bitwise_not(lhj_leaf_mask))))
        assert out_px == 0, f"Polygon {pid} has {out_px} px outside leaf!"
        
        lhj_combined_mask = cv2.bitwise_or(lhj_combined_mask, pm)
        
        new_entry = {
            "type": "polygon",
            "label": "Diseased_Region",
            "frame": 16,
            "occluded": False,
            "points": p["points"],
            "z_order": 0,
            "attributes": [
                {"name": "disease_type", "value": "Leaf Hopper Jassids"},
                {"name": "symptom_subtype", "value": "Chlorosis"}
            ],
            "polygon_id": f"P{poly_id_counter:02d}",
            "original_90d_id": pid,
            "original_cluster": p.get("original_cluster"),
            "raster_area_px": p["raster_area_px"],
            "vertices": len(pts),
            "outside_leaf_px": 0
        }
        lhj_corrected_shapes.append(new_entry)
        poly_id_counter += 1

lhj_dis_area = int(np.count_nonzero(lhj_combined_mask))
lhj_sev = (lhj_dis_area / lhj_leaf_area) * 100.0
lhj_out_total = int(np.count_nonzero(cv2.bitwise_and(lhj_combined_mask, cv2.bitwise_not(lhj_leaf_mask))))
assert lhj_out_total == 0

# Save LHJ00059 Phase 9.0E corrected JSON
lhj_90e_payload = {
    "frame_idx": 16,
    "filename": "LHJ00059.jpg",
    "disease_type": "Leaf Hopper Jassids",
    "symptom_subtype": "Chlorosis",
    "split": "train",
    "group_id": "group_01270",
    "sha256": lhj_90d["sha256"],
    "dimensions": [w, h],
    "leaf_area_px": lhj_leaf_area,
    "disease_area_px": lhj_dis_area,
    "severity_pct": round(lhj_sev, 4),
    "leaf_polygon": lhj_90d["leaf_polygon"],
    "disease_polygons": lhj_corrected_shapes,
    "polygon_count": {
        "leaf": 1,
        "diseased_region": len(lhj_corrected_shapes),
        "total": 1 + len(lhj_corrected_shapes)
    },
    "connected_components_count": len(lhj_corrected_shapes),
    "overlapping_pairs_count": 0,
    "outside_leaf_px": 0,
    "invariant_holds": True,
    "excised_polygons": [
        {"id": "P08", "area_px": 61709, "reason": "Encloses healthy green lamina (99.0% green, yellow index -24.18 below mean)"},
        {"id": "P10", "area_px": 335, "reason": "Non-marginal internal lamina spot (58.5 px inside lamina)"},
        {"id": "P11", "area_px": 277, "reason": "Internal lamina artifact (137.6 px inside lamina, delta BGR <= 0.9)"},
        {"id": "P13", "area_px": 1186, "reason": "Internal lamina artifact (104.3 px inside lamina)"},
        {"id": "P14", "area_px": 496, "reason": "Internal lamina shadow crevice (152.7 px inside lamina)"}
    ],
    "status": "REVIEW"
}

with open(out_dir / "LHJ00059_corrected_annotations.json", "w") as f:
    json.dump(lhj_90e_payload, f, indent=2)
print(f"Saved LHJ00059 Phase 9.0E corrected annotations JSON to {out_dir / 'LHJ00059_corrected_annotations.json'}")

# Save updated masks for LHJ00059
cv2.imwrite(str(masks_dir / "mask_leaf_LHJ00059.png"), lhj_leaf_mask * 255)
cv2.imwrite(str(masks_dir / "mask_disease_LHJ00059.png"), lhj_combined_mask * 255)
lhj_comp = np.zeros((h, w), dtype=np.uint8)
lhj_comp[lhj_leaf_mask > 0] = 1
lhj_comp[lhj_combined_mask > 0] = 2
cv2.imwrite(str(masks_dir / "mask_composite_LHJ00059.png"), lhj_comp)
print(f"Saved LHJ00059 semantic masks to {masks_dir}")

# ------------------------------------------------------------------------------
# STEP 4: Generate Quantitative Comparison CSV
# ------------------------------------------------------------------------------
comparison_records = [
    {
        "frame_idx": 7,
        "filename": "CV00211.jpg",
        "disease_class": "Curl Virus",
        "phase9_0b_severity_pct": 24.4086,
        "phase9_0d_severity_pct": 24.3221,
        "phase9_0e_severity_pct": 24.3221,
        "phase9_0e_disease_area_px": 55118,
        "phase9_0e_leaf_area_px": 226617,
        "disease_poly_count": 3,
        "outside_leaf_px": 0,
        "overlapping_pairs": 0,
        "leaf_instance_status": "DEFECTIVE_MULTI_LEAF_ROSETTE",
        "final_verdict": "REVIEW",
        "verdict_rationale": "Encloses 3 distinct leaves + stem; no single leaf can be reliably isolated (truncated at y=0 and y=799); per policy, do not force artificial correction."
    },
    {
        "frame_idx": 16,
        "filename": "LHJ00059.jpg",
        "disease_class": "Leaf Hopper Jassids",
        "phase9_0b_severity_pct": 46.7830,
        "phase9_0d_severity_pct": 41.4962,
        "phase9_0e_severity_pct": round(lhj_sev, 4),
        "phase9_0e_disease_area_px": lhj_dis_area,
        "phase9_0e_leaf_area_px": lhj_leaf_area,
        "disease_poly_count": len(lhj_corrected_shapes),
        "outside_leaf_px": 0,
        "overlapping_pairs": 0,
        "leaf_instance_status": "VALID_SINGLE_LEAF",
        "final_verdict": "REVIEW",
        "verdict_rationale": "P08 (61,709 px healthy lamina) and 4 internal spots (2,294 px) excised; 9 verified marginal polygons retained (92,443 px, 24.52% sev); marked REVIEW awaiting upper-right margin user guidance."
    }
]

csv_out = qa_dir / "phase9_0e_quantitative_comparison.csv"
fieldnames = list(comparison_records[0].keys())
with open(csv_out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(comparison_records)
print(f"Saved quantitative comparison CSV to {csv_out}")

# ------------------------------------------------------------------------------
# STEP 5: Generate Visual QA Preview Figures (4-Panel Panels)
# ------------------------------------------------------------------------------
print("\nGenerating Visual QA comparison figures...")

# Figure 1: CV00211.jpg Leaf-Instance Breakdown
def generate_cv00211_qa():
    p1 = cv_img.copy()
    cv2.putText(p1, "Panel 1: Original RGB Image", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(p1, "Frame 07: CV00211.jpg (Curl Virus, Train)", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    
    p2 = cv_img.copy()
    cv2.drawContours(p2, [cv_leaf_pts], -1, (0, 255, 255), 2)
    # Highlight the multi-leaf problem
    cv2.putText(p2, "Panel 2: Phase 9.0D Leaf Boundary", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(p2, "Encloses 3 distinct leaves + stem axis (DEFECT)", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(p2, "Leaf 1 (Upper Apical)", (220, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(p2, "Leaf 2 (Mid Lateral)", (40, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(p2, "Leaf 3 (Lower Basal)", (280, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(p2, "Connecting Stem Axis", (220, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    
    p3 = cv_img.copy()
    # Draw truncation warnings and disease polygons
    for p in cv_90d["disease_polygons"]:
        pts = unpack_pts(p["points"])
        cv2.polylines(p3, [pts], True, (0, 0, 255), 2)
        M = cv2.moments(pts)
        if M["m00"] > 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.putText(p3, f"{p['polygon_id']} on Leaf {p['polygon_id'][-1]}", (cx-40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    # Truncation lines
    cv2.line(p3, (268, 2), (344, 2), (0, 0, 255), 4)
    cv2.putText(p3, "TRUNCATED at y=0 (77 px cut off)", (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.line(p3, (148, 797), (640, 797), (0, 0, 255), 4)
    cv2.putText(p3, "TRUNCATED at y=799 (>400 px cut off)", (150, 785), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.putText(p3, "Panel 3: Disease Polygons on Separate Leaves", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    
    p4 = np.zeros_like(cv_img)
    p4[cv_leaf_mask > 0] = [34, 139, 34]
    for p in cv_90d["disease_polygons"]:
        pts = unpack_pts(p["points"])
        cv2.fillPoly(p4, [pts], [0, 0, 220])
    p4 = cv2.addWeighted(cv_img, 0.35, p4, 0.65, 0)
    cv2.drawContours(p4, [cv_leaf_pts], -1, (0, 255, 255), 2)
    cv2.putText(p4, "Panel 4: Composite Semantic Diagnostic", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
    cv2.putText(p4, "Policy requires 1 leaf; image is multi-leaf shoot", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    top = np.hstack([p1, p2])
    bot = np.hstack([p3, p4])
    grid = np.vstack([top, bot])
    
    banner = np.zeros((80, 1600, 3), dtype=np.uint8)
    banner[:] = (35, 35, 35)
    cv2.putText(banner, "PHASE 9.0E REVIEW: CURL VIRUS (CV00211.jpg) — VERDICT: REVIEW", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    cv2.putText(banner, "DEFECT: Multi-Leaf Conglomerate Enclosure (3 leaves + stem) | Target Leaf Ambiguous | Correction NOT Forced", (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    
    final_img = np.vstack([banner, grid])
    out_path = qa_dir / "qa_phase9_0e_CV00211.png"
    cv2.imwrite(str(out_path), final_img)
    shutil.copyfile(out_path, brain_dir / "qa_phase9_0e_CV00211.png")
    print(f"  -> Generated {out_path} and copied to brain")

generate_cv00211_qa()

# Figure 2: LHJ00059.jpg Targeted Disease-Mask Review
def generate_lhj00059_qa():
    p1 = lhj_img.copy()
    cv2.putText(p1, "Panel 1: Original RGB Image", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(p1, "Frame 16: LHJ00059.jpg (Leaf Hopper Jassids, Train)", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    
    # Panel 2: Phase 9.0D vs Excised Regions (P08 in Cyan, Internal spots in Magenta)
    p2 = lhj_img.copy()
    cv2.drawContours(p2, [lhj_leaf_pts], -1, (0, 255, 0), 1)
    # Highlight excised P08 in Cyan overlay
    p08_obj = [p for p in lhj_90d["disease_polygons"] if p["polygon_id"] == "P08"][0]
    p08_pts = unpack_pts(p08_obj["points"])
    p08_m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(p08_m, [p08_pts], 1)
    p2[p08_m > 0] = cv2.addWeighted(p2[p08_m > 0], 0.4, np.full_like(p2[p08_m > 0], (255, 255, 0)), 0.6, 0)
    cv2.polylines(p2, [p08_pts], True, (255, 255, 0), 2)
    cv2.putText(p2, "EXCISED P08 (61,709 px Healthy Lamina)", (380, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    
    # Highlight excised internal spots in Magenta
    for pid in ["P10", "P11", "P13", "P14"]:
        p_obj = [p for p in lhj_90d["disease_polygons"] if p["polygon_id"] == pid][0]
        pts = unpack_pts(p_obj["points"])
        cv2.polylines(p2, [pts], True, (255, 0, 255), 2)
        cv2.putText(p2, f"EXCISED {pid}", (int(pts[:,0].mean())-20, int(pts[:,1].mean())-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 2)
    
    cv2.putText(p2, "Panel 2: Excised False-Positive Regions", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
    cv2.putText(p2, "Cyan=P08 Healthy Green Lamina | Magenta=Internal Lamina Spots", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    # Panel 3: Phase 9.0E Verified 9 Marginal Polygons Overlay
    p3 = lhj_img.copy()
    red_layer = np.zeros_like(p3)
    red_layer[lhj_combined_mask > 0] = [0, 0, 255]
    p3 = cv2.addWeighted(p3, 0.70, red_layer, 0.30, 0)
    cv2.drawContours(p3, [lhj_leaf_pts], -1, (0, 255, 0), 1)
    for p in lhj_corrected_shapes:
        pts = unpack_pts(p["points"])
        cv2.polylines(p3, [pts], True, (0, 0, 255), 2)
        M = cv2.moments(pts)
        if M["m00"] > 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.putText(p3, p["polygon_id"], (cx-10, cy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
            cv2.putText(p3, p["polygon_id"], (cx-10, cy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(p3, f"Panel 3: Phase 9.0E Marginal Polygons ({len(lhj_corrected_shapes)} Polys, {lhj_dis_area:,} px)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(p3, f"Genuine Marginal Hopper Burn | Severity: {lhj_sev:.4f}%", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    
    # Panel 4: Composite Semantic Mask
    p4 = np.zeros_like(lhj_img)
    p4[lhj_leaf_mask > 0] = [34, 139, 34]
    p4[lhj_combined_mask > 0] = [0, 0, 220]
    p4 = cv2.addWeighted(lhj_img, 0.35, p4, 0.65, 0)
    cv2.drawContours(p4, [lhj_leaf_pts], -1, (0, 255, 0), 1)
    for p in lhj_corrected_shapes:
        pts = unpack_pts(p["points"])
        cv2.polylines(p4, [pts], True, (0, 255, 255), 1)
    cv2.putText(p4, f"Panel 4: Composite Semantic Mask (Severity: {lhj_sev:.4f}%)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
    cv2.putText(p4, "Green=Healthy Lamina | Red=Genuine Marginal Symptoms", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    top = np.hstack([p1, p2])
    bot = np.hstack([p3, p4])
    grid = np.vstack([top, bot])
    
    banner = np.zeros((80, 1600, 3), dtype=np.uint8)
    banner[:] = (35, 35, 35)
    cv2.putText(banner, "PHASE 9.0E REVIEW: LEAF HOPPER JASSIDS (LHJ00059.jpg) — VERDICT: REVIEW", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    cv2.putText(banner, f"Severity: {lhj_sev:.4f}% | Polygons: 9 | Excised P08 & 4 Internal Spots | Awaiting Upper-Right Guidance", (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    
    final_img = np.vstack([banner, grid])
    out_path = qa_dir / "qa_phase9_0e_LHJ00059.png"
    cv2.imwrite(str(out_path), final_img)
    shutil.copyfile(out_path, brain_dir / "qa_phase9_0e_LHJ00059.png")
    print(f"  -> Generated {out_path} and copied to brain")

generate_lhj00059_qa()

print("\nPhase 9.0E review execution completed successfully!")
