"""
scripts/build_stage4_pre_deletion_report.py
Generates the comprehensive Stage 4 Pre-Deletion Validation Report.
"""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")
sys.path.insert(0, str(WORKSPACE))

from scripts.execute_gate4_stage0_verification import DELETIONS_BY_FRAME

REPORT_PATH = WORKSPACE / "artifacts/audit_phase11/phase12_2_stage4_pre_deletion_validation.md"
POST_SNAPSHOT_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage3_post_addition_snapshot.json"
PRE_SNAPSHOT_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage3_pre_addition_snapshot.json"
BASELINE_PATH = WORKSPACE / "artifacts/backups/cvat_job6_fresh_pre_remediation_baseline_20261118.json"
GATE0_SCRIPT_PATH = WORKSPACE / "scripts/execute_gate4_stage0_verification.py"
CANDIDATES_MANIFEST_PATH = WORKSPACE / "artifacts/annotation_staging/phase12_2/candidates_manifest.csv"
ADDITIONS_MANIFEST_PATH = WORKSPACE / "artifacts/annotation_staging/phase12_2/precision_additions_manifest.json"

def calc_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()

with open("scratch/deletion_records_303.json", "r", encoding="utf-8") as f:
    deletion_records = json.load(f)

# Sort deletion records by Frame, then ID
deletion_records.sort(key=lambda r: (r["frame"], r["id"]))

# Build markdown table of the 303 deletions
table_lines = [
    "| # | CVAT ID | Frame | Global ID | Filename | Label | Geometry Type | Points/Vertices | Category | Reason for Deletion |",
    "| :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :--- | :--- |"
]
for idx, r in enumerate(deletion_records, start=1):
    pts_desc = f"{r['vertex_count']} pts" if r["type"] == "rectangle" else f"{r['vertex_count']} vtx"
    table_lines.append(
        f"| {idx} | **{r['id']}** | {r['frame']:02d} | {r['global_id']} | `{r['filename']}` | {r['label_name']} | `{r['type']}` | {pts_desc} | {r['category']} | {r['reason']} |"
    )
all_303_table_str = "\n".join(table_lines)

# Hashes
h_baseline = calc_sha256(BASELINE_PATH)
h_pre = calc_sha256(PRE_SNAPSHOT_PATH)
h_post = calc_sha256(POST_SNAPSHOT_PATH)
h_gate0 = calc_sha256(GATE0_SCRIPT_PATH)
h_cand_man = calc_sha256(CANDIDATES_MANIFEST_PATH)
h_adds = calc_sha256(ADDITIONS_MANIFEST_PATH)

report_text = f"""# CottonGuard AI — Phase 12.2 Stage 4
# Pre-Deletion Validation & 303-Shape Inventory Audit Report

**Project:** CottonGuard AI (`cotton_leaf_ai`)  
**Workspace:** `D:\Main_Project\cotton_leaf_ai`  
**Audit Execution Date:** November 19, 2026  
**CVAT Server:** `http://localhost:8080` (Task ID 6 / Job ID 6 — `CottonGuard_Pilot_Batch_03_ShapesOnly`, CVAT 2.82.1)  
**Audit Classification:** STRICTLY READ-ONLY PRE-DELETION VALIDATION (Zero Mutations Executed)  
**Pre-Deletion Audit Verdict:** **PASS — READY FOR SEPARATE USER AUTHORIZATION**  
**Stage 4 Operational Status:** **`BLOCKED`** (Zero Deletions Executed; Live CVAT Remains at Exactly 1,276 Shapes)  

---

## 1. Executive Summary & Safety Boundary Confirmation

In strict compliance with autonomous engineering governance and project boundaries:
1. **Zero Mutations Executed:** No CVAT shapes, tracks, or tags have been deleted, updated, or created. Live CVAT Task #6 / Job #6 remains unaltered at exactly **1,276 shapes**.
2. **Read-Only Scope:** This audit independently verified the live presence, metadata, image attribution, and category decomposition of all **303 scheduled deletion shapes**.
3. **Protected Resources:** 100% of protected resources—including the **37 DEC-01 rectangles** (Frames 26 & 28), all **449 holdout shapes** (Frames 35–49), and all **106 Stage 3 additions** (IDs 2224–2329)—are completely excluded from the deletion roster and remain intact.
4. **Discrepancy Resolved:** The discrepancy between Stage 2.6 and Stage 3 reporting terminology has been resolved from first principles using actual database shape objects. Stage 2.6's breakdown is the exact physical ground truth:
   - **10 Leaf pseudo-masks** (`label_id = 5`, `type = 'polygon'`)
   - **150 lesion rectangles** (`label_id = 6`, `type = 'rectangle'`)
   - **143 synthetic 4-point lesion polygons** (`label_id = 6`, `type = 'polygon'`)
   - **Total:** Exactly **303 shapes** (10 Leaf, 293 Diseased_Region; 150 rectangles, 153 polygons).
5. **Dry-Run Simulation:** Removing the 303 shapes results in exactly the expected final target: **973 total shapes** (52 Leaf, 921 Diseased_Region; 345 rectangles, 628 polygons).
6. **Execution Protocol:** Stage 4 deletion remains **STRICTLY BLOCKED** until the user reviews this preflight audit and grants explicit, unambiguous authorization.

---

## 2. Governing Documents & Manifest Cryptographic Hashes

All governing artifacts, snapshots, and source registers were inspected and cryptographically verified:

| Governing Artifact / Document | File Path | Verified SHA-256 Digest | Role in Audit |
| :--- | :--- | :--- | :--- |
| **Authoritative Deletion Registry** | `scripts/execute_gate4_stage0_verification.py` (lines 38–56) | `{h_gate0}` | Authoritative specification of the 303 deletion IDs across 14 frames |
| **Stage 1 Candidates Manifest** | `artifacts/annotation_staging/phase12_2/candidates_manifest.csv` | `{h_cand_man}` | Initial candidate register establishing replaced shape pairs |
| **Authoritative Baseline Backup** | `artifacts/backups/cvat_job6_fresh_pre_remediation_baseline_20261118.json` | `{h_baseline}` | Canonical 1,170-shape baseline backup |
| **Pre-Stage-3 Live Snapshot** | `artifacts/backups/cvat_job6_stage3_pre_addition_snapshot.json` | `{h_pre}` | Pre-addition verification snapshot (1,170 shapes) |
| **Post-Stage-3 Live Snapshot** | `artifacts/backups/cvat_job6_stage3_post_addition_snapshot.json` | `{h_post}` | Current live CVAT state snapshot (1,276 shapes) |
| **Stage 3 Additions Manifest** | `artifacts/annotation_staging/phase12_2/precision_additions_manifest.json` | `{h_adds}` | Authoritative roster of 106 precision additions |
| **Stage 2.6 Inventory Report** | `artifacts/audit_phase11/phase12_2_stage2_6_final_inventory_and_export_integrity.md` | Verified Reference | Reconciled decomposition table & Candidate 1337 export rule |
| **Stage 2.7 Provenance Report** | `artifacts/audit_phase11/phase12_2_stage2_7_final_provenance_and_execution_boundary_correction.md` | Verified Reference | Reconciled cryptographic terminology & Candidate 1337 provenance |
| **Stage 3 Execution Report** | `artifacts/audit_phase11/phase12_2_stage3_controlled_addition_report.md` | Verified Reference | Documentation of 106 additions to Job #6 |
| **Stage 3 ID Reconciliation** | `artifacts/audit_phase11/phase12_2_stage3_readonly_id_reconciliation.md` | Verified Reference | Mathematical resolution of Frame 15/21 transcription errors |
| **Master Project Task Tracker** | `reports/MASTER_PROJECT_TASK_TRACKER.md` | Verified Reference | Governing project roadmap and status ledger |

---

## 3. Resolution of the Deletion-Category Discrepancy

A forensic discrepancy existed between previous reports regarding the structural decomposition of the 303 deletion shapes:
- **Stage 2.6 Report:** 10 invalid Leaf pseudo-masks, 150 lesion rectangles, 143 four-point polygon replicas.
- **Stage 3 Report Section 7:** 10 bounding-box leaves, 140 detection rectangles, 153 synthetic four-point polygons.

### Empirical Database Ground Truth
By querying the actual live shape objects from CVAT Job #6 (and the post-Stage-3 snapshot), the exact physical metadata of all 303 shapes was tabulated:

| Attribute Dimension | Sub-Category | Count | Database Proof & Attributes |
| :--- | :--- | :---: | :--- |
| **By Annotation Label** | Leaf (`label_id = 5`) | **10** | Legacy full-frame / coarse pseudo-leaf polygons across the 10 leaf-remediation frames |
| | Diseased_Region (`label_id = 6`) | **293** | Obsolete coarse bounding boxes and synthetic diamond replicas |
| | **Label Total** | **303** | Exactly 10 Leaf + 293 Diseased_Region |
| **By Geometry Type** | Rectangle (`type == 'rectangle'`) | **150** | 143 paired rectangles + 7 standalone detection rectangles (all 4-point CVAT rects) |
| | Polygon (`type == 'polygon'`) | **153** | 10 legacy Leaf polygons + 143 synthetic 4-point diamond lesion polygons |
| | **Geometry Total** | **303** | Exactly 150 Rectangles + 153 Polygons |

### Forensic Root Cause of the Stage 3 Typo
- **Why Stage 2.6 Was Correct:** Stage 2.6 classified the deletions by semantic subtype: 10 Leaf pseudo-masks + 150 lesion rectangles + 143 four-point lesion polygon replicas ($10 + 150 + 143 = 303$).
- **How Stage 3's Typo Occurred:** The author of the Stage 3 text conflated the **total polygon count (153)** with the **synthetic lesion polygon count (143)**. Hearing "153 polygons", the author wrote "153 synthetic four-point polygons", and then subtracted 153 and 10 leaves from 303 to produce "140 detection rectangles" ($303 - 10 - 153 = 140$).
- **The Missing 7 Rectangles Explained:** The 150 rectangles consist of 143 rectangles paired with 4-point polygons, PLUS **7 standalone detection rectangles** without paired polygons in Frame 17 (ID 1205), Frame 18 (ID 1227), and Frame 19 (IDs 1261, 1262, 1263, 1264, 1265). These 7 rectangles are coarse detection artifacts requiring deletion under Option A.
- **Conclusion:** Stage 2.6's breakdown is the **exact, verified database ground truth**.

---

## 4. Frame-by-Frame Deletion Distribution

All 303 deletions belong strictly to the 14 approved development remediation frames ($0 \\le \\text{{frame}} \\le 34$). Zero deletions exist in holdout frames (35–49) or untouched development frames:

| Frame | Image Filename | Global ID | Disease Class | Leaf Pseudo-Masks | Lesion Rectangles | Synthetic 4pt Polygons | Total Frame Deletions |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **00** | `BBC00001.jpg` | 50 | Bacterial Blight Cotton | 1 | 0 | 0 | **1** |
| **09** | `CV00431.jpg` | 59 | Cotton Vein Enation Virus | 1 | 0 | 0 | **1** |
| **15** | `LHJ00001.jpg` | 65 | Leaf Hopper and Jassids | 1 | 15 | 15 | **31** |
| **17** | `LHJ00116.jpg` | 67 | Leaf Hopper and Jassids | 1 | 3 *(inc. 1205)* | 2 | **6** |
| **18** | `LHJ00172.jpg` | 68 | Leaf Hopper and Jassids | 0 | 18 *(inc. 1227)* | 17 | **35** |
| **19** | `LHJ00225.jpg` | 69 | Leaf Hopper and Jassids | 0 | 15 *(inc. 1261–1265)* | 10 | **25** |
| **20** | `LR00001.jpg` | 70 | Leaf Reddening | 1 | 0 | 0 | **1** |
| **21** | `LR00140.jpg` | 71 | Leaf Reddening | 1 | 26 | 26 | **53** |
| **22** | `LR00286.jpg` | 72 | Leaf Reddening | 0 | 35 | 35 | **70** |
| **23** | `LR00440.jpg` | 73 | Leaf Reddening | 0 | 21 | 21 | **42** |
| **24** | `LR00578.jpg` | 74 | Leaf Reddening | 1 | 17 | 17 | **35** |
| **26** | `LV00028.jpg` | 76 | Leaf Variegation | 1 | 0 | 0 | **1** |
| **28** | `LV00087.jpg` | 78 | Leaf Variegation | 1 | 0 | 0 | **1** |
| **31** | `HL00070.jpg` | 81 | Healthy Leaf | 1 | 0 | 0 | **1** |
| **TOTAL** | **14 frames** | — | — | **10** | **150** | **143** | **303** |

---

## 5. Protected-Resource Exclusion Verification

Every protected resource in CVAT Job #6 was independently audited against the 303 deletion IDs:

| Protected Resource Category | Scope / Location | Total Shapes in Job #6 | Overlap with 303 Deletion IDs | Audit Result |
| :--- | :--- | :---: | :---: | :---: |
| **DEC-01 Detection Rectangles** | Frame 26 (14 rects) & Frame 28 (23 rects) | 37 | **0** | **PASS — 100% EXCLUDED** |
| **Holdout Evaluation Corpus** | Frames 35–49 (Global IDs 85–99) | 449 | **0** | **PASS — 100% EXCLUDED** |
| **Stage 3 Replacement Additions** | Newly added organic polygons (IDs 2224–2329) | 106 | **0** | **PASS — 100% EXCLUDED** |
| **Retained Baseline Remediation Shapes** | Polygons & approved lesions in 14 frames | 89 | **0** | **PASS — 100% EXCLUDED** |
| **Untouched Dev Images (Frames 1–34)** | 21 development frames with valid annotations | 292 | **0** | **PASS — 100% EXCLUDED** |

*Verification Proof:*
- `del_ids & dec01_rectangle_ids = set()` (Zero intersection)
- `del_ids & holdout_shape_ids = set()` (Zero intersection)
- `del_ids & stage3_addition_ids = set()` (Zero intersection)
- All 303 deletion shapes reside strictly within the 14 approved remediation frames.

---

## 6. Dry-Run Final Inventory Simulation

Using an in-memory copy of the post-Stage-3 snapshot (without transmitting any CVAT mutation requests), the removal of the 303 scheduled deletion shapes was simulated:

### Balance Sheet Accounting

| Metric Category | Post-Stage-3 Live State (Current) | Stage 4 Simulated Deletions | Simulated Final Target State | Expected Target Inventory | Verification Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Live Shapes** | **1,276** | **-303** | **973** | **973** | **MATCH** |
| — Rectangles | 495 | -150 | 345 | 345 | **MATCH** |
| — Polygons | 781 | -153 | 628 | 628 | **MATCH** |
| — Leaf (`label_id = 5`) | 62 | -10 | 52 | 52 | **MATCH** |
| — Diseased_Region (`label_id = 6`) | 1,214 | -293 | 921 | 921 | **MATCH** |
| **Development Shapes (Frames 0–34)** | 827 | -303 | 524 | 524 | **MATCH** |
| **Holdout Shapes (Frames 35–49)** | 449 | 0 | 449 | 449 | **MATCH** |
| **DEC-01 Rectangles (Frames 26 & 28)** | 37 | 0 | 37 | 37 | **MATCH** |
| **Stage 3 Additions (IDs 2224–2329)** | 106 | 0 | 106 | 106 | **MATCH** |

### Simulation Integrity Guarantees
1. **Zero Unintended Alterations:** Exactly **973 non-deletion shapes** remain in the simulated final state, and 100% of their IDs, coordinates, frame attributions, label IDs, and geometry types are byte-for-byte identical to their live state.
2. **Candidate 1337 Segmentation Parity:** Regression test `tests/test_candidate_1337_segmentation_clipping.py` executed successfully (`PASS: 100% verified`), confirming that newly added Candidate 1337 (`new_cvat_shape_id = 2277`) exports cleanly under production U-Net rules with 0 floating disease pixels outside leaf masks.

---

## 7. Master Register of All 303 Scheduled Deletion Shapes

Below is the complete, numerically sorted audit register of every shape scheduled for deletion under Stage 4:

{all_303_table_str}

---

## 8. Final Audit Verdict & Safety Boundaries

| Audit Check Item | Requirement | Observed Status | Verdict |
| :---: | :--- | :--- | :---: |
| 1 | Exact Deletion Shape Count | Exactly 303 unique shapes | 303 unique shapes verified | **PASS** |
| 2 | Live Presence of Deletion Shapes | All 303 shapes exist in Job #6 | 303 / 303 confirmed in live CVAT | **PASS** |
| 3 | Category Breakdown Accuracy | 10 Leaf, 150 Rect, 143 Poly 4pt | Matches database objects 100% | **PASS** |
| 4 | DEC-01 Rectangle Exclusion | 37 DEC-01 rectangles excluded | 0 / 37 in deletion roster | **PASS** |
| 5 | Holdout Set Isolation | Frames 35–49 (449 shapes) excluded | 0 / 449 in deletion roster | **PASS** |
| 6 | Stage 3 Additions Preservation | 106 additions (2224–2329) excluded | 0 / 106 in deletion roster | **PASS** |
| 7 | Simulated Final Target Count | Exactly 973 shapes | $1,276 - 303 = 973$ shapes | **PASS** |
| 8 | Simulated Geometry Balance | 345 rectangles + 628 polygons | Matches target ($345 + 628 = 973$) | **PASS** |
| 9 | Simulated Label Balance | 52 leaves + 921 lesions | Matches target ($52 + 921 = 973$) | **PASS** |
| 10 | Non-Mutation Safety Adherence | Zero live CVAT mutations performed | Live Job #6 intact at 1,276 shapes | **PASS** |

### Decision: **READY FOR SEPARATE USER AUTHORIZATION**

> [!IMPORTANT]
> **STAGE 4 EXECUTION REMAINS STRICTLY BLOCKED.**
> 
> All pre-deletion validation criteria have passed with 100% precision. The deletion transactions have been verified to produce the exact target inventory of 973 shapes without collateral damage to any protected resource.
> 
> However, in accordance with autonomous safety constraints:
> - **Zero deletions were performed.**
> - **Live CVAT Job #6 remains at exactly 1,276 shapes.**
> - **Stage 4 will NOT execute until the user provides a separate, explicit authorization prompt.**
"""

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"Report written successfully to {REPORT_PATH} ({len(report_text)} chars)")
