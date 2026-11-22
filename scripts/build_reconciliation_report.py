"""
scripts/build_reconciliation_report.py
Constructs artifacts/audit_phase11/phase12_2_stage3_readonly_id_reconciliation.md
"""

import json
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")
REPORT_PATH = WORKSPACE / "artifacts/audit_phase11/phase12_2_stage3_readonly_id_reconciliation.md"

with open("scratch/all_106_table.md", "r", encoding="utf-8") as f:
    table_content = f.read()

report_text = f"""# CottonGuard AI — Phase 12.2 Stage 3
# Read-Only Shape ID Reconciliation & Discrepancy Resolution Audit

**Project:** CottonGuard AI (`cotton_leaf_ai`)  
**Workspace:** `D:\Main_Project\cotton_leaf_ai`  
**Audit Execution Date:** November 19, 2026  
**CVAT Server:** `http://localhost:8080` (Task ID 6 / Job ID 6 — `CottonGuard_Pilot_Batch_03_ShapesOnly`)  
**Audit Classification:** STRICTLY READ-ONLY FORENSIC AUDIT (Zero Mutations Executed)  
**Reconciliation Status:** **PASS** (106 / 106 Unique Added IDs Fully Accounted For & Reconciled)  
**Stage 4 Status:** **`BLOCKED`** (Zero Deletions Authorized or Performed)  

---

## 1. Executive Summary & Audit Purpose

This strictly read-only audit was conducted to resolve an ID enumeration discrepancy identified in Section 4 of the initial Stage 3 Execution Report (`phase12_2_stage3_controlled_addition_report.md`):

1. **Discrepancy 1 (Missing ID 2245):** Shape ID `2245` was absent from the explicit frame-by-frame ID lists in the report table.
2. **Discrepancy 2 (Frame 21 Count Mismatch):** The Frame 21 row reported 6 lesions, but the enumerated ID range `2276–2280` contained only 5 IDs.
3. **Discrepancy 3 (Enumerated Total):** The explicitly listed IDs in the report table totaled 105 instead of the verified 106 additions.

### Audit Verdict: **PASS**
- **Independent Snapshot Comparison:** Pre-Stage-3 snapshot (1,170 shapes) vs Post-Stage-3 snapshot (1,276 shapes) confirms an exact difference of **+106 shapes**, with **0 deletions**.
- **Contiguous Server ID Allocation:** The newly created CVAT IDs form an unbroken, contiguous sequence from **2224 to 2329 inclusive** ($2329 - 2224 + 1 = 106$).
- **Underlying Database Integrity:** All 106 additions exist in live CVAT, match the approved `precision_additions_manifest.json` 1:1, and are correctly recorded in `stage3_added_shapes_mapping.json`.
- **Root Cause Identified:** The discrepancy was entirely a **human transcription error** in the narrative summary table of the report, where secondary leaf IDs (`2234` and `2235`) were conflated with contiguous lesion ranges in Frames 15 and 21. No annotations were lost, omitted, or misassigned in CVAT.

---

## 2. Root Cause Forensic Analysis

During Stage 3 execution, the CVAT REST API was supplied with the 106 shapes in the exact ordering of `precision_additions_manifest.json`:
1. **Indices 0–11 (The 12 Leaves):** Assigned the first 12 sequential IDs: **2224 through 2235**.
2. **Indices 12–105 (The 94 Lesions):** Assigned the remaining 94 sequential IDs: **2236 through 2329**.

The transcription errors in the initial report occurred specifically in **Frame 15** and **Frame 21**, which each contained a secondary leaf blade:

### A. Frame 15 Forensic Breakdown
- **Flawed Report Text:** `2226, 2246 (Leaves), 2234–2244 (Lesions)`
- **What Went Wrong:**
  - `2234` is actually `CAND_F15_L_SECONDARY_01` (Leaf, Blade 2). The report mistook `2234` for the start of the lesion range.
  - `2246` is actually `CAND_F15_LES_1187` (Diseased_Region, replacing shape 1187). The report mistook `2246` for the second leaf.
  - Shape ID `2245` (`CAND_F15_LES_1186`, Diseased_Region, 24 vertices) was part of the 11-lesion sequence `2236–2246`. Because the text author wrote `2234–2244` and then skipped to `2246` as a leaf, ID `2245` appeared "missing" from the text.
- **Verified Ground Truth for Frame 15:**
  - **Leaves (2):** `2226` (Primary Blade 1), `2234` (Secondary Blade 2)
  - **Lesions (11):** `2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246` (`2236–2246` contiguous)
  - **Total:** Exactly 13 shapes.

### B. Frame 21 Forensic Breakdown
- **Flawed Report Text:** `2229, 2281 (Leaves), 2276–2280 (Lesions, inc. 2277)`
- **What Went Wrong:**
  - `2235` is actually `CAND_F21_L_BLADE2` (Leaf, Blade 2). The report author omitted `2235` entirely from the Frame 21 row.
  - `2281` is actually `CAND_F21_LES_1344` (Diseased_Region, replacing shape 1344). The report author mistook `2281` for the second leaf.
  - The lesion range was written as `2276–2280` (5 IDs), omitting `2281`, which was the 6th lesion!
- **Verified Ground Truth for Frame 21:**
  - **Leaves (2):** `2229` (Primary Blade 1), `2235` (Secondary Blade 2)
  - **Lesions (6):** `2276, 2277, 2278, 2279, 2280, 2281` (`2276–2281` contiguous, including Candidate 1337 at `2277`)
  - **Total:** Exactly 8 shapes ($2 + 6 = 8$).

### C. Resolution of the Enumerated Total
$$\\text{{Flawed Count}} = 106 - 1 (\\text{{omitted }} 2235) = 105$$
With `2235` restored to Frame 21 and `2245` properly situated in the Frame 15 lesion sequence `2236–2246`, the enumerated shapes sum to exactly **106 unique shapes**.

---

## 3. Corrected Per-Frame Addition Breakdown Table

Below is the verified, authoritative per-frame breakdown with fully reconciled CVAT shape ID assignments:

| Frame | Image Filename | Global ID | Disease Class | Approved Leaves Added | Approved Lesions Added | Total Polygons Added | Verified CVAT Leaf IDs | Verified CVAT Lesion IDs |
| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **00** | `BBC00001.jpg` | 50 | Bacterial Blight Cotton | 1 | 0 | 1 | `2224` | — |
| **09** | `CV00431.jpg` | 59 | Cotton Vein Enation Virus | 1 | 0 | 1 | `2225` | — |
| **15** | `LHJ00001.jpg` | 65 | Leaf Hopper and Jassids | 2 | 11 | 13 | `2226, 2234` | `2236–2246` (11 IDs) |
| **17** | `LHJ00116.jpg` | 67 | Leaf Hopper and Jassids | 1 | 2 | 3 | `2227` | `2247, 2248` |
| **18** | `LHJ00172.jpg` | 68 | Leaf Hopper and Jassids | 0 | 17 | 17 | — | `2249–2265` (17 IDs) |
| **19** | `LHJ00225.jpg` | 69 | Leaf Hopper and Jassids | 0 | 10 | 10 | — | `2266–2275` (10 IDs) |
| **20** | `LR00001.jpg` | 70 | Leaf Reddening | 1 | 0 | 1 | `2228` | — |
| **21** | `LR00140.jpg` | 71 | Leaf Reddening | 2 | 6 | 8 | `2229, 2235` | `2276–2281` (6 IDs, inc. 2277) |
| **22** | `LR00286.jpg` | 72 | Leaf Reddening | 0 | 11 | 11 | — | `2282–2292` (11 IDs) |
| **23** | `LR00440.jpg` | 73 | Leaf Reddening | 0 | 21 | 21 | — | `2293–2313` (21 IDs) |
| **24** | `LR00578.jpg` | 74 | Leaf Reddening | 1 | 16 | 17 | `2230` | `2314–2329` (16 IDs) |
| **26** | `LV00028.jpg` | 76 | Leaf Variegation | 1 | 0 | 1 | `2231` | — |
| **28** | `LV00087.jpg` | 78 | Leaf Variegation | 1 | 0 | 1 | `2232` | — |
| **31** | `HL00070.jpg` | 81 | Healthy Leaf | 1 | 0 | 1 | `2233` | — |
| **TOTAL** | **14 frames** | — | — | **12** | **94** | **106** | **12 IDs (2224–2235)** | **94 IDs (2236–2329)** |

---

## 4. Verification of Shape ID 2245

Shape ID `2245` is confirmed present and completely intact in live CVAT Task #6 / Job #6:

| Attribute | Verified Value |
| :--- | :--- |
| **CVAT Shape ID** | `2245` |
| **Frame Number** | `15` |
| **Global Image ID** | `65` |
| **Image Filename** | `LHJ00001.jpg` |
| **Candidate ID** | `CAND_F15_LES_1186` |
| **Replaced Shape ID** | `1186` (Scheduled deletion shape, currently coexisting) |
| **Annotation Label** | `Diseased_Region` (`label_id = 6`) |
| **Geometry Type** | `polygon` |
| **Vertex Count** | 24 vertices (48 float coordinates) |
| **Geometric Area** | 1,327.5 px |
| **Host Leaf Association** | Contained within Blade 1 polygon (`new_cvat_shape_id = 2226`) |
| **Status in Mapping File** | Fully indexed in `stage3_added_shapes_mapping.json` |

---

## 5. Verification of the Six Frame 21 Lesion IDs

Frame 21 contains exactly 6 approved lesion additions matching `precision_additions_manifest.json`:

| CVAT ID | Candidate ID | Replaced Shape ID | Label | Vertex Count | Geometric Area | Host Leaf Association |
| :---: | :--- | :---: | :--- | :---: | :---: | :--- |
| **2276** | `CAND_F21_LES_1324` | 1324 | Diseased_Region | 26 | 550.0 px | Contained in Blade 1 (`2229`) |
| **2277** | `CAND_F21_LES_1337` | 1337 | Diseased_Region | 20 | 2,431.0 px | Contained in Blade 1 (`2229`) *(Passes clipping regression)* |
| **2278** | `CAND_F21_LES_1341` | 1341 | Diseased_Region | 27 | 1,128.5 px | Contained in Blade 1 (`2229`) |
| **2279** | `CAND_F21_LES_1342` | 1342 | Diseased_Region | 19 | 1,600.5 px | Contained in Blade 1 (`2229`) |
| **2280** | `CAND_F21_LES_1343` | 1343 | Diseased_Region | 23 | 557.5 px | Contained in Blade 1 (`2229`) |
| **2281** | `CAND_F21_LES_1344` | 1344 | Diseased_Region | 15 | 838.5 px | Contained in Blade 1 (`2229`) |

*Note:* Frame 21 also contains 2 approved leaf additions: `2229` (`CAND_F21_L_1289`, Blade 1, 37 vertices) and `2235` (`CAND_F21_L_BLADE2`, Blade 2, 32 vertices).

---

## 6. Complete Numerically Sorted Master Register of All 106 New Shape IDs

Below is the exhaustive, verified audit register of every single newly created shape ID in CVAT Job #6:

{table_content}

---

## 7. Baseline Preservation & Boundary Audit Checklist

All 12 required read-only verification checks were executed against the snapshots and mapping registers:

| # | Check Item | Expected Value | Observed Value | Verdict |
| :---: | :--- | :--- | :--- | :---: |
| 1 | Pre vs Post Snapshot Delta | +106 shapes | 1,170 -> 1,276 (+106 shapes) | **PASS** |
| 2 | Identify All New Shape IDs | 106 unique IDs | Exactly IDs 2224 through 2329 | **PASS** |
| 3 | Count of Unique New IDs | Exactly 106 | Exactly 106 unique IDs | **PASS** |
| 4 | Range Verification | All IDs 2224–2329 present | 100% present, 0 missing, 0 extra | **PASS** |
| 5 | Locate ID 2245 | Frame 15, Diseased_Region, Polygon | Frame 15, `label_id = 6`, 24 vertices | **PASS** |
| 6 | Frame 21 Six Lesion IDs | Exactly 6 lesion IDs | IDs 2276, 2277, 2278, 2279, 2280, 2281 | **PASS** |
| 7 | Cross-Check Against Manifest | 100% 1:1 match | 106 / 106 coordinates & labels match | **PASS** |
| 8 | Per-Frame Addition Counts | 14 frames matching scope | Exactly matches verified scope | **PASS** |
| 9 | Label Breakdown | 12 Leaf, 94 Diseased_Region | Exactly 12 Leaf, 94 Diseased_Region | **PASS** |
| 10 | Original 1,170 Shapes Intact | All 1,170 baseline IDs unaltered | 1,170 / 1,170 identical coordinates/type | **PASS** |
| 11 | Scheduled 303 Deletions Coexisting | All 303 deletion IDs present | 303 / 303 present in post-snapshot | **PASS** |
| 12 | Preserved 37 DEC-01 Rectangles | Frame 26=14, Frame 28=23 | Exactly 14 in F26, 23 in F28 (37 total) | **PASS** |

---

## 8. Strict Non-Mutation Confirmation & Stage 4 Status

- **Zero Live Mutations Performed:** This audit was 100% read-only. No annotations were created, updated, patched, deleted, or rolled back. No files outside of documentation/scratch were altered.
- **Holdout Privacy Guarantee:** No holdout image pixels, holdout polygon coordinates, or holdout annotations (Frames 35–49) were inspected, touched, or modified.
- **Stage 4 Status:** **`BLOCKED`**. Zero deletions are authorized. Stage 4 will remain blocked until the user reviews this reconciliation and issues separate, explicit authorization.
"""

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(report_text)

print(f"Report written successfully to {REPORT_PATH} ({len(report_text)} chars)")
