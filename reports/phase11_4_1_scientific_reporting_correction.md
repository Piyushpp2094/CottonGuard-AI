# CottonGuard AI — Phase 11.4.1: Scientific Reporting Correction & Evidence Reconciliation

**Date:** November 17, 2026  
**Auditor Role:** Senior Machine Learning Engineer, Computer Vision Engineer and Independent AI Model Auditor  
**Project:** CottonGuard AI (`D:\Main_Project\cotton_leaf_ai`)  
**Phase:** 11.4.1 — Scientific Reporting Correction & Evidence Reconciliation  
**Audit Status:** `COMPLETED`  
**Classification Checkpoint:** `artifacts/checkpoints/classification/best_model.pt`  
**Checkpoint SHA-256:** `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e` (Authoritative Match: **YES**)  
**Sealed Test Set Protection:** 100% Preserved (0/321 test images accessed, loaded, or evaluated)  
**Database State:** 100% Preserved (48 predictions rows, 5 users rows; zero writes)  
**Approval Gate 4:** `PENDING_APPROVAL` (Strictly Preserved)  

---

## 1. Executive Summary

This report documents the documentation-only scientific reporting review, evidence reconciliation, and terminology correction conducted under **Phase 11.4.1** for the CottonGuard AI platform.

The objective of Phase 11.4.1 was to ensure that the findings and conclusions of Phase 11.4 (*Controlled Prediction Repeatability & Reliability Audit*) strictly and faithfully reflect empirical measurements without overstating prediction repeatability, label agreement, statistical validity, or application robustness.

### Summary of Actions Taken
1. **Reconciliation of Primary Evidence:** All 6 Phase 11.4 evidence files were inspected. A transcription discrepancy between the narrative table in Section 6 and the primary CSV/JSON probability evidence was identified and corrected.
2. **Correction of Correctness Interpretation:** Replaced generic "100% accuracy" phrasing with **"100% selected-sample label agreement (14/14)"**. Explicitly clarified that training samples were part of model development and validation samples were part of checkpoint selection, so these results cannot represent overall model accuracy, validation accuracy, or external generalization.
3. **Statistical Methodology Clarification:** Documented that selecting the first valid non-excluded sample per class/split is a **deterministic selection protocol** for pipeline testing, not a randomized or statistically representative sample of dataset accuracy.
4. **Correction of Probability Margin Units:** Expressed all Top-2 decision margins in **percentage points (pp)** rather than percent (%), and verified every calculation against primary probability logs.
5. **Delineation of Robustness Boundaries:** Renamed the robustness test to **"Model Input Channel Contract Check"** and explicitly clarified that testing PyTorch convolutional input dimensions does not verify HTTP upload boundaries, image decoding resilience, or end-to-end API error handling.
6. **Precision in Determinism Terminology:** Replaced claims of "universal mathematical guarantees" with **"Observed deterministic repeatability in the tested environment"** (14 images, 5 passes each, 70 forward passes, 0.00000000 drift).
7. **Resolution of OS Metadata Discrepancy:** Analyzed the operating-system metadata, demonstrating that Python's `platform.platform()` emitted `Windows-10-10.0.26300-SP0` because Microsoft maintains the legacy registry value `ProductName = "Windows 10 Home Single Language"`, whereas the underlying system is **Windows 11 Build 26300 (26H2)**.
8. **Documentation Invariance:** Cryptographically recorded before-and-after document hashes and verified that zero model weights, code files, test-set images, or database rows were modified.

---

## 2. Evidence Files Inspected

All 6 primary files associated with Phase 11.4 were cryptographically inspected and verified:

| File Path | File Size | SHA-256 Checksum | Inspection Outcome |
| :--- | :---: | :--- | :--- |
| [`reports/phase11_4_controlled_prediction_repeatability_audit.md`](file:///D:/Main_Project/cotton_leaf_ai/reports/phase11_4_controlled_prediction_repeatability_audit.md) | 20,776 bytes (pre) | `7b4336d40902e21c5f7db34a185a6c6f44f864db082f9664cc0443f89718bee2` | Reviewed; corrections required |
| [`artifacts/audit_phase11/phase11_4/phase11_4_selected_samples.csv`](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_4/phase11_4_selected_samples.csv) | 4,836 bytes | `80ec8b0af3dc5456e645e7e3b2a9dbb21f6648cf369fc59b42bf53c6af0b1dde` | Validated; 14 non-test samples verified |
| [`artifacts/audit_phase11/phase11_4/phase11_4_repeatability_results.json`](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_4/phase11_4_repeatability_results.json) | 46,915 bytes | `ab699a31478ca5d702b584fc7bd63f85d594cfd1402b1b541f78886a35fd583b` | Validated; 70 forward passes logged |
| [`artifacts/audit_phase11/phase11_4/phase11_4_probability_comparison.csv`](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_4/phase11_4_probability_comparison.csv) | 16,296 bytes | `3c1e9c84970a3753ad5883cca68cf3abce5395aa0d2dc83f34d53a249caab82c` | Validated; full 7-class vectors verified |
| [`artifacts/audit_phase11/phase11_4/phase11_4_checkpoint_integrity.json`](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_4/phase11_4_checkpoint_integrity.json) | 496 bytes | `8c74099299d8839f014995a1b02696556d0c08ccb829281b1356953d122a1035` | Validated; pre/post hash match confirmed |
| [`artifacts/audit_phase11/phase11_4/phase11_4_environment.json`](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_4/phase11_4_environment.json) | 591 bytes | `348f81f9cc8da36a9b219c6774c6753f9bb633caa44114045573d2b3d7ceacf5` | Validated; runtime metadata inspected |

---

## 3. Discrepancy Identification & Evidence Reconciliation

During the inspection of `reports/phase11_4_controlled_prediction_repeatability_audit.md` against primary evidence in `phase11_4_probability_comparison.csv` and `phase11_4_repeatability_results.json`, an evidence inconsistency was uncovered:

### Inconsistency Analysis
In Section 6 of the preliminary Phase 11.4 report, the summary table displaying mean softmax probabilities for samples 2 through 14 contained transcription offsets in the non-winning class columns. For example:
- **Sample 2 (`BBC00014.jpg`):**
  - Preliminary markdown table reported: Curl Virus `0.22%`, Healthy Leaf `1.01%`, Herbicide Damage `1.63%`, Leaf Hopper `1.77%`, Leaf Redding `0.83%`, Leaf Variegation `1.08%`.
  - Actual primary CSV/JSON record: Curl Virus `1.43%`, Healthy Leaf `0.99%`, Herbicide Damage `0.35%`, Leaf Hopper `2.02%`, Leaf Redding `0.44%`, Leaf Variegation `1.32%`.
  - True runner-up: Leaf Hopper Jassids at `2.02%`, yielding a Top-2 margin of `93.45% - 2.02% = +91.43 percentage points (pp)`.

### Resolution
The probability distribution table in Section 6 was regenerated directly from `phase11_4_probability_comparison.csv` using an automated verification script (`scratch/format_exact_table.py`). All 98 probability cells (14 samples $\times$ 7 classes) now strictly mirror the primary evidence down to the hundredth of a percent.

---

## 4. Detailed Scientific Corrections Made

### Correction 1: Selected-Sample Label Agreement vs. Model Accuracy
- **Preliminary Phrasing:** "Per-Sample Correctness on Known Splits (100.0%)", "14/14 correct; 100.0%".
- **Corrected Scientific Phrasing:** **"100% selected-sample label agreement (14/14)"**.
- **Scientific Justification:** The 14 evaluated images consisted of 7 training samples and 7 validation samples. Because the training samples were part of gradient optimization and the validation samples were used for model selection and early stopping, label agreement on this small cohort cannot be extrapolated as general model accuracy, validation accuracy, or external accuracy. The historical 97.8193% holdout benchmark is preserved as a metric retrieved from the earlier evaluation artifact.

### Correction 2: Statistical Methodology of Sample Selection
- **Preliminary Phrasing:** Selection presented without methodological qualification.
- **Corrected Scientific Phrasing:** Explicitly documented that choosing the first valid non-excluded sample per class from `train.csv` and `validation.csv` is a **deterministic selection protocol**, chosen solely to verify that the pipeline executes consistently across all 7 classes. It is not a randomized or statistically representative sample for measuring population-level accuracy.

### Correction 3: Probability Margin Units (Percentage Points)
- **Preliminary Phrasing:** Top-2 Margin reported with percent symbol (e.g., `+95.14%`, `+86.98%`).
- **Corrected Scientific Phrasing:** Top-2 Margin reported in **percentage points (pp)** (e.g., `+95.14 pp`, `+85.38 pp`).
- **Scientific Justification:** The margin between two percentages is an absolute arithmetic difference and must be denoted in percentage points ($p_1 - p_2$), not percent (which denotes a relative ratio $(p_1 - p_2)/p_2$).

### Correction 4: Model Input Channel Contract Check Scope
- **Preliminary Section Title:** "7. Pipeline Robustness & Exception Testing"
- **Corrected Section Title:** **"7. Model Input Channel Contract Check"**
- **Scientific Justification:** Supplying a 4-channel tensor `(1, 4, 224, 224)` directly to the PyTorch module tests whether the convolution stem enforces its 3-channel weight shape `[32, 3, 3, 3]`. It does **not** verify:
  1. FastAPI multipart upload parsing or file size limit enforcement.
  2. Image decoding error handling in PIL or OpenCV (corrupt headers, truncated JPEG streams).
  3. Structured HTTP 422/400 JSON error schemas in backend endpoints.
  4. Overall end-to-end application robustness.

### Correction 5: Determinism Terminology
- **Preliminary Phrasing:** "The mathematical guarantee that when the model evaluates identical input tensors... it yields the exact same logits..."
- **Corrected Scientific Phrasing:** **"Observed deterministic repeatability in the tested environment."**
- **Scientific Justification:** Repeatability was measured empirically on a specific platform (`cuda:0`, NVIDIA RTX 3050, PyTorch 2.15.0+cu132, CuDNN 92600). Floating-point arithmetic on GPUs can vary across different hardware architectures, compiler optimization levels, driver versions, or non-deterministic CUDA kernels. Claims of universal mathematical guarantees across all environments are scientifically invalid.

### Correction 6: Operating System Metadata Reconciliation
- **Discrepancy:** The Phase 11.4 environment report listed `Windows-10-10.0.26300-SP0`, while earlier project documentation identified `Windows 11`.
- **Forensic Investigation:**
  - Raw Windows NT Kernel Version: `10.0.26300.9974`
  - Current Build Number: `26300`
  - DisplayVersion: `26H2`
  - Build Branch: `ge_release` (Germanium)
  - Registry `ProductName`: `Windows 10 Home Single Language`
- **Root Cause & Reconciliation:** In Windows 11, Microsoft retains `ProductName = "Windows 10 ..."` in `HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion` for backwards compatibility with legacy application installers that parse that string. Python's `platform.platform()` reads this registry entry, emitting the string `Windows-10-10.0.26300-SP0`. However, in Microsoft's commercial versioning, all builds $\ge 22000$ are **Windows 11**. Build 26300 specifically corresponds to Windows 11 (26H2 / Insider preview). Both the raw string and commercial classification are now documented with this explanation.

---

## 5. Documentation Cryptographic Hashes (Before vs. After)

| Document | Pre-Review SHA-256 | Post-Review SHA-256 | Status |
| :--- | :--- | :--- | :---: |
| `reports/phase11_4_controlled_prediction_repeatability_audit.md` | `7b4336d40902e21c5f7db34a185a6c6f44f864db082f9664cc0443f89718bee2` | `9dce45be63b01e6440f003cd85f494f8c7ba94e3e1b6b64d784968c78e98be6b` | **RECONCILED** |
| `reports/phase11_4_1_scientific_reporting_correction.md` | *N/A (New File)* | Generated & Sealed | **CREATED** |
| `reports/MASTER_PROJECT_TASK_TRACKER.md` | `99e46a9e14a87265a6f87ca9f1d07c0f...` | Updated with Phase 11.4.1 | **UPDATED** |

---

## 6. Confirmed Repeatability Findings

Following complete reconciliation against primary evidence, the empirical repeatability findings remain fully confirmed:

| Metric | Measured Value | Scientific Interpretation |
| :--- | :---: | :--- |
| **Evaluated Images** | 14 samples | 2 per class (1 train, 1 validation; non-test) |
| **Consecutive Passes per Image** | 5 passes | 70 total forward passes executed |
| **Top-1 Prediction Consistency** | **100.0% (14/14)** | Zero flips in predicted class across all passes |
| **Maximum Confidence Drift** | **0.00000000** | Peak-to-peak variance = 0 across all 5 runs |
| **Maximum Probability Drift** | **0.00000000** | Zero variance across all 7 classes and 70 passes |
| **Selected-Sample Label Agreement** | **100.0% (14/14)** | Predictions match annotations on tested development samples |
| **Top-2 Decision Margin Range** | **+85.38 pp to +99.20 pp** | Min: `LR00001.jpg` (+85.38 pp); Max: `LV00002.jpg` (+99.20 pp) |

---

## 7. Checkpoint, Database & Test Set Invariance

Throughout Phase 11.4 and Phase 11.4.1, all system invariants were strictly maintained:

1. **Classification Checkpoint (`best_model.pt`):**
   - Authoritative SHA-256: `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e`
   - Current SHA-256: `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e` (**MATCH**)
2. **Detection Checkpoint (`best_faster_rcnn.pt`):**
   - Authoritative SHA-256: `01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2`
   - Current SHA-256: `01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2` (**MATCH**)
3. **Segmentation Checkpoint (`best_unet.pt`):**
   - Authoritative SHA-256: `612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43`
   - Current SHA-256: `612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43` (**MATCH**)
4. **Sealed Test Split:**
   - 0 of 321 test images accessed, loaded, or evaluated (**100% SEALED**).
5. **Production SQLite Database (`cotton_leaf_ai.db`):**
   - Predictions row count: **48** (zero writes)
   - Users row count: **5** (zero writes)
6. **Approval Gate 4:**
   - Preserved as **`PENDING_APPROVAL`**.

---

## 8. Final Audit Verdict

> **PHASE 11.4.1 VERDICT: FULLY AUDITED & SCIENTIFICALLY RECONCILED**  
> All reporting claims in Phase 11.4 have been rigorously reconciled against primary CSV/JSON evidence. Language has been corrected to distinguish observed repeatability from universal guarantees, label agreement on development samples from overall dataset accuracy, and convolutional tensor shape validation from end-to-end application robustness. All cryptographic checkpoints, sealed test datasets, and database tables remain pristine.

**Approval Gate 4:** `PENDING_APPROVAL`  
**Execution Status:** Halted. Ready for subsequent project directives.
