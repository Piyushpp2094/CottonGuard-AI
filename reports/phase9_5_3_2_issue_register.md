# CottonGuard AI — Phase 9.5.3.2 Issue Register
## Autonomous Final Detector Configuration Audit & Safe Corrections

**Date:** 2026-11-17  
**Project:** CottonGuard AI (`cotton_leaf_ai`)  
**Phase:** Phase 9.5.3.2 — Detector Configuration Audit & Issue Detection  
**Author:** Autonomous Senior AI/ML & Computer Vision Engineer  
**Status:** All 8 identified issues audited, prioritized, safely resolved, or bounded. Zero unauthorized training executed.

---

### Issue Severity Summary

| Severity | Count | Resolved / Bounded | Blocking Training |
| :--- | :---: | :---: | :---: |
| **CRITICAL** | 2 | 2 Resolved | 0 |
| **HIGH** | 4 | 4 Resolved / Documented | 0 |
| **MEDIUM** | 1 | 1 Resolved | 0 |
| **LOW** | 1 | 1 Resolved | 0 |
| **Total** | **8** | **8** | **0** |

---

### Detailed Prioritized Issue Register

#### ISSUE-01-OPT-HARDCODED
- **Severity:** `HIGH`
- **Component / File:** `scripts/train_faster_rcnn.py` (Line 401)
- **Actual Evidence:** `optimizer = torch.optim.SGD(...)` was hardcoded, completely ignoring `cfg["training"]["optimizer"]["type"]`. Running with an AdamW configuration would silently execute SGD with $lr = 0.0001$ (far too low for SGD) or fail to engage adaptive momentum estimation ($\beta_1, \beta_2$).
- **Root Cause:** The training script was originally adapted from the TorchVision detection reference pipeline, which hardcodes SGD.
- **Proposed Correction:** Update `train_faster_rcnn.py` to inspect `cfg["training"]["optimizer"]["type"]` and instantiate `torch.optim.AdamW` with configurable learning rate, betas, and weight decay when `"ADAMW"` is specified.
- **Correction Status:** **RESOLVED.** Code updated to support dynamic optimizer dispatch for both AdamW and SGD.
- **Verification Result:** Verified via `scripts/train_faster_rcnn.py --config configs/phase9_5_faster_rcnn_retrain_config.yaml --dry-run-check` which successfully initialized AdamW with $lr=1.0 \times 10^{-4}$ and $\beta=(0.9, 0.999)$.

---

#### ISSUE-02-DATASET-STAGING-MISSING
- **Severity:** `CRITICAL`
- **Component / File:** `data/detection/` vs. `src/data/detection_dataset.py` (Lines 73–74)
- **Actual Evidence:** `data/detection/` contained only 50 images and 50 labels from Tasks 4 and 5. Task 6 images were isolated in `data/pilot_batch_03/images/` and annotations were stored in `candidate_annotations_task6.json`. Pointing `CottonDetectionDataset` to `artifacts/manifests/consolidated_100_annotation_manifest.csv` would raise immediate `FileNotFoundError`.
- **Root Cause:** Staging of the 100-image dataset was intentionally deferred in earlier phases to avoid mutating the legacy Phase 9.2 dataset directory.
- **Proposed Correction:** Create a non-destructive dataset staging script (`scripts/prepare_consolidated_100_detection_data.py`) to construct an isolated directory `data/detection_consolidated_100/` containing:
  - `images/`: 100 images (50 from Tasks 4/5, 50 from Task 6).
  - `labels/`: 100 YOLO `.txt` files containing exactly 1,009 focal lesion boxes.
  - `ignore_labels/`: 100 ignore `.txt` files containing exactly 130 diffuse ignore boxes.
- **Correction Status:** **RESOLVED.** Staging script executed; `data/detection_consolidated_100/` successfully created.
- **Verification Result:** Verified via `CottonDetectionDataset` test. All 70 train samples (683 focal, 62 ignore) and 30 validation samples (326 focal, 68 ignore) load with 100% precision and zero missing files. Legacy `data/detection/` remains 100% untouched.

---

#### ISSUE-03-DATASET-ASSERTIONS-HARDCODED
- **Severity:** `HIGH`
- **Component / File:** `scripts/train_faster_rcnn.py` (Lines 358–359)
- **Actual Evidence:** `assert len(train_ds) == 35` and `assert len(val_ds) == 15` were hardcoded for the legacy 50-image split. Running on the 100-image dataset (70 train / 30 val) immediately triggered an `AssertionError`.
- **Root Cause:** Legacy sample counts from Phase 9.2A were hardcoded in the script body instead of reading from configuration.
- **Proposed Correction:** Update assertions to read `train_samples` and `val_samples` dynamically from configuration:
  ```python
  expected_train = int(ds_cfg.get("train_samples", 35))
  expected_val = int(ds_cfg.get("val_samples", 15))
  assert len(train_ds) == expected_train
  assert len(val_ds) == expected_val
  ```
- **Correction Status:** **RESOLVED.** Parameterized in `train_faster_rcnn.py`.
- **Verification Result:** Verified via `--dry-run-check` confirming 70 train images and 30 validation images pass assertion cleanly.

---

#### ISSUE-04-MODEL-BUILDER-IGNORE-FLAG
- **Severity:** `HIGH`
- **Component / File:** `scripts/train_faster_rcnn.py` (Lines 393–398)
- **Actual Evidence:** `train_faster_rcnn.py` invoked `build_faster_rcnn_model()` without passing `enable_ignore_regions=True`, `rpn_ignore_threshold`, or `roi_ignore_threshold`. As a result, the model was constructed with standard TorchVision RPN and RoIHeads, omitting Tier 2 ignore masking.
- **Root Cause:** The ignore-region classes (`IgnoreAwareRPN`, `IgnoreAwareRoIHeads`) were implemented in `src/models/ignore_region_faster_rcnn.py`, but the caller in `train_faster_rcnn.py` had not wired the YAML parameters to the factory.
- **Proposed Correction:** Wire `enable_ignore_regions`, `rpn_ignore_threshold`, and `roi_ignore_threshold` from `cfg["model"]` into `build_faster_rcnn_model()`.
- **Correction Status:** **RESOLVED.** Wired dynamically in `train_faster_rcnn.py`.
- **Verification Result:** Verified in `--dry-run-check` and unit tests confirming `IgnoreAwareRPN` and `IgnoreAwareRoIHeads` are equipped when `enable_ignore_regions: true`.

---

#### ISSUE-05-AMP-SKIPPED-STEPS-UNTRACKED
- **Severity:** `MEDIUM`
- **Component / File:** `scripts/train_faster_rcnn.py` (Lines 530–535)
- **Actual Evidence:** While scheduler advancement was correctly gated on `scale_after >= scale_before`, skipped steps were not explicitly counted, logged, or recorded in epoch history.
- **Root Cause:** Missing counter instrumentation for AMP skipped steps.
- **Proposed Correction:** Add `epoch_skipped_steps` and `total_skipped_steps` tracking in the training loop and record them in `epoch_record`.
- **Correction Status:** **RESOLVED.** Tracking instrumented in `train_faster_rcnn.py`.
- **Verification Result:** Verified in `scripts/test_amp_step_detection.py` confirming counter accurately records skipped steps when gradients contain NaN.

---

#### ISSUE-06-CHECKPOINT-OVERWRITE-RISK
- **Severity:** `CRITICAL`
- **Component / File:** `configs/phase9_2a_detector_config.yaml` (Line 107)
- **Actual Evidence:** Legacy configuration specified `best_checkpoint_path: artifacts/checkpoints/detection/best_faster_rcnn.pt`. Running training would trigger `FileExistsError` or overwrite the production baseline.
- **Root Cause:** Legacy configuration was authored before the isolated retrain directory standard was established.
- **Proposed Correction:** Create `configs/phase9_5_faster_rcnn_retrain_config.yaml` specifying isolated destination:
  - `best_checkpoint_path: artifacts/checkpoints/detection/phase9_5_retrain/best_faster_rcnn_ignore.pt`
  - `latest_checkpoint_path: artifacts/checkpoints/detection/phase9_5_retrain/latest_faster_rcnn_ignore.pt`
- **Correction Status:** **RESOLVED.** Configuration file created and validated.
- **Verification Result:** Verified in `--dry-run-check` that output directory resolves cleanly to `phase9_5_retrain` and baseline checkpoint `artifacts/checkpoints/detection/best_faster_rcnn.pt` (SHA-256: `01d82b32...`) remains 100% protected.

---

#### ISSUE-07-CLASS-MAPPING-AMBIGUOUS-COMMENTS
- **Severity:** `LOW`
- **Component / File:** `reports/phase9_5_3_1_final_training_authorization_preflight.md` (Line 297)
- **Actual Evidence:** Comments in previous reports described class 0 as `(0: healthy/background)`.
- **Root Cause:** Conflation of Healthy Leaf negative-control images with background class label 0.
- **Proposed Correction:** Eradicate ambiguous comments. Class 0 is strictly **Background** (non-lesion context). Healthy Leaf is a negative-control image containing 0 foreground bounding boxes.
- **Correction Status:** **RESOLVED.** Standardized across all code, configuration, and documentation.
- **Verification Result:** Verified in `configs/phase9_5_faster_rcnn_retrain_config.yaml` and `scripts/test_metric_mapping.py`.

---

#### ISSUE-08-BACKEND-TESTCLIENT-LOOPBACK-HANG
- **Severity:** `HIGH` (Environment Boundary Constraint)
- **Component / File:** `tests/backend/test_phase3_backend.py` (Lines 48–49)
- **Actual Evidence:** Pytest process hung in `setUpClass()` at `TestClient(app)` / `TestClient.__enter__()` for >28 minutes.
- **Root Cause:** Windows terminal security sandbox blocks localhost socket connect handshakes with `[WinError 10013]`. Starlette's `TestClient` uses `anyio.from_thread.start_blocking_portal()` which hangs indefinitely waiting for loopback communication.
- **Proposed Correction / Resolution:**
  1. Cancelled the hung background process `task-8880`.
  2. Bounded diagnostic probe proved that `connect(('127.0.0.1', port))` raises `WinError 10013` inside the sandbox.
  3. Established execution boundary: All model, data pipeline, and unit tests (56 tests) run cleanly inside the sandbox. Backend web integration tests (`TestClient`) require execution outside the sandbox (`BypassSandbox: true` or native PowerShell) where Windows loopback socket permissions are permitted.
- **Correction Status:** **BOUNDED & DOCUMENTED.** Process cancelled; environmental boundary verified.
- **Verification Result:** 36/36 model tests and 20/20 pipeline tests (56 total tests) pass with 100% success inside the sandbox.
