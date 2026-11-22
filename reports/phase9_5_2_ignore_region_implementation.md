# CottonGuard AI — Phase 9.5.2 Technical Report
## Faster R-CNN Tier 2 Ignore-Region Implementation and Synthetic Verification

**Date:** 2026-11-16  
**Project:** CottonGuard AI (`cotton_leaf_ai`)  
**Phase:** Phase 9.5.2 — Faster R-CNN Ignore-Region Handling  
**Author:** Autonomous Senior AI/ML & Computer Vision Engineer  
**Status:** **`IMPLEMENTED AND TESTED — READY FOR SEPARATE TRAINING REVIEW`**

---

## 1. Executive Summary

During Phase 9.5.1 independent audit, a critical design limitation in the default TorchVision Faster R-CNN ResNet-50 FPN V2 architecture was formally documented:
> *TorchVision's default Region Proposal Network (RPN) anchor matcher and RoI proposal matcher treat all image areas lacking explicit positive bounding boxes as negative background. Consequently, in images exhibiting broad, non-focal Tier 2 disease symptoms (such as diffuse anthocyanin blush in Leaf Redding, chlorosis ribbons in Jassids, and laminar curling in Curl Virus), the detector treats genuine disease symptoms as background, penalizing the model when it detects abnormal tissue.*

Under **Phase 9.5.2**, we designed, implemented, and rigorously verified a modular, non-destructive ignore-region masking framework for both the RPN and RoI box heads. The system was verified across **10 synthetic test cases** plus dedicated unit tests and gradient backpropagation checks (**13 / 13 tests passed, 100%**), without modifying any model checkpoints, production database, CVAT tasks, or sealed test datasets.

---

## 2. Mathematical Formulation & Overlap Policy

### 2.1 Compound Overlap Metric: $\text{IoU}$ and $\text{IoA}$
In plant pathology, diffuse disease symptoms often encompass large portions of the lamina (e.g., $200 \times 200\text{ px}$ to $500 \times 500\text{ px}$). When a small candidate anchor or proposal $B$ (e.g., $32 \times 32\text{ px}$, area $1,024\text{ px}$) is **100% submerged** within a large diffuse ignore region $I$ (e.g., $300 \times 300\text{ px}$, area $90,000\text{ px}$), standard Intersection-over-Union ($\text{IoU}$) produces:
$$\text{IoU}(B, I) = \frac{|B \cap I|}{|B \cup I|} = \frac{1,024}{1,024 + 90,000 - 1,024} = \frac{1,024}{90,000} \approx 0.0114$$

If an ignore filter relied solely on $\text{IoU} \ge 0.30$, this completely submerged candidate would **fail** the threshold ($0.0114 < 0.30$) and would erroneously be sampled as a **negative background example**, perpetuating the exact pathology identified in Phase 9.5.1.

To resolve this mathematical reality, Phase 9.5.2 implements a compound overlap metric combining $\text{IoU}$ with **Intersection-over-Box-Area ($\text{IoA}$)**:
$$\text{IoA}(B, I) = \frac{|B \cap I|}{|B|}$$
$$\text{OverlapScore}(B, I) = \max\left(\text{IoA}(B, I), \;\text{IoU}(B, I)\right)$$

### 2.2 Decision Rule & Thresholds
A candidate box $B_i$ is classified as substantially overlapping ignore region set $\mathcal{I} = \{I_1, \dots, I_M\}$ if:
$$\max_{j \in \{1,\dots,M\}} \text{OverlapScore}(B_i, I_j) \ge \tau_{\text{ignore}}$$

- **RPN Anchor Ignore Threshold ($\tau_{\text{RPN}}$):** **`0.30`** (Default). Configurable via `rpn_ignore_threshold`.
- **RoI Proposal Ignore Threshold ($\tau_{\text{RoI}}$):** **`0.30`** (Default). Configurable via `roi_ignore_threshold`.

---

## 3. Architecture & Implementation Approach

The implementation is modularized in `src/models/ignore_region_faster_rcnn.py` and cleanly integrated into `src/models/faster_rcnn.py` and `src/data/detection_dataset.py`.

```
+--------------------------------------------------------------------------------------------------+
|                                    TorchVision Faster R-CNN V2                                    |
+--------------------------------------------------------------------------------------------------+
       |
       +---> Backbone: ResNet-50 + FPN V2 (features)
       |
       +---> RPN: IgnoreAwareRPN (subclasses RegionProposalNetwork)
       |       * Base Anchor Matching:
       |           - Positive anchors (IoU >= 0.70 with focal GT): label = 1.0
       |           - Negative anchors (IoU < 0.30 with focal GT):  label = 0.0
       |           - Discarded anchors (0.30 <= IoU < 0.70):        label = -1.0
       |       * Ignore-Region Masking:
       |           - If anchor substantially overlaps Tier 2 ignore box AND label != 1.0:
       |                 label -> -1.0
       |       * Sampling (BalancedPositiveNegativeSampler):
       |           - Positives: label >= 1.0 (valid focal lesions preserved)
       |           - Negatives: label == 0.0 (ordinary background preserved)
       |           - label == -1.0 is completely omitted from both positive and negative samples!
       |       * Loss Computation:
       |           - Zero-anchor safety guard prevents NaN on empty sample edge cases.
       |
       +---> RoI Heads: IgnoreAwareRoIHeads (subclasses RoIHeads)
               * Ground Truth Augmentation: Proposals + GT boxes.
               * Base Proposal Matching:
                   - Positives (IoU >= 0.50 with focal GT): label in {1..6}
                   - Background candidates (IoU < 0.50):    label = 0
               * Ignore-Region Masking:
                   - If proposal substantially overlaps Tier 2 ignore box AND label <= 0:
                         label -> -1
               * Subsampling (BalancedPositiveNegativeSampler):
                   - label == -1 proposals are dropped from sampled_inds.
               * Loss Computation (fastrcnn_loss):
                   - class_logits & labels contain only valid positives and true negatives.
                   - Diffuse regions NEVER receive negative background classification penalty.
                   - Diffuse regions NEVER contribute to box regression loss.
                   - Zero-sample safety guard prevents NaN on empty sample edge cases.
```

### 3.1 Key Architectural Invariants
1. **Tier 1 Positive Priority:** If a candidate anchor or proposal matches a valid Tier 1 focal lesion ($\text{IoU} \ge 0.50 / 0.70$), its positive label is strictly maintained. The condition `~is_positive` prevents positive lesions from ever being masked to `-1`.
2. **Selective Negative Suppression:** Only anchors/proposals that would otherwise be classified as background (`label == 0`) and overlap Tier 2 regions are suppressed to `-1`.
3. **Pristine True Background:** Lamina and background regions outside Tier 2 ignore boundaries maintain `label == 0` and continue to supply true negative background supervision.
4. **Zero-Sample Autograd Safety:** If an extreme test fixture contains zero sampled anchors or proposals, the custom loss heads return `0.0 * tensor.sum()` preserving finite values and graph connectivity without triggering PyTorch zero-division or empty-cross-entropy NaNs.

---

## 4. Software Environment & Compatibility Notes

- **OS:** Windows 11 Pro (x86_64)
- **Python:** 3.11.9 (Virtual Environment: `d:\Main_Project\cotton_leaf_ai\.venv`)
- **PyTorch:** `2.15.0+cu132` (Verified CUDA & CPU support)
- **TorchVision:** `0.29.2+cu132`
- **Compatibility Note:** In TorchVision `0.29.2`, `RegionProposalNetwork` and `RoIHeads` employ `torchvision.models.detection._utils.BalancedPositiveNegativeSampler`. The sampler's internal logic explicitly defines:
  ```python
  positive = torch.where(matched_idxs_per_image >= 1)[0]
  negative = torch.where(matched_idxs_per_image == 0)[0]
  ```
  Consequently, assigning `-1` to candidate labels is TorchVision's native discard mechanism. Subclassing `assign_targets_to_anchors` and `select_training_samples` is fully aligned with TorchVision's upstream API design and introduces zero brittle monkey-patching or binary modifications.

---

## 5. Automated Test Suite Execution & Results

A dedicated test suite was created in `tests/models/test_ignore_region_faster_rcnn.py` containing **13 automated tests** covering all 10 synthetic test cases, unit-level RPN/RoI validations, and autograd stability.

### 5.1 Test Execution Matrix

| # | Test Case Identifier | Scenario / Fixture Description | Expected Behavior | Status |
|---|---|---|---|:---:|
| 1 | `test_box_overlap_metrics_iou_and_ioa` | Submerged $20\times20$ box inside $100\times100$ patch vs. outside box | $\text{IoA}=1.0$, $\text{IoU}=0.04$, outside $\text{IoA}=0.0$ | **`PASSED`** |
| 2 | `test_case_1_focal_alone` | Focal lesion $[30, 30, 80, 80]$, 0 ignore regions | All 4 losses finite, `loss_box_reg > 0` | **`PASSED`** |
| 3 | `test_case_2_diffuse_alone` | Diffuse patch $[40, 40, 160, 160]$, 0 focal boxes | Overlapping anchors masked to `-1`, `loss_box_reg == 0.0` | **`PASSED`** |
| 4 | `test_case_3_healthy_control` | Golden healthy leaf: 0 boxes, 0 ignore regions | `loss_box_reg == 0.0`, background supervised, no crash | **`PASSED`** |
| 5 | `test_case_4_mixed_focal_and_diffuse` | Co-occurring focal $[20, 20, 50, 50]$ & diffuse $[100, 100, 180, 180]$ | Focal trained, diffuse masked, all 4 losses active | **`PASSED`** |
| 6 | `test_case_5_multiple_non_overlapping` | 3 distinct non-overlapping diffuse ribbons | All diffuse ribbons masked, outside background active | **`PASSED`** |
| 7 | `test_case_6_ignore_overlapping_positive` | Ignore box $[20, 20, 100, 100]$ encloses focal $[30, 30, 70, 70]$ | Focal priority preserved (`~is_positive`), regression active | **`PASSED`** |
| 8 | `test_case_7_no_disease_annotations` | Target dictionary omitting `"ignore_boxes"` key | Graceful backward-compatible fallback, finite losses | **`PASSED`** |
| 9 | `test_case_8_empty_detection_target` | Explicit empty tensors `(0, 4)` for boxes & ignore boxes | Zero crashes, finite losses | **`PASSED`** |
| 10 | `test_case_9_proposals_outside_ignore` | Proposals completely outside ignore region | Retain `label == 0`, sampled into negative indices | **`PASSED`** |
| 11 | `test_case_10_proposals_inside_ignore` | Proposals submerged in ignore region | Masked to `label == -1`, 0 sampled into negatives | **`PASSED`** |
| 12 | `test_rpn_anchor_ignore_masking` | RPN unit test: anchor inside, anchor outside, anchor focal | Inside: `-1.0`, Outside: `0.0`, Focal: `1.0` | **`PASSED`** |
| 13 | `test_gradient_backprop_stability` | Multi-image batch forward + backward pass | Backward succeeds, all parameter gradients finite | **`PASSED`** |

**Execution Result:**
```
tests/models/test_ignore_region_faster_rcnn.py::test_box_overlap_metrics_iou_and_ioa PASSED
tests/models/test_ignore_region_faster_rcnn.py::test_case_1_focal_alone PASSED
tests/models/test_ignore_region_faster_rcnn.py::test_case_2_diffuse_alone PASSED
tests/models/test_ignore_region_faster_rcnn.py::test_case_3_healthy_control PASSED
tests/models/test_ignore_region_faster_rcnn.py::test_case_4_mixed_focal_and_diffuse PASSED
tests/models/test_ignore_region_faster_rcnn.py::test_case_5_multiple_non_overlapping_ignore_regions PASSED
tests/models/test_ignore_region_faster_rcnn.py::test_case_6_ignore_region_overlapping_positive_detection PASSED
tests/models/test_ignore_region_faster_rcnn.py::test_case_7_no_disease_annotations PASSED
tests/models/test_ignore_region_faster_rcnn.py::test_case_8_empty_detection_target PASSED
tests/models/test_ignore_region_faster_rcnn.py::test_case_9_proposals_outside_ignore_regions PASSED
tests/models/test_ignore_region_faster_rcnn.py::test_case_10_proposals_inside_ignore_regions PASSED
tests/models/test_ignore_region_faster_rcnn.py::test_rpn_anchor_ignore_masking PASSED
tests/models/test_ignore_region_faster_rcnn.py::test_gradient_backprop_stability PASSED
======================= 13 passed, 1 warning in 39.36s =======================
```

### 5.2 Full Regression Test Suite Execution
To verify that existing models, datasets, and pipelines remain 100% operational, the full model test suite was executed:
- `tests/models/test_ignore_region_faster_rcnn.py` (13 tests)
- `tests/models/test_phase4_ai_integration.py` (4 tests)
- `tests/models/test_phase6_annotations.py` (5 tests)
- `tests/models/test_phase9_2a_detector.py` (6 tests)
- `tests/models/test_phase9_2b_2_training_implementation.py` (8 tests)
- `tests/test_segmentation_severity_recommendations.py` (6 tests)
- `tests/test_classifier.py` (5 tests)
- `tests/test_data_pipeline.py` (6 tests)

**Result: 53 / 53 PASSED (100% pass rate).**

---

## 6. U-Net Segmentation Loader Compatibility Verification

An independent inspection and audit of the segmentation data pipeline was conducted:
1. **Mask Integrity:** All 50 mask files in `data/annotations/masks/*.png` were inspected.
   - Unique pixel values across all 50 masks: strictly **`{0, 1, 2}`**.
   - `0 = Background`, `1 = Healthy Leaf Lamina`, `2 = Diseased Tissue`.
   - Golden healthy controls (e.g., `HL00031.png`, `HL00087.png`): strictly `{0, 1}`. Value `2` is strictly absent.
2. **Discrete Nearest-Neighbor Resampling:**
   - In `scripts/train_segmentation.py` (`PairedSegmentationDataset`), line 55:
     ```python
     mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
     ```
   - Discrete integer class indices `{0, 1, 2}` are preserved with zero continuous blurring artifacts.
3. **Pipeline Independence:**
   - Faster R-CNN ignore-region handling operates strictly on detection candidate tensors and does not modify, write to, or access segmentation masks. U-Net training and evaluation are completely unaffected.

---

## 7. Safety, Invariant & Checkpoint Integrity Audit

In compliance with the mandatory safety directives:
1. **Model Training:** Strictly **ZERO** training runs or training epochs were executed.
2. **Model Checkpoint Integrity:** All existing production checkpoints were verified via SHA-256 cryptographic hashing. Not a single byte was altered:
   - `artifacts/checkpoints/classification/best_model.pt`: `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e` (48,685,496 bytes) — **VERIFIED UNCHANGED**
   - `artifacts/checkpoints/detection/best_faster_rcnn.pt`: `01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2` (345,797,103 bytes) — **VERIFIED UNCHANGED**
   - `artifacts/checkpoints/segmentation/best_unet.pt`: `612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43` (97,921,803 bytes) — **VERIFIED UNCHANGED**
3. **Database Integrity:** The SQLite database `cotton_guard.db` was not modified.
4. **CVAT Task Integrity:** CVAT Tasks #4, #5, and #6 were not modified. Zero CVAT synchronizations were performed.
5. **Sealed Test Set Isolation:**
   - `artifacts/manifests/splits/test.csv`: `6025d86f89b9ec4d3f95ad4a54f7d633478c9eb152c67057dfb021d8c00bdd9d` (110,597 bytes) — **VERIFIED UNCHANGED (ZERO DATASET LEAKAGE)**.

---

## 8. Files Inspected & Modified Summary

### Files Inspected:
- `src/models/faster_rcnn.py`
- `src/data/detection_dataset.py`
- `scripts/train_segmentation.py`
- `scripts/train_faster_rcnn.py`
- `reports/phase9_5_1_final_independent_audit.md`
- `docs/ANNOTATION_STANDARD_V3.md`
- `data/annotation_pilot/phase9_5/candidate_annotations_task6.json`
- `artifacts/manifests/consolidated_100_annotation_manifest.csv`

### Files Created:
- [src/models/ignore_region_faster_rcnn.py](file:///D:/Main_Project/cotton_leaf_ai/src/models/ignore_region_faster_rcnn.py) (Modular RPN and RoI ignore-region classes, overlap functions, and model builder).
- [tests/models/test_ignore_region_faster_rcnn.py](file:///D:/Main_Project/cotton_leaf_ai/tests/models/test_ignore_region_faster_rcnn.py) (13 automated test cases covering all 10 synthetic requirements).
- [reports/phase9_5_2_ignore_region_implementation.md](file:///D:/Main_Project/cotton_leaf_ai/reports/phase9_5_2_ignore_region_implementation.md) (This formal report).

### Files Modified:
- [src/models/faster_rcnn.py](file:///D:/Main_Project/cotton_leaf_ai/src/models/faster_rcnn.py) (Integrated `equip_faster_rcnn_with_ignore_regions` with `enable_ignore_regions` flag).
- [src/data/detection_dataset.py](file:///D:/Main_Project/cotton_leaf_ai/src/data/detection_dataset.py) (Added `"ignore_boxes"` tensor to target dictionary with safe default `torch.zeros((0, 4))`).

---

## 9. Known Limitations & Remaining Risks

1. **Threshold Hyperparameter Tuning:**
   The default overlap threshold is set to $\tau = 0.30$. While this value cleanly captures diffuse ribbons and blush in synthetic testing, empirical ablation across the consolidated 70-image training set during actual training review will reveal whether $\tau = 0.30$ or $\tau = 0.50$ yields optimal mAP.
2. **Anchor Granularity vs. Small Diffuse Punctures:**
   In cases where diffuse symptoms take the form of extremely fine, scattered stippling (rather than continuous patches), bounding box approximation of the diffuse region may enclose healthy lamina. The compound $\text{IoA}$ metric ensures small anchors inside the bounding box are ignored, but precise polygon rasterization could be explored in future research.

---

## 10. Final Authorization Gate Determination

**Gate Outcome:**
### **`A. IMPLEMENTED AND TESTED — READY FOR SEPARATE TRAINING REVIEW`**

- Faster R-CNN RPN anchor ignore masking is fully implemented and tested.
- Faster R-CNN RoI proposal ignore masking is fully implemented and tested.
- Valid focal lesion priority is strictly maintained.
- All 10 synthetic test cases are implemented and 100% passing.
- U-Net compatibility and `{0, 1, 2}` mask values are verified.
- Existing model checkpoints, database, and sealed test splits remain untouched.
- **NO MODEL TRAINING HAS BEEN STARTED.** Awaiting explicit user approval before proceeding to any model training phase.
