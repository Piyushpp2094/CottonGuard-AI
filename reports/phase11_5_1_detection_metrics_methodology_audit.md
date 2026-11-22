# CottonGuard AI — Phase 11.5.1: Detection Metrics Methodology Audit & Scientific Reconciliation Report

**Date:** November 17, 2026  
**Auditor Role:** Senior Machine Learning Engineer, Computer Vision Engineer and Independent AI Model Auditor  
**Project:** CottonGuard AI (`D:\Main_Project\cotton_leaf_ai`)  
**Phase:** 11.5.1 — Detection Metrics Methodology Audit & Scientific Reconciliation  
**Audit Status:** `COMPLETED`  
**Detection Checkpoint:** `artifacts/checkpoints/detection/best_faster_rcnn.pt`  
**Authoritative SHA-256:** `01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2` (Verified Match: **YES**)  
**Sealed Test Split Protection:** 100% Preserved (0/321 classifier test images accessed, loaded, or evaluated)  
**Database State:** 100% Preserved (48 predictions rows, 5 users rows; zero writes)  
**Approval Gate 4:** `PENDING_APPROVAL` (Strictly Preserved)  

---

## 1. Executive Summary

This report delivers a forensic audit of the evaluation methodology, metric computation pipelines, and mathematical assumptions utilized during **Phase 11.5** for the **Faster R-CNN ResNet-50 FPN V2** object detector (`best_faster_rcnn.pt`).

The audit was conducted strictly on existing source code, raw predictions, ground-truth annotations, and evaluation records to verify whether reported metrics ($\text{mAP}_{50}$, $\text{mAP}_{50:95}$, Precision, Recall, F1, and ignore-region absorption) adhere to formal computer vision evaluation conventions and plant pathology clinical realities.

### Key Audit Findings & Reconciliations

1. **AP Implementation Convention (VOC-Style Continuous Integration):**
   - The project's AP implementation in `src/training/detection_losses.py` calculates the **exact area under the monotonically decreasing precision curve** across all distinct recall step increments (PASCAL VOC 2010–2012 continuous AUC convention).
   - This metric is evaluated at 10 IoU thresholds $[0.50:0.05:0.95]$ (the standard IoU range used by MS COCO).
   - **Critical Convention Clarification:** This is **not** official MS COCO 101-point interpolated AP (`pycocotools`). It is formally classified as **VOC-style continuous interpolated AP evaluated across the COCO IoU threshold range**.
2. **Native Confidence Floor ($T_{\text{native}} = 0.05$):**
   - The detector's TorchVision RoI box heads enforce a native classification cutoff of `roi_heads.score_thresh = 0.05`.
   - The 1,130 candidate proposals stored in `phase11_5_detection_predictions.json` have a minimum observed confidence of **0.050082** and a maximum of **0.997260**.
   - Standard project AP is calculated strictly over the candidate predictions emitted under this native cutoff. Proposals with confidence $< 0.05$ were pruned internally by the detector during inference.
3. **Resolution of Threshold-Dependent mAP Variance:**
   - In Phase 11.5, different $\text{mAP}_{50}$ values were reported at $T=0.05$ (13.94%), $T=0.25$ (11.81%), and $T=0.40$ (10.39%).
   - Forensic inspection revealed that `evaluate_cohort()` in `scratch/run_phase11_5_complete_audit.py` filtered predictions by `confidence >= score_thresh` *prior* to curve integration.
   - Truncating lower-confidence true positives capped maximum recall at lower levels (31.47% at $T=0.25$, 27.97% at $T=0.40$), artificially reducing the integrated area.
   - Standard AP is ranking-based and threshold-free across the native proposal set. The values at $T=0.25$ and $T=0.40$ are formally reclassified as **Threshold-Truncated Average Precision ($\text{AP}@T$)**.
   - Standard project $\text{mAP}_{50}$ is **13.94%** for Cohort A and **5.16%** for Cohort B (Tier 1 only).
4. **Ignore-Region Terminology & Clinical Provenance:**
   - **Cohort B Tier 1 Only:** Formally designated as **Standard project AP without ignore suppression** ($\text{mAP}_{50} = 5.16\%$).
   - **Cohort B Class-Agnostic Ignore:** Formally designated as **Custom ignore-aware AP** ($\text{mAP}_{50} = 5.39\%$).
   - **Cohort B Class-Specific Ignore:** Formally designated as **Custom class-specific ignore-aware AP** ($\text{mAP}_{50} = 5.28\%$).
   - **Focal Tier 1 Protection:** Zero focal ground-truth matches were suppressed ($0$ conflicts).
   - **Diagnosis Provenance & Limitation:** Class-specific ignore suppression derives its disease identity from `class_label` in `artifacts/manifests/consolidated_100_annotation_manifest.csv`. This represents a single global image-level diagnosis. If a leaf exhibits mixed-pathogen infections or secondary symptoms, unverified polygon-level ignore regions cannot be independently confirmed.
5. **Cohort Relationship & Nesting:**
   - Cohort A (15 images, 143 boxes) is a **proper subset nested directly inside Cohort B** (30 images, 326 Tier 1 boxes, 68 Tier 2 ignore regions).
   - Cohort B adds 15 newly annotated validation images from Batch 03.
   - They are **nested cohorts**, not independent datasets. Both cohorts maintain 0 sample-level and 0 group-level overlap with the 35 detector training images.
   - Cohort B is an **extended disjoint validation cohort relative to detector training data**; it does **not** evaluate out-of-distribution or external multi-site clinical generalization.
6. **Inference Throughput & Latency Interpretation:**
   - The measured 172.15 ms average latency (~5.8 FPS on RTX 3050 Laptop GPU) represents **isolated model forward-pass GPU execution**. It does not represent full end-to-end web application response time (which includes multipart decoding, EXIF sanitization, U-Net inference, and SQLite logging), nor does VRAM headroom guarantee multi-tenant concurrency.

---

## 2. Source Files & Evaluation Functions Inspected

The following implementation and artifact files were forensically reviewed:

| Source File / Artifact | Location | Existence Status | Operational Purpose & Findings |
| :--- | :--- | :---: | :--- |
| `scripts/evaluate_faster_rcnn.py` | `scripts/` | **`MISSING`** | Recorded honestly as missing. Historical evaluation was conducted via `scripts/train_faster_rcnn.py` validation loop. |
| `scratch/run_phase11_5_complete_audit.py` | `scratch/` | **`VERIFIED`** | Primary execution script for Phase 11.5. Contained the score-threshold pre-filtering step in `evaluate_cohort()`. |
| `src/training/detection_losses.py` | `src/training/` | **`VERIFIED`** | Contains `_box_iou_np()`, `_compute_ap()`, and `compute_detection_metrics()`. Implements VOC-style continuous AP. |
| `src/models/ignore_region_faster_rcnn.py` | `src/models/` | **`VERIFIED`** | Implements `compute_box_overlap_metrics()` and compound $\max(\text{IoA}, \text{IoU})$ criteria. |
| `scripts/train_faster_rcnn.py` | `scripts/` | **`VERIFIED`** | Training script from Phase 9.2C. Confirmed native `roi_heads.score_thresh = 0.05` produces historical 13.94% $\text{mAP}_{50}$. |
| `artifacts/audit_phase11/phase11_5/phase11_5_detection_predictions.json` | `artifacts/` | **`VERIFIED`** | All 1,130 raw model candidate predictions across 30 images, bounded by native confidence $\ge 0.05008$. |

---

## 3. AP Convention, Native Confidence Floor & Threshold Handling

### 3.1 AP Implementation Convention (VOC-Style Continuous AUC)
The project computes Average Precision in `src/training/detection_losses.py` via `_compute_ap()`:
```python
def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Computes exact area under the monotonically interpolated precision-recall envelope."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
```
- **Mathematical Procedure:** This algorithm monotonically smoothes the precision curve ($p_{\text{interp}}(r) = \max_{r' \ge r} p(r')$) and computes the exact area under the curve across all distinct recall step increments:
  $$\text{AP}_{\text{VOC}} = \sum_{k=0}^{n} (r_{k+1} - r_k) \cdot p_{\text{interp}}(r_{k+1})$$
- **Distinction from Official COCO Evaluation:** Official MS COCO evaluation (`pycocotools.cocoeval.COCOeval`) computes Average Precision as the arithmetic mean of interpolated precision sampled at **101 discrete recall thresholds**:
  $$\text{AP}_{\text{COCO}} = \frac{1}{101} \sum_{r \in \{0.00, 0.01, \dots, 1.00\}} p_{\text{interp}}(r)$$
  Because the project integrates the continuous curve rather than sampling at 101 points, it is formally characterized as **VOC-style continuous interpolated AP evaluated across the COCO IoU threshold range $[0.50:0.05:0.95]$**. Official COCO evaluation metrics are not available and are not claimed.

### 3.2 Native Confidence Floor
TorchVision's Faster R-CNN RoI heads enforce a native classification score threshold:
- **Architecture Config:** `model.roi_heads.score_thresh = 0.05`
- **Observed Candidate Pool:** In `phase11_5_detection_predictions.json`, exactly 1,130 predictions exist across the 30 images. Minimum confidence is **0.050082**, and maximum confidence is **0.997260**.
- **Definition of Standard Project AP:** Standard project AP is evaluated over the candidate prediction set emitted above this native $0.05$ cutoff. It does not evaluate candidates with confidence $< 0.05$, as those were discarded internally during inference post-processing.

### 3.3 Root Cause of Threshold-Dependent mAP in Phase 11.5
In `scratch/run_phase11_5_complete_audit.py`, `evaluate_cohort()` applied:
```python
filt_preds = [p for p in r["predictions"] if p["confidence"] >= score_thresh]
```
When `score_thresh = 0.05`, all 528 candidate detections in Cohort A were evaluated, allowing recall to reach its native peak ($38.46\%$), resulting in $\text{mAP}_{50} = 13.94\%$.  
When `score_thresh = 0.25`, only 204 detections were evaluated; recall was truncated at $31.47\%$. Area between $0.3147$ and $1.0$ recall became zero, truncating AP to $11.81\%$.  
When `score_thresh = 0.40`, only 132 detections were evaluated; recall was truncated at $27.97\%$, truncating AP to $10.39\%$.

```
Precision
  1.0 |-----\
      |      \     Standard PR Curve (Evaluates all candidate proposals)
      |       \-------
      |               \------- [T=0.40 Cutoff: Recall capped at 27.97%]
      |                       \------- [T=0.25 Cutoff: Recall capped at 31.47%]
      |                               \------- [T=0.05 Native Limit: Recall capped at 38.46%]
  0.0 +--------------------------------------------------------------------> Recall
      0.0                    0.28    0.31     0.38                    1.0
```

- **Reclassification:** These values are officially designated as **Threshold-Truncated Average Precision ($\text{AP}@T$)**. Standard project $\text{mAP}_{50}$ is uniquely **13.94%**.

---

## 4. IoU Matching Implementation Audit

The ground-truth to prediction matching logic in `src/training/detection_losses.py` was audited against the following criteria:

| Evaluation Criterion | Implementation Rule | Audit Verdict | Evidence / Code Location |
| :--- | :--- | :---: | :--- |
| **Class Segregation** | Predictions of class $c$ only match ground-truth of class $c$ | **COMPLIANT** | `c_boxes = gt_boxes[gt_labels == c]` |
| **Image Isolation** | Predictions only match targets on the same image index | **COMPLIANT** | `img_idx` grouped in dictionary `class_gts[img_idx]` |
| **Greedy Priority** | Highest-scoring prediction matches first | **COMPLIANT** | `class_preds.sort(key=lambda item: item[0], reverse=True)` |
| **One-to-One Match** | Maximum one GT box per prediction | **COMPLIANT** | `matched[img_idx][best_j] = True` prevents reuse |
| **Duplicate Penalty** | Subsequent overlapping predictions scored as FP | **COMPLIANT** | `if not matched: ... else: fp[i] = 1.0` |
| **Unmatched GT Penalty** | Ground-truth boxes lacking a match scored as FN | **COMPLIANT** | `total_fn = total_gt - total_tp` |
| **Healthy Control Safety** | Zero target boxes; all predictions scored as FP | **COMPLIANT** | `gt_boxes.shape = (0, 4)` forces `fp[i] = 1.0` |

---

## 5. Ignore-Region Evaluation & Clinical Diagnosis Provenance

### 5.1 Formulation and Compound Threshold
The ignore-region mechanism uses a compound criterion combining Intersection-over-Area ($\text{IoA}$) and Intersection-over-Union ($\text{IoU}$):
$$\text{IoA}(P, I) = \frac{|P \cap I|}{|P|}, \quad \text{IoU}(P, I) = \frac{|P \cap I|}{|P \cup I|}$$
$$\text{Suppressed if } \max\left(\max_j \text{IoA}(P, I_j), \; \max_j \text{IoU}(P, I_j)\right) \ge 0.30$$

This formulation prevents small candidate proposals completely submerged inside large diffuse ignore regions from being penalized as negative background false positives.

### 5.2 Forensic Verification & Diagnosis Provenance

1. **Suppression Counts Across Thresholds:**
   - Candidate pool ($T=0.05$): **141 predictions** excluded from false-positive accounting.
   - Screening threshold ($T=0.25$): **48 predictions** excluded from false-positive accounting.
   - High-precision threshold ($T=0.40$): **32 predictions** excluded from false-positive accounting.
2. **Protection of Focal Tier 1 Ground-Truth Matches:**
   - Every one of the 141 suppressed candidate predictions was audited against all Tier 1 ground-truth boxes in its image.
   - **Tier 1 Conflicts: Exactly 0.** Zero True Positive detections were suppressed by ignore regions.
3. **Diagnosis Provenance for Class-Specific Ignore Suppression:**
   - In `phase11_5_detection_predictions.json`, image diagnosis is populated from `class_label` in `artifacts/manifests/consolidated_100_annotation_manifest.csv`.
   - **Clinical Limitation:** `class_label` is an image-level primary diagnosis. In field agriculture, cotton leaves can exhibit co-infections (e.g. secondary chlorosis or mixed fungal/bacterial blight). Class-specific ignore suppression assumes that all ignore regions on that leaf belong to the primary diagnosis. If an ignore region covers a secondary symptom, this heuristic cannot independently confirm the polygon-level pathogen.
4. **Impact of Class-Agnostic vs. Class-Specific Policy at $T=0.25$:**
   - **Class-Agnostic Ignore Suppression (Custom Ignore-Aware AP):**
     - Excused 48 predictions from FP accounting: 27 matched the leaf's primary diagnosis, while 21 were cross-class hallucinations (e.g., predicting Bacterial Blight or Variegation on a Leaf Redding ignore polygon on `LR00003.jpg`).
     - Operational Precision: **11.86%** | FP: **379** | $\text{mAP}_{50}$: **5.39%**.
   - **Class-Specific Ignore Suppression (Custom Class-Specific Ignore-Aware AP):**
     - Excused only the 27 disease-matched predictions; the 21 cross-class hallucinations correctly remain False Positives.
     - Operational Precision: **11.31%** | FP: **400** | $\text{mAP}_{50}$: **5.28%**.

*Detailed record of all 48 suppressed predictions archived in [phase11_5_1_ignore_region_audit.json](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_5_1/phase11_5_1_ignore_region_audit.json).*

---

## 6. Standard Project AP vs. Custom Ignore-Aware AP Metrics

Below are the official threshold-independent AP metrics computed across the native candidate pool ($T \ge 0.05$):

| Evaluation Cohort | Metric Classification | Standard $\text{mAP}_{50}$ | Standard $\text{mAP}_{50:95}$ | Bacterial Blight $\text{AP}_{50}$ | Curl Virus $\text{AP}_{50}$ | Herbicide $\text{AP}_{50}$ | Jassids $\text{AP}_{50}$ | Redding $\text{AP}_{50}$ | Variegation $\text{AP}_{50}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cohort A (Original 15)** | **Standard Project AP** | **13.94%** | **8.53%** | 46.77% | 0.00% | 0.00% | 34.06% | 1.85% | 0.97% |
| **Cohort B (Total 30)** | **Standard Project AP (No Ignores)** | **5.16%** | **2.95%** | 18.05% | 0.00% | 0.00% | 11.58% | 0.21% | 1.10% |
| **Cohort B (Total 30)** | **Custom Ignore-Aware AP (Class-Agnostic)** | **5.39%** | **3.08%** | 18.87% | 0.00% | 0.00% | 12.04% | 0.23% | 1.18% |
| **Cohort B (Total 30)** | **Custom Class-Specific Ignore-Aware AP** | **5.28%** | **3.02%** | 18.80% | 0.00% | 0.00% | 11.58% | 0.23% | 1.10% |

*Per-class details archived in [phase11_5_1_per_class_ap.csv](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_5_1/phase11_5_1_per_class_ap.csv).*

---

## 7. Threshold-Dependent Operating Metrics

Below are the discrete operational performance metrics evaluated at specific confidence thresholds:

| Cohort & Evaluation Mode | Threshold ($T$) | TP | FP | FN | Precision | Recall | F1-Score | Excluded Ignores | Operating Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cohort A (Original 15)** | **0.05** | 55 | 473 | 88 | 10.42% | 38.46% | 0.1639 | 0 | Raw native candidate baseline |
| **Cohort A (Original 15)** | **0.25** | 45 | 159 | 98 | 22.06% | 31.47% | 0.2594 | 0 | Operational screening threshold |
| **Cohort A (Original 15)** | **0.40** | 40 | 92 | 103 | 30.30% | 27.97% | 0.2909 | 0 | High-precision operating point |
| **Cohort B (Tier 1 Only)** | **0.05** | 65 | 1065 | 261 | 5.75% | 19.94% | 0.0893 | 0 | Standard raw baseline (all 30 images) |
| **Cohort B (Tier 1 Only)** | **0.25** | 51 | 427 | 275 | 10.67% | 15.64% | 0.1269 | 0 | Standard operational screening (no ignore handling) |
| **Cohort B (Tier 1 Only)** | **0.40** | 46 | 299 | 280 | 13.33% | 14.11% | 0.1371 | 0 | Standard high-precision operating point |
| **Cohort B (Class-Agnostic Ignore)** | **0.05** | 65 | 924 | 261 | 6.57% | 19.94% | 0.0989 | 141 | Custom ignore-aware raw baseline |
| **Cohort B (Class-Agnostic Ignore)** | **0.25** | 51 | 379 | 275 | 11.86% | 15.64% | 0.1349 | 48 | Custom ignore-aware screening (48 FP excused) |
| **Cohort B (Class-Agnostic Ignore)** | **0.40** | 46 | 267 | 280 | 14.70% | 14.11% | 0.1440 | 32 | Custom ignore-aware high-precision |
| **Cohort B (Class-Specific Ignore)** | **0.05** | 65 | 1020 | 261 | 5.99% | 19.94% | 0.0921 | 45 | Custom class-specific raw baseline |
| **Cohort B (Class-Specific Ignore)** | **0.25** | 51 | 400 | 275 | 11.31% | 15.64% | 0.1313 | 27 | Custom class-specific screening (27 FP excused) |
| **Cohort B (Class-Specific Ignore)** | **0.40** | 46 | 277 | 280 | 14.24% | 14.11% | 0.1418 | 22 | Custom class-specific high-precision |

*Operating metrics archived in [phase11_5_1_threshold_operating_metrics.csv](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_5_1/phase11_5_1_threshold_operating_metrics.csv).*

---

## 8. Historical Baseline vs. Phase 11.5 vs. Phase 11.5.1 Reconciliation

| Dimension / Metric | Historical Phase 9.2C | Phase 11.5 Reported | Phase 11.5.1 Reconciled | Methodological Reconciliation Status |
| :--- | :---: | :---: | :---: | :--- |
| **Cohort A Standard $\text{mAP}_{50}$** | **13.94%** | 13.94% | **13.94%** | **HISTORICAL REPRODUCED (Standard Project AP)** |
| **Cohort A Standard $\text{mAP}_{50:95}$** | **8.53%** | 8.53% | **8.53%** | **HISTORICAL REPRODUCED (Standard Project AP)** |
| **Cohort A Reported mAP @ $T=0.25$** | *Not Evaluated* | 11.81% | **11.81% ($\text{AP}@0.25$)** | **RECLASSIFIED AS THRESHOLD-TRUNCATED AP** |
| **Cohort A Reported mAP @ $T=0.40$** | *Not Evaluated* | 10.39% | **10.39% ($\text{AP}@0.40$)** | **RECLASSIFIED AS THRESHOLD-TRUNCATED AP** |
| **Cohort A Precision @ $T=0.25$** | **22.06%** | 22.06% | **22.06%** | **HISTORICAL REPRODUCED (Operating Metric)** |
| **Cohort A Recall @ $T=0.25$** | **31.47%** | 31.47% | **31.47%** | **HISTORICAL REPRODUCED (Operating Metric)** |
| **Cohort B Tier 1 Standard $\text{mAP}_{50}$** | *N/A* | 5.16% | **5.16%** | **RECALCULATED (Standard Project AP)** |
| **Cohort B Class-Agnostic Ignore $\text{mAP}_{50}$** | *N/A* | 5.39% | **5.39%** | **RECALCULATED (Custom Ignore-Aware AP)** |
| **Cohort B Class-Specific Ignore $\text{mAP}_{50}$** | *N/A* | *Not Measured* | **5.28%** | **NEW BENCHMARK (Custom Class-Specific AP)** |
| **Cohort B FP @ $T=0.25$ (Ignore)** | *N/A* | 379 | **379 (Agnostic) / 400 (Specific)** | **RECONCILED (21 Cross-Class FP Excused in Agnostic)** |

*Direct comparison matrix saved in [phase11_5_1_metric_comparison.csv](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_5_1/phase11_5_1_metric_comparison.csv).*

---

## 9. Inference Performance & Hardware Interpretation

Hardware telemetry from Phase 11.5 recorded:
- Mean inference latency: **172.15 ms per image** (~5.8 FPS)
- Median latency: **157.12 ms per image** (~6.4 FPS)
- Dedicated GPU allocated memory: **507.0 MB** (out of 6.00 GB available)

### Scientific Qualifications
1. **Scope of Measurement:** This latency measures strictly the **isolated PyTorch forward pass and RoI head post-processing** (`model(image_tensor)`) on the local NVIDIA GeForce RTX 3050 Laptop GPU.
2. **End-to-End Application Latency:** Full web platform latency in production will include HTTP multipart ingestion, image validation (magic bytes, EXIF sanitization), PIL decode, resolution re-scaling, concurrent U-Net segmentation, SQLite prediction logging, and JSON payload serialization. Therefore, isolated detector inference throughput must not be characterized as "end-to-end real-time application performance".
3. **Concurrency Limitations:** The $86.55\%$ VRAM headroom reflects single-process evaluation on a developer workstation. It does not provide empirical evidence of concurrent multi-tenant capacity or latency stability under high-concurrency production load.

---

## 10. Dataset Provenance, Cohort Nesting & Invariants

### 10.1 Cohort Nesting Clarification
- **Nesting Structure:** Cohort A (15 images, 143 boxes) is a **proper subset** of Cohort B (30 images, 326 Tier 1 boxes, 68 Tier 2 ignore regions):
  $$\text{Cohort A} \subset \text{Cohort B}$$
  Cohort B consists of the 15 images from Cohort A plus 15 newly annotated validation images from Batch 03.
  They are **nested cohorts**, not independent or parallel validation datasets.
- **Relationship to Training Data:**
  Cross-referencing `data/detection/coco_train.json` (35 images) and `artifacts/manifests/duplicate_groups.csv` confirms that both Cohort A and Cohort B have **0 sample-level and 0 group-level overlap** with detector training data.
  Cohort B is an **extended disjoint validation cohort relative to detector training data**.
- **Generalization Limitation:**
  Because both cohorts are drawn from the same experimental imaging campaigns and environmental conditions, Cohort B provides an expanded validation sample within the same visual domain; it does **not** evaluate out-of-distribution or external multi-site clinical generalization.

### 10.2 System Invariants Verified

1. **Checkpoints:**
   - Faster R-CNN (`best_faster_rcnn.pt`): `01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2` (**INVARIANT**)
   - EfficientNet-B0 (`best_model.pt`): `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e` (**INVARIANT**)
   - U-Net (`best_unet.pt`): `612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43` (**INVARIANT**)
2. **Sealed Test Set:**
   - `artifacts/manifests/splits/test.csv` (321 images): **0 images accessed or evaluated** (**100% SEALED**).
3. **Database State:**
   - `cotton_leaf_ai.db`: Exactly 48 predictions rows, 5 users rows (**0 writes**).
4. **Approval Gate 4:**
   - Preserved as **`PENDING_APPROVAL`**.

---

## 11. Final Scientific Verdict & Recommendations

### Final Audit Verdict
> **PHASE 11.5.1 VERDICT: AUDITED, RECONCILED & SCIENTIFICALLY VALIDATED**  
> 1. **AP Convention:** The detector evaluation uses **VOC-style continuous interpolated AP evaluated across the COCO IoU threshold range $[0.50:0.05:0.95]$**. It is not official COCO 101-point interpolated AP.  
> 2. **Native Cutoff:** Standard project AP is evaluated over candidates emitted above the native $0.05$ score cutoff. The historical baseline of **13.94% $\text{mAP}_{50}$** on Cohort A is genuine. Lower mAP values at $T=0.25$ (11.81%) and $T=0.40$ (10.39%) are **Threshold-Truncated AP ($\text{AP}@T$)**.  
> 3. **Ignore Terminology & Provenance:** Cohort B without ignore regions achieves **5.16% Standard Project AP**. Class-agnostic ignore suppression yields **5.39% Custom Ignore-Aware AP** (excusing 48 FP at $T=0.25$, of which 21 are cross-class false alarms). Refined class-specific ignore suppression yields **5.28% Custom Class-Specific Ignore-Aware AP** (excusing 27 disease-matched FP at $T=0.25$).  
> 4. **Cohort Relationship:** Cohort A is nested inside Cohort B. Cohort B is an extended disjoint validation cohort relative to training data, but does not establish external generalization.  
> 5. **Readiness:** Production readiness remains strictly **`EXPERIMENTAL / NOT READY FOR AUTONOMOUS FIELD DEPLOYMENT`**.

---

**Execution Status:** Halted. Ready for subsequent user directives.
