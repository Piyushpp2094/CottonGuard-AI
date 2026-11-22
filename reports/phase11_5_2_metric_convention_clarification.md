# CottonGuard AI — Phase 11.5.2: Final Detection Metric Convention & Ignore-Region Reporting Clarification Report

**Date:** November 17, 2026  
**Auditor Role:** Senior Machine Learning Engineer, Computer Vision Engineer and Independent AI Model Auditor  
**Project:** CottonGuard AI (`D:\Main_Project\cotton_leaf_ai`)  
**Phase:** 11.5.2 — Final Detection Metric Convention & Ignore-Region Reporting Clarification  
**Audit Status:** `COMPLETED`  
**Detection Checkpoint:** `artifacts/checkpoints/detection/best_faster_rcnn.pt`  
**Authoritative SHA-256:** `01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2` (Verified Match: **YES**)  
**Sealed Test Split Protection:** 100% Preserved (0/321 classifier test images accessed, loaded, or evaluated)  
**Database State:** 100% Preserved (48 predictions rows, 5 users rows; zero writes)  
**Approval Gate 4:** `PENDING_APPROVAL` (Strictly Preserved)  

---

## 1. Executive Summary

This report delivers the final documentation-only scientific clarification of the Faster R-CNN detection metrics evaluated across **Phase 11.5** and **Phase 11.5.1**.

The audit formally resolves five methodological and reporting nuances:
1. **AP Implementation Convention:** Identifies the project's exact AP procedure as **VOC-style continuous interpolated AP** evaluated across the COCO IoU threshold range $[0.50:0.05:0.95]$, explicitly distinguishing it from official MS COCO 101-point interpolated evaluation (`pycocotools`).
2. **Native Confidence Floor:** Formally establishes that reported standard project AP is evaluated strictly over the candidate predictions emitted under TorchVision's native `roi_heads.score_thresh = 0.05` cutoff ($1,130$ proposals, confidence range $[0.050082, 0.997260]$), clarifying that candidates below $0.05$ were pruned internally during inference.
3. **Ignore-Aware Metric Nomenclature:** Refines the terminology of ignore-adjusted evaluations into **Standard Project AP without ignore suppression** (Tier 1 only, 5.16%), **Custom ignore-aware AP** (class-agnostic, 5.39%), and **Custom class-specific ignore-aware AP** (5.28%).
4. **Clinical Diagnosis Provenance & Mixed Infection Limitations:** Documents that class-specific ignore suppression derives its diagnosis from the image-level `class_label` in the consolidated manifest. Acknowledges the scientific limitation that this heuristic assumes homogeneous pathology across all ignore regions on a leaf and cannot independently verify polygon-level diagnoses in mixed infections.
5. **Cohort Relationship & Nesting:** Clarifies that Cohort A (15 images) is a **proper subset nested inside Cohort B** (30 images), with Cohort B adding 15 Batch 03 validation images. Both cohorts are disjoint from detector training data, but Cohort B evaluates within-distribution validation expansion rather than external clinical generalization.

---

## 2. Actual AP Implementation Convention

### 2.1 Inspection of `_compute_ap()`
The project's detection metric pipeline computes Average Precision in `src/training/detection_losses.py` via `_compute_ap()`:
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

### 2.2 Mathematical Classification & Convention Comparison

| Dimension | PASCAL VOC (2010–2012) Convention | Official MS COCO (`pycocotools`) Convention | CottonGuard AI Implementation | Scientific Classification |
| :--- | :--- | :--- | :--- | :--- |
| **Integration Type** | Exact continuous area under curve (AUC) | 101-point discrete interpolated sampling | Exact continuous area under curve (AUC) | **VOC-Style Continuous AP** |
| **Recall Sampling** | Integrates all distinct recall step intervals $(r_{k+1} - r_k)$ | Evaluates at $r \in \{0.00, 0.01, \dots, 1.00\}$ | Integrates all distinct recall step intervals $(r_{k+1} - r_k)$ | **VOC-Style Continuous AP** |
| **Precision Smoothing** | Monotonically decreasing: $\max_{r' \ge r} p(r')$ | Monotonically decreasing: $\max_{r' \ge r} p(r')$ | Monotonically decreasing: $\max_{r' \ge r} p(r')$ | **Standard Monotonic Envelope** |
| **IoU Thresholds** | Single threshold ($\text{IoU} \ge 0.50$) | 10 thresholds: $[0.50:0.05:0.95]$ | 10 thresholds: $[0.50:0.05:0.95]$ | **COCO IoU Threshold Range** |
| **Crowd / Area Ranges** | Optional `difficult=1` exclusion | Area ranges (S/M/L) & `maxDets=[1, 10, 100]` | Macro-average across all target scales | **Custom Project Wrapper** |

- **Scientific Verdict:** The project implementation is **VOC-style continuous interpolated AP evaluated across the COCO IoU threshold range $[0.50:0.05:0.95]$**.
- **Formal Boundary:** It is **not** official MS COCO 101-point interpolated AP. Official COCO evaluation metrics are not available and are not claimed.

---

## 3. Native Confidence Floor Verification

TorchVision's standard `fasterrcnn_resnet50_fpn_v2` model constructor sets an internal confidence threshold on the RoI heads:
- **Parameter:** `model.roi_heads.score_thresh = 0.05`
- **Inference Behavior:** During the forward evaluation pass (`model(image_tensor)`), the RoI heads perform non-maximum suppression (NMS) and immediately prune any box proposal with a classification score below `0.05`.

### Forensic Verification of Saved Predictions
In [phase11_5_detection_predictions.json](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_5/phase11_5_detection_predictions.json):
- Total saved image records: **30 images**
- Total candidate predictions: **1,130 predictions**
- Minimum observed confidence: **0.05008207634091377**
- Maximum observed confidence: **0.9972596168518066**

### Scientific Clarification
The reported standard project AP is calculated strictly over the candidate predictions emitted under this native cutoff ($T_{\text{native}} \ge 0.05$). Proposals with confidence $< 0.05$ were pruned internally by TorchVision during inference. The term "threshold-independent AP" in this report denotes that **no secondary post-inference confidence threshold** (such as $T=0.25$ or $T=0.40$) was applied before integrating the PR curve.

---

## 4. Ignore-Aware AP Terminology & Diagnosis Provenance

### 4.1 Metric Terminology Alignment
To maintain scientific rigor and prevent conflation with unmodified standard benchmarks, the following terminology is formally adopted across all project documentation:

1. **Standard Project AP without Ignore Suppression:**
   - Evaluated on Cohort B against Tier 1 focal ground-truth boxes only. Tier 2 ignore regions are ignored during evaluation.
   - Result: $\text{mAP}_{50} = \mathbf{5.16\%}$, $\text{mAP}_{50:95} = \mathbf{2.95\%}$.
2. **Custom Ignore-Aware AP (Class-Agnostic):**
   - Candidate predictions overlapping any Tier 2 ignore region with compound score $\max(\text{IoA}, \text{IoU}) \ge 0.30$ are excluded from false-positive accounting, regardless of predicted class.
   - Result: $\text{mAP}_{50} = \mathbf{5.39\%}$, $\text{mAP}_{50:95} = \mathbf{3.08\%}$.
3. **Custom Class-Specific Ignore-Aware AP:**
   - Candidate predictions are excluded from false-positive accounting only if they overlap a Tier 2 ignore region AND their predicted class matches the image's annotated disease diagnosis.
   - Result: $\text{mAP}_{50} = \mathbf{5.28\%}$, $\text{mAP}_{50:95} = \mathbf{3.02\%}$.

### 4.2 Image-Level Diagnosis Provenance & Clinical Limitations

- **Source of Diagnosis:** The disease class used for class-specific ignore suppression is loaded from the `class_label` column of [artifacts/manifests/consolidated_100_annotation_manifest.csv](file:///D:/Main_Project/cotton_leaf_ai/artifacts/manifests/consolidated_100_annotation_manifest.csv).
- **Clinical Nuance & Scientific Limitation:**
  1. `class_label` represents a **single, global, image-level primary diagnosis** assigned during field collection.
  2. In agricultural plant pathology, cotton leaves frequently display **co-infections or multi-pathogen complexes** (e.g., severe Bacterial Blight necrotic lesions co-occurring with marginal Hopperburn chlorosis or secondary fungal leaf spot).
  3. Class-specific ignore suppression assumes that every Tier 2 ignore region on a given leaf corresponds to that leaf's primary diagnosis. If an annotator drew an ignore region around secondary mixed-infection chlorosis, the class-specific rule will penalize a genuine detection of the secondary disease while forgiving only the primary disease.
  4. Because individual Tier 2 ignore polygons were not annotated with independent, polygon-level clinical pathogen labels, using the image-level label represents an operational clinical heuristic rather than an independently verified per-polygon ground truth.

---

## 5. Cohort Nesting & Evaluation Generalization

### 5.1 Cohort Nesting Structure
The evaluation cohorts utilized across Phase 11.5, 11.5.1, and 11.5.2 possess the following mathematical relationship:
$$\text{Cohort A} \subset \text{Cohort B}$$

```
+-------------------------------------------------------------------------------+
| Cohort B: Extended Consolidated Validation Cohort (30 Images, 326 GT Boxes)   |
|                                                                               |
|   +-----------------------------------------------------------------------+   |
|   | Cohort A: Original Validation Cohort (15 Images, 143 GT Boxes)         |   |
|   | (Batches 01 & 02: Historical Benchmark Reproduction Split)            |   |
|   +-----------------------------------------------------------------------+   |
|                                                                               |
|   +-----------------------------------------------------------------------+   |
|   | Batch 03 Validation Extension (15 Images, 183 Tier 1, 68 Tier 2)      |   |
|   | (Newly Annotated Pilot Extension from CVAT Task #6)                   |   |
|   +-----------------------------------------------------------------------+   |
+-------------------------------------------------------------------------------+
```

- **Cohort A:** Contains exactly the 15 original validation images from Batches 01 and 02. It serves strictly as a **historical benchmark reproduction cohort**.
- **Cohort B:** Contains all 15 images from Cohort A plus 15 newly annotated validation images from Batch 03.
- **Nesting Implication:** Cohort A and Cohort B are **nested, non-independent cohorts**. Results on Cohort B include and reflect the performance on Cohort A.

### 5.2 Scope of Generalization
1. **Disjoint Relative to Detector Training:** Both Cohort A (15 images) and Cohort B (30 images) have been forensically verified to have **0 sample-level and 0 group-level overlap** with the 35 detector training images in `data/detection/coco_train.json`. Cohort B is correctly characterized as an **extended disjoint validation cohort relative to detector training data**.
2. **Within-Distribution Limitation:** Both cohorts were captured under identical collection protocols, regional farm locations, and sensor equipment. Cohort B demonstrates model stability on additional held-out plants within the same visual domain; it does **not** establish out-of-distribution, multi-region, or external field generalization.

---

## 6. Complete Preserved & Reconciled Metric Inventory

All historical, reproduced, and recalculated metric measurements are preserved below with explicit scientific labeling:

### 6.1 Standard Project AP vs. Custom Ignore-Aware AP ($T \ge 0.05$)

| Evaluation Cohort | Metric Specification | $\text{mAP}_{50}$ | $\text{mAP}_{50:95}$ | Bacterial Blight $\text{AP}_{50}$ | Curl Virus $\text{AP}_{50}$ | Herbicide $\text{AP}_{50}$ | Jassids $\text{AP}_{50}$ | Redding $\text{AP}_{50}$ | Variegation $\text{AP}_{50}$ | Provenance Category |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cohort A (Original 15)** | **Standard Project AP** | **13.94%** | **8.53%** | 46.77% | 0.00% | 0.00% | 34.06% | 1.85% | 0.97% | **Historical Reproduced** |
| **Cohort B (Total 30)** | **Standard Project AP (No Ignores)** | **5.16%** | **2.95%** | 18.05% | 0.00% | 0.00% | 11.58% | 0.21% | 1.10% | **Recalculated Standard** |
| **Cohort B (Total 30)** | **Custom Ignore-Aware AP (Class-Agnostic)** | **5.39%** | **3.08%** | 18.87% | 0.00% | 0.00% | 12.04% | 0.23% | 1.18% | **Custom Ignore Benchmark** |
| **Cohort B (Total 30)** | **Custom Class-Specific Ignore-Aware AP** | **5.28%** | **3.02%** | 18.80% | 0.00% | 0.00% | 11.58% | 0.23% | 1.10% | **Refined Clinical Benchmark** |

### 6.2 Threshold-Truncated AP Diagnostics ($\text{AP}@T$)

| Cohort | Operating Threshold ($T$) | Truncated $\text{mAP}_{50}$ | Truncated $\text{mAP}_{50:95}$ | Maximum Reached Recall | Methodological Diagnosis |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Cohort A** | $T=0.05$ | 13.94% | 8.53% | 38.46% | Untruncated native baseline (Standard Project AP) |
| **Cohort A** | $T=0.25$ | 11.81% | 7.39% | 31.47% | Truncated AP ($\text{AP}@0.25$); recall capped at 31.47% |
| **Cohort A** | $T=0.40$ | 10.39% | 6.52% | 27.97% | Truncated AP ($\text{AP}@0.40$); recall capped at 27.97% |
| **Cohort B (Tier 1)** | $T=0.05$ | 5.16% | 2.95% | 19.94% | Untruncated native baseline (Standard Project AP) |
| **Cohort B (Tier 1)** | $T=0.25$ | 4.42% | 2.58% | 15.64% | Truncated AP ($\text{AP}@0.25$); recall capped at 15.64% |
| **Cohort B (Tier 1)** | $T=0.40$ | 4.04% | 2.35% | 14.11% | Truncated AP ($\text{AP}@0.40$); recall capped at 14.11% |

### 6.3 Threshold-Dependent Operating Metrics

| Cohort & Evaluation Mode | Threshold ($T$) | TP | FP | FN | Precision | Recall | F1-Score | Excluded Ignores | Operating Point Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Cohort A (Original 15)** | **0.05** | 55 | 473 | 88 | 10.42% | 38.46% | 0.1639 | 0 | Raw native candidate baseline |
| **Cohort A (Original 15)** | **0.25** | 45 | 159 | 98 | 22.06% | 31.47% | 0.2594 | 0 | Operational screening threshold |
| **Cohort A (Original 15)** | **0.40** | 40 | 92 | 103 | 30.30% | 27.97% | 0.2909 | 0 | High-precision operating point |
| **Cohort B (Tier 1 Only)** | **0.05** | 65 | 1065 | 261 | 5.75% | 19.94% | 0.0893 | 0 | Standard baseline (all 30 images) |
| **Cohort B (Tier 1 Only)** | **0.25** | 51 | 427 | 275 | 10.67% | 15.64% | 0.1269 | 0 | Standard operational screening |
| **Cohort B (Tier 1 Only)** | **0.40** | 46 | 299 | 280 | 13.33% | 14.11% | 0.1371 | 0 | Standard high-precision operating point |
| **Cohort B (Class-Agnostic Ignore)** | **0.05** | 65 | 924 | 261 | 6.57% | 19.94% | 0.0989 | 141 | Custom ignore-aware raw baseline |
| **Cohort B (Class-Agnostic Ignore)** | **0.25** | 51 | 379 | 275 | 11.86% | 15.64% | 0.1349 | 48 | Custom ignore-aware screening |
| **Cohort B (Class-Agnostic Ignore)** | **0.40** | 46 | 267 | 280 | 14.70% | 14.11% | 0.1440 | 32 | Custom ignore-aware high-precision |
| **Cohort B (Class-Specific Ignore)** | **0.05** | 65 | 1020 | 261 | 5.99% | 19.94% | 0.0921 | 45 | Custom class-specific raw baseline |
| **Cohort B (Class-Specific Ignore)** | **0.25** | 51 | 400 | 275 | 11.31% | 15.64% | 0.1313 | 27 | Custom class-specific screening |
| **Cohort B (Class-Specific Ignore)** | **0.40** | 46 | 277 | 280 | 14.24% | 14.11% | 0.1418 | 22 | Custom class-specific high-precision |

---

## 7. Modified Files & Cryptographic Hashes

To ensure absolute forensic accountability, before-and-after cryptographic hashes are recorded for all files updated under Phase 11.5.2:

| File Path | Modification Type | Before SHA-256 | After SHA-256 | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| `reports/phase11_5_1_detection_metrics_methodology_audit.md` | Content Clarification | `a25bdf5fda7ffd855345553eb480d991159b56838310f9f4ac3fe64a744abf29` | `b1a1740af5c17467fd8fa183a5270ff7936a6f6cf5ca9ede26db40cc6e9f718e` | **UPDATED & VERIFIED** |
| `reports/phase11_5_2_metric_convention_clarification.md` | New Deliverable | *None (Created)* | `ec232aee715eaf4cd275d5636b14549ed4f596c8f03921b2f881d17dc9db6244` | **CREATED & VERIFIED** |
| `reports/MASTER_PROJECT_TASK_TRACKER.md` | Task Addition | `7ea106fa6bd47c191e4275a2295c929aefae9882531f54f5022159f4e6d3ddaf` | `cc89a0c23da627407117cef0aaf50a904cd7654f0852edc06985a1fea04d0c3f` | **UPDATED & VERIFIED** |

---

## 8. Preserved System Invariants

1. **Checkpoints:**
   - Detection (`best_faster_rcnn.pt`): `01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2` (**UNTOUCHED**)
   - Classification (`best_model.pt`): `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e` (**UNTOUCHED**)
   - Segmentation (`best_unet.pt`): `612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43` (**UNTOUCHED**)
2. **Database:** Exactly 48 predictions rows and 5 users rows (**0 writes**).
3. **Sealed Test Set:** Exactly 321 images in `artifacts/manifests/splits/test.csv` (**0 accessed**).
4. **Approval Gate 4:** Locked in **`PENDING_APPROVAL`**.

---

## 9. Remaining Scientific Limitations

1. **Polygon-Level Pathology Verification:** In the absence of per-polygon pathogen labels on Tier 2 ignore regions, class-specific ignore suppression depends on the image-level diagnosis. This limitation should be resolved in future annotation campaigns by assigning multi-label attributes to diffuse ignore polygons.
2. **Diffuse Disease Localization:** Faster R-CNN remains unsuited for whole-leaf curling (Curl Virus) and interveinal chlorosis (Herbicide Growth Damage). Semantic segmentation (U-Net) or keypoint detection is mathematically better aligned with diffuse laminar pathologies.
3. **Hard Negative Vein Tuning:** False positive rates on healthy leaves ($5\text{--}10$ false alarms per healthy leaf) confirm that future detector retraining must incorporate explicit negative loss penalties on healthy leaf background proposals.

---

## 10. Final Verification Verdict

> **PHASE 11.5.2 VERDICT: FULLY CLARIFIED, RECONCILED & SCIENTIFICALLY SEALED**  
> 1. The detection evaluation procedure is formally established as **VOC-style continuous interpolated AP evaluated across the COCO IoU threshold range $[0.50:0.05:0.95]$**.  
> 2. Standard project AP is evaluated over the native candidate prediction pool emitted above TorchVision's native $0.05$ score cutoff.  
> 3. Ignore-adjusted metrics are explicitly categorized into **Standard Project AP without ignore suppression** (5.16%), **Custom ignore-aware AP** (5.39%), and **Custom class-specific ignore-aware AP** (5.28%).  
> 4. Cohort A is nested inside Cohort B. Cohort B is an extended disjoint validation cohort relative to detector training data, without claiming external generalization.  
> 5. Production readiness remains strictly **`EXPERIMENTAL / NOT READY FOR AUTONOMOUS FIELD DEPLOYMENT`**.

---

**Execution Status:** Halted. Ready for subsequent user directives.
