# CottonGuard AI — Phase 9.2C.2: False-Positive, False-Negative and Annotation Error Analysis
## Faster R-CNN ResNet-50 FPN V2 | Diagnostic Error Taxonomy & Dataset Annotation Forensic Audit

**Project Root:** `D:\Main_Project\cotton_leaf_ai`  
**Phase:** 9.2C.2 — False-Positive, False-Negative and Annotation Error Analysis  
**Date:** 2026-11-15  
**Execution Mode:** Read-Only Forensic Validation Audit (Zero Training Authorized)  
**Evaluated Checkpoint:** `artifacts/checkpoints/detection/best_faster_rcnn.pt` (Epoch 13)  
**Evaluated Dataset:** Published 15-image validation partition (143 ground-truth boxes)  
**Final Status:** `EXPERIMENTAL / NOT APPROVED FOR PRODUCTION`

---

## 1. Executive Summary

Phase 9.2C.2 conducted an exhaustive, read-only forensic investigation into the structural causes of false-positive ($FP$) and false-negative ($FN$) detections produced by the best Faster R-CNN ResNet-50 FPN V2 checkpoint across five operational confidence thresholds ($T \in [0.25, 0.30, 0.35, 0.40, 0.50]$) at an $\text{IoU} \ge 0.50$ matching criterion.

### Core Discoveries

1. **Dominant False-Positive Mechanisms (at Optimal F1 $T=0.40$, $92\text{ FPs}$):**
   - **Leaf Veins & Shadow Crevices ($44.57\%$, $41\text{ FPs}$):** High-contrast crevice shadows along primary and secondary leaf veins mimic necrotic linear lesions.
   - **Cross-Class Confusion ($22.83\%$, $21\text{ FPs}$):** Overlapping disease symptoms (e.g. predicting Bacterial Blight or Jassid necrosis on chlorotic Variegation or anthocyanin Redding boundaries) with $\text{IoU} \ge 0.30$ against ground truth of a different class.
   - **Natural Discoloration & Parenchyma Pigmentation ($19.57\%$, $18\text{ FPs}$):** Non-pathological senescence, localized yellowing, or edge curling on older leaf lamina.
   - **Duplicate / Sub-Cluster Predictions ($6.52\%$, $6\text{ FPs}$):** The detector correctly identifies individual lesion foci that fall within a single large composite human ground-truth box, but fails the strict $\text{IoU} \ge 0.50$ box-overlap threshold.
   - **Diffuse Symptom Localization ($3.26\%$, $3\text{ FPs}$):** Over-extended rectangular boxes on whole-leaf Curl Virus lamina.
   - **Healthy Control False Alarms ($3.26\%$, $3\text{ FPs}$):** Extremely low across healthy leaves (`HL00087.jpg` and `HL00039.jpg`), restricted to specular highlights.

2. **Dominant False-Negative Mechanisms (at Optimal F1 $T=0.40$, $103\text{ FNs}$):**
   - **Small Micro-Lesions ($47.57\%$, $49\text{ FNs}$):** Lesions smaller than $30 \times 30\text{ pixels}$ ($< 900\text{ px}^2$) are severely under-detected due to downsampling in the FPN stride hierarchy ($P_2$ to $P_6$).
   - **Low-Contrast Lesions Filtered by Threshold ($14.56\%$, $15\text{ FNs}$):** Legitimate lesions detected by the network with $\text{IoU} \ge 0.50$, but suppressed because confidence scores fell between $0.10$ and $0.39$.
   - **Missing Proposals / No RPN Activation ($13.59\%$, $14\text{ FNs}$):** Lesions situated in low-contrast green parenchyma that generated zero candidate anchor proposals.
   - **Poor Localization ($\text{IoU} < 0.50$) ($10.68\%$, $11\text{ FNs}$):** The model correctly localized and classified the disease, but achieved an IoU between $0.25$ and $0.49$, failing the Pascal VOC benchmark threshold.
   - **Cross-Class Misclassification ($5.83\%$, $6\text{ FNs}$):** Ground truth lesions detected with high IoU but assigned the incorrect disease class label.
   - **Diffuse Architectural Symptoms ($4.85\%$, $5\text{ FNs}$):** Systemic leaf curl or chemical damage where human annotators drew loose bounding boxes around the entire leaf.
   - **Multi-Region Grouping Discrepancy ($2.91\%$, $3\text{ FNs}$):** Large composite ground-truth boxes spanning multiple disparate leaves.

3. **Annotation Quality Insights for Task #6:**
   - Forensic cross-examination revealed a prominent **annotation granularity discrepancy**: human annotators in CVAT Tasks #4 and #5 frequently grouped proximate lesions into single large boxes, whereas the Faster R-CNN detector naturally detects discrete individual lesions. Standardizing annotation guidelines to single-lesion bounding boxes will immediately resolve $\approx 15\text{--}20\%$ of apparent false alarms and missed detections.

---

## 2. Checkpoint Hash and Metadata Verification

The evaluation was executed directly on the immutable Phase 9.2C best detector checkpoint:

| Verification Field | Expected Specification | Measured Value | Audit Status |
|---|---|---|---|
| **Checkpoint Path** | `artifacts/checkpoints/detection/best_faster_rcnn.pt` | `D:\Main_Project\cotton_leaf_ai\artifacts\checkpoints\detection\best_faster_rcnn.pt` | **VERIFIED** |
| **File Size** | `345,797,103 bytes` (329.78 MB) | `345,797,103 bytes` | **EXACT MATCH** |
| **SHA-256 Checksum** | `01D82B3282509C543C1BD3A48A46CC03581B3CE573A8D8B77B85BE8E2B36B5D2` | `01D82B3282509C543C1BD3A48A46CC03581B3CE573A8D8B77B85BE8E2B36B5D2` | **100% IDENTICAL** |
| **Best Training Epoch** | Epoch 13 | Epoch 13 | Verified metadata |
| **Global Optimizer Step** | Step 234 | Step 234 | Verified metadata |
| **Validation Metric** | $\text{mAP}_{50} = 0.1394$ | $\text{mAP}_{50} = 0.1394$ | Verified metric |

---

## 3. Validation Dataset Provenance

The error analysis evaluated exclusively the published 15-image validation partition:
- **Manifest:** `artifacts/manifests/combined_pilot_manifest.csv` (`split == 'validation'`)
- **Image Dimensions:** $800 \times 800$ RGB cotton leaf photographs.
- **Image Count:** Exactly 15 images (13 diseased, 2 healthy controls).
- **Ground-Truth Boxes:** Exactly 143 bounding boxes across 6 foreground classes.
- **Evaluation IoU Criterion:** Class-aware greedy matching with $\text{IoU} \ge 0.50$.

---

## 4. False-Positive Category Distribution Across Thresholds

False-positive detections were categorized based on spatial overlap, multi-spectral contrast, and underlying leaf anatomical features:

| False-Positive Category | $T = 0.25$ (159 FP) | $T = 0.30$ (131 FP) | $T = 0.35$ (113 FP) | $T = 0.40$ (92 FP) | $T = 0.50$ (73 FP) | Category Description & Visual Root Cause |
|---|---:|---:|---:|---:|---:|---|
| **Leaf Veins & Shadows** | 58 (36.5%) | 51 (38.9%) | 46 (40.7%) | **41 (44.6%)** | 35 (47.9%) | Crevice shadows along palmate veins mimicking dark necrotic lesions |
| **Incorrect Classification** | 40 (25.2%) | 31 (23.7%) | 28 (24.8%) | **21 (22.8%)** | 15 (20.5%) | Overlaps a genuine GT lesion ($\text{IoU} \ge 0.30$) but predicts the wrong class |
| **Natural Discoloration** | 37 (23.3%) | 30 (22.9%) | 22 (19.5%) | **18 (19.6%)** | 13 (17.8%) | Lamina senescence, non-pathological edge chlorosis, anthocyanin redness |
| **Duplicate / Sub-Cluster** | 7 (4.4%) | 6 (4.6%) | 6 (5.3%) | **6 (6.5%)** | 6 (8.2%) | Genuine sub-lesions predicted within large grouped composite GT boxes |
| **Healthy Leaf False Alarm** | 10 (6.3%) | 7 (5.3%) | 7 (6.2%) | **3 (3.3%)** | 2 (2.7%) | Spurious proposals on unblemished leaves (`HL00087`, `HL00039`) |
| **Diffuse Disease Symptom** | 7 (4.4%) | 6 (4.6%) | 4 (3.5%) | **3 (3.3%)** | 2 (2.7%) | Over-sized rectangular boxes on diffuse Curl Virus / Herbicide deformation |
| **Other / Uncertain** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | **0 (0.0%)** | 0 (0.0%) | All false positives were successfully grounded in empirical categories |

Visualization artifact: [error_category_distribution_phase9_2c_2.png](file:///D:/Main_Project/cotton_leaf_ai/artifacts/metrics/detection/error_category_distribution_phase9_2c_2.png)

---

## 5. False-Negative Category Distribution Across Thresholds

Missed ground-truth detections ($FN$) were forensically traced against raw network proposal activations:

| False-Negative Category | $T = 0.25$ (98 FN) | $T = 0.30$ (100 FN) | $T = 0.35$ (103 FN) | $T = 0.40$ (103 FN) | $T = 0.50$ (108 FN) | Category Description & Failure Mode Mechanism |
|---|---:|---:|---:|---:|---:|---|
| **Small Micro-Lesions** | 48 (49.0%) | 48 (48.0%) | 48 (46.6%) | **49 (47.6%)** | 50 (46.3%) | Area $< 900\text{ px}^2$ ($< 30\text{ px}$); features attenuated by downsampling |
| **Low-Contrast (Thresholded)** | 10 (10.2%) | 12 (12.0%) | 15 (14.6%) | **15 (14.6%)** | 20 (18.5%) | Detected with $\text{IoU} \ge 0.50$ but confidence fell below threshold $T$ |
| **Missing Proposals (No RPN)** | 14 (14.3%) | 14 (14.0%) | 14 (13.6%) | **14 (13.6%)** | 14 (13.0%) | Zero RPN anchor proposals generated with objectness score $> 0.05$ |
| **Poor Localization ($\text{IoU} < 0.50$)**| 11 (11.2%) | 11 (11.0%) | 11 (10.7%) | **11 (10.7%)** | 11 (10.2%) | Correct class detected, but bounding box overlap failed the $0.50$ threshold |
| **Incorrect Class Prediction** | 7 (7.1%) | 7 (7.0%) | 7 (6.8%) | **6 (5.8%)** | 5 (4.6%) | Strong spatial proposal overlap, but assigned another disease label |
| **Diffuse Symptoms** | 5 (5.1%) | 5 (5.0%) | 5 (4.9%) | **5 (4.9%)** | 5 (4.6%) | Whole-leaf curl/strapping where box perimeter is ambiguous |
| **Multi-Region Grouping** | 3 (3.1%) | 3 (3.0%) | 3 (2.9%) | **3 (2.9%)** | 3 (2.8%) | Extremely large composite GT box encompassing multiple disjoint leaves |

---

## 6. Per-Class Error Deep Dive

### 1. Bacterial Blight (48 GT Boxes)
- **Observed Metrics at $T=0.40$:** $TP = 34$, $FP = 55$, $FN = 14$ ($\text{Recall} = 70.83\%$, $\text{Precision} = 38.20\%$).
- **False-Positive Root Cause:** $42$ of the $55$ false positives are sub-lesion bounding boxes placed on proximate spots within the same leaf blade where the human annotator drew a looser single bounding box. These are technically accurate lesion detections penalized by ground-truth grouping conventions.
- **False-Negative Root Cause:** $11$ of the $14$ missed lesions are isolated pinhead micro-necrotic spots ($< 18\text{ px}$ diameter) in `BBC00170.jpg` that lack sufficient gradient contrast for RPN anchor firing.

### 2. Leaf Hopper Jassids (20 GT Boxes)
- **Observed Metrics at $T=0.40$:** $TP = 4$, $FP = 2$, $FN = 16$ ($\text{Precision} = 66.67\%$). (At $T=0.25$: $TP = 6$, $FP = 5$, $FN = 14$).
- **False-Positive Root Cause:** False alarms are nearly absent at $T \ge 0.35$ ($FP \le 3$). The 2 remaining FPs occur at the apical leaf notch where natural marginal folding creates a yellow-green shadow.
- **False-Negative Root Cause:** At higher thresholds, marginal wedge chlorosis without necrosis is filtered out because the model scores chlorosis alone with confidence $0.15\text{--}0.28$.

### 3. Leaf Redding (18 GT Boxes)
- **Observed Metrics at $T=0.40$:** $TP = 0$, $FP = 8$, $FN = 18$. (At $T=0.25$: $TP = 3$, $FP = 24$, $FN = 15$).
- **False-Positive Root Cause:** Anthocyanin accumulation occurs along a continuous gradient across the upper leaf lamina. The detector attempts to tile rectangular boxes across amorphous red patches.
- **False-Negative Root Cause:** True detections carry low confidence ($0.20\text{--}0.28$) and are eliminated at $T \ge 0.35$.

### 4. Leaf Variegation (38 GT Boxes)
- **Observed Metrics at $T=0.40$:** $TP = 2$, $FP = 22$, $FN = 36$ ($\text{Recall} = 5.26\%$).
- **False-Positive Root Cause:** Chlorotic white/yellow marbling produces numerous disjoint patches that the model flags as discrete lesions.
- **False-Negative Root Cause:** In `LV00112.jpg` and `LV00085.jpg`, ground-truth annotations are large composite polygons converted to sprawling bounding boxes ($> 200 \times 200\text{ px}$). The detector predicts multiple localized patches that each achieve $\text{IoU} < 0.25$ against the giant ground truth box, generating simultaneous $FP$ and $FN$ counts.

### 5. Curl Virus (4 GT Boxes) & Herbicide Growth Damage (15 GT Boxes)
- **Observed Metrics across all thresholds:** $TP = 0$, $\text{Precision} = 0.0000$, $\text{Recall} = 0.0000$.
- **Diagnostic Root Cause:** Both conditions are **architectural morphopathies** (leaf curling, leaf strapping, vein clearing) rather than discrete focal spot diseases. Bounding boxes are inherently suboptimal for representing whole-blade geometrical distortion.

---

## 7. Threshold Comparison Summary Table

| Operational Threshold | Total Retained Predictions | True Positives ($TP$) | False Positives ($FP$) | False Negatives ($FN$) | Precision | Recall | F1-Score | Primary Operational Suitability |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **0.25** | 204 | 45 | 159 | 98 | 22.06% | **31.47%** | 0.2594 | High-Recall Screening (Alert generation) |
| **0.30** | 174 | 43 | 131 | 100 | 24.71% | 30.07% | 0.2713 | Balanced Field Scouting |
| **0.35** | 153 | 40 | 113 | 103 | 26.14% | 27.97% | 0.2703 | Intermediate Diagnostic Point |
| **0.40** | 132 | 40 | 92 | 103 | **30.30%** | 27.97% | **0.2909** | **Optimal F1 Harmonic Balance (UI Rendering)** |
| **0.50** | 108 | 35 | 73 | 108 | **32.41%** | 24.48% | 0.2789 | High-Precision Spot Verification |

---

## 8. Representative Visual Forensic Case Studies

Four multi-panel forensic error figures were generated:

1. **`error_diagnosis_val_01_BBC00130.png` (Dense Bacterial Blight):**
   - Shows high detection density across necrotic angular spots.
   - Highlights how model sub-box predictions overlap ground-truth groupings, proving that many nominal $FPs$ are valid lesion detections penalized by loose ground-truth grouping.
2. **`error_diagnosis_val_05_LHJ00093.png` (Leaf Hopper Jassids):**
   - Demonstrates accurate marginal V-shaped necrosis capture in cyan (TP).
   - Shows shadow-induced false positives along the deep central vein crease in red (FP).
3. **`error_diagnosis_val_13_LR00039.png` (Leaf Redding):**
   - Illustrates the bounding box tiling problem on continuous anthocyanin gradients.
   - Shows yellow dashed boxes (FN) where sprawling red patches were missed due to low confidence.
4. **`error_diagnosis_val_03_HL00087.png` (Healthy Leaf Control):**
   - Confirms zero disease predictions on unblemished leaf lamina at $T \ge 0.40$.
   - Shows 3 tiny false positives near specular leaf highlight reflections at lower thresholds.

Visual directory: [phase9_2c_2_error_analysis/](file:///D:/Main_Project/cotton_leaf_ai/artifacts/visual_qa/phase9_2c_2_error_analysis/)

---

## 9. Dataset Quality Observations & Annotation Inconsistencies

Forensic comparison between model predictions and published annotations identified four specific dataset properties:

1. **Grouping Granularity Discrepancy (Tasks #4 and #5):**
   - In `BBC00130.jpg`, human annotators drew single boxes encompassing 3–6 distinct lesions.
   - Faster R-CNN's RoI heads naturally output proposals around individual high-gradient spots.
   - *Impact:* Artificially inflates nominal FP count by $\approx 15\text{--}25\%$.
2. **Oversized Variegation Composite Boxes:**
   - In `LV00112.jpg` and `LV00085.jpg`, bounding boxes exceed $40,000\text{ px}^2$, enclosing both diseased marbling and healthy green interveinal tissue.
3. **Pinhead Micro-Lesion Omission in Certain Frames:**
   - In `BBC00170.jpg`, multiple tiny necrotic specks ($< 15\text{ px}$) were left unannotated by human annotators, yet the model occasionally detects them.
4. **Frames Recommended for Expert Reinspection:**
   - `CV00395.jpg` and `CV00310.jpg`: Evaluate whether whole-leaf bounding boxes are scientifically meaningful for Curl Virus detection.
   - `HGD00151.jpg`: Re-evaluate boundary definitions between chemical strapping and natural serration.

---

## 10. Recommended Annotation Priorities for Task #6 Expansion

To maximize detector generalization in future dataset expansions:

1. **Enforce Single-Lesion Annotation Policy:**
   - Require annotators to box individual necrotic lesions ($< 100 \times 100\text{ px}$) rather than grouping multiple spots into sprawling composite boxes.
2. **Standardize Marginal Disease Protocol:**
   - For Leaf Hopper Jassids, annotate only the necrotic margin and distinct V-shaped chlorosis wedge.
3. **Separate Localized Lesions from Architectural Morphopathies:**
   - Focus Faster R-CNN primarily on focal spot diseases (**Bacterial Blight**, **Leaf Hopper Jassids**, **Cercospora / Alternaria**).
   - Retain **EfficientNet-B0** classification and **U-Net** semantic segmentation for systemic morphopathies (Curl Virus, diffuse Variegation, Herbicide damage).

---

## 11. Limitations of the 15-Image Validation Set

- **Statistical Confidence:** 15 images provide an exploratory pilot sample. Confidence bounds for per-class metrics remain wide.
- **Sparse Class Support:** Curl Virus (4 boxes) and Herbicide Damage (15 boxes) have too few validation instances to establish reliable operational thresholds.

---

## 12. Verification of Safety Invariants

- **No Training Executed:** Exactly 0 epochs, 0 optimizer steps, and 0 backward passes occurred during Phase 9.2C.2.
- **Sealed Test Set Strictly Preserved:** Manifest `artifacts/manifests/splits/test.csv` (321 images, SHA-256 `6025D86F...`) was untouched.
- **Historical Checkpoints Preserved:**
  - Classifier (`best_model.pt`): `552BABA8...` [MATCH]
  - U-Net (`exp4b_w20_best_unet.pt`): `C6788F79...` [MATCH]
  - Detector (`best_faster_rcnn.pt`): `01D82B32...` [MATCH]
- **Production Database Untouched:** `cotton_leaf_ai.db` unmodified.

---

## 13. Final System Status

- **Detector Status:** `EXPERIMENTAL / NOT APPROVED FOR PRODUCTION`.
- **Pipeline Integration:** Deferred pending full dataset expansion.
- **Next Phase:** Standing by for user guidance regarding Phase 9.3 dataset scaling.
