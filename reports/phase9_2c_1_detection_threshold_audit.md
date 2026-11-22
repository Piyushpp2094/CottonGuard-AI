# CottonGuard AI — Phase 9.2C.1: Post-Training Audit and Confidence Threshold Analysis
## Faster R-CNN ResNet-50 FPN V2 | Validation Operating-Point & Decision-Boundary Audit

**Project Root:** `D:\Main_Project\cotton_leaf_ai`  
**Phase:** 9.2C.1 — Post-Training Audit and Confidence Threshold Analysis  
**Date:** 2026-11-15  
**Execution Mode:** Evaluation-Only Audit (Zero Training Authorized)  
**Evaluated Checkpoint:** `artifacts/checkpoints/detection/best_faster_rcnn.pt` (Epoch 13)  
**Evaluated Dataset:** Published 15-image validation split (143 ground-truth boxes)  
**Final Status:** `EXPERIMENTAL / NOT APPROVED FOR PRODUCTION`

---

## 1. Checkpoint Hash and Metadata Verification

Before commencing the threshold sweep, the integrity of the target detector checkpoint was cryptographically verified on disk:

| Verification Field | Expected Specification | Measured Value | Status |
|---|---|---|---|
| **Checkpoint Path** | `artifacts/checkpoints/detection/best_faster_rcnn.pt` | `D:\Main_Project\cotton_leaf_ai\artifacts\checkpoints\detection\best_faster_rcnn.pt` | **VERIFIED** |
| **File Size** | `345,797,103 bytes` | `345,797,103 bytes` (329.78 MB) | **EXACT MATCH** |
| **SHA-256 Checksum** | `01D82B3282509C543C1BD3A48A46CC03581B3CE573A8D8B77B85BE8E2B36B5D2` | `01D82B3282509C543C1BD3A48A46CC03581B3CE573A8D8B77B85BE8E2B36B5D2` | **100% IDENTICAL** |
| **Best Training Epoch** | Epoch 13 | Epoch 13 | Stored metadata verified |
| **Global Optimizer Step** | Step 234 | Step 234 | Stored metadata verified |
| **Stored Validation $\text{mAP}_{50}$** | $0.1394$ | $0.1394$ | Recomputed: $0.1394$ |
| **Stored Validation $\text{mAP}_{50:95}$** | $0.0853$ | $0.0853$ | Recomputed: $0.0853$ |
| **Checkpoint Serialization Timestamp** | `2026-11-15T06:52:11.483061+00:00` | Verified UTC timestamp | Stored metadata verified |

---

## 2. Validation Dataset Provenance

The confidence threshold analysis was conducted strictly and exclusively on the published 15-image validation partition:
- **Manifest:** `artifacts/manifests/combined_pilot_manifest.csv` (`split == 'validation'`)
- **Published Sources:** CVAT Task #4 (Batch 01) and CVAT Task #5 (Batch 02)
- **Validation Images:** Exactly 15 images (13 diseased, 2 healthy controls)
- **Validation Ground-Truth Boxes:** Exactly 143 bounding boxes across 6 foreground disease classes
- **Healthy Controls:** `HL00087.jpg` and `HL00039.jpg` (both verified with exactly 0 ground-truth boxes)
- **Class Ground-Truth Breakdown:**
  - Bacterial Blight: 48 boxes
  - Curl Virus: 4 boxes
  - Herbicide Growth Damage: 15 boxes
  - Leaf Hopper Jassids: 20 boxes
  - Leaf Redding: 18 boxes
  - Leaf Variegation: 38 boxes
  - **Total:** 143 boxes

---

## 3. Distinction Between mAP Metric and Confidence Operating Points

### Academic Metric Integrity
It is vital to distinguish between Mean Average Precision ($\text{mAP}$) and discrete confidence thresholding:
- **$\text{mAP}_{50}$ ($0.1394$) & $\text{mAP}_{50:95}$ ($0.0853$):** Average Precision is an integral metric computed over the entire monotonic precision-recall curve across all potential confidence thresholds. It reflects the fundamental ranking capability and spatial feature representation of the trained neural network. Filtering by confidence does **not** change the model's underlying mAP.
- **Confidence Threshold Filtering:** Thresholding selects a single discrete operating point $(\text{Precision}(T), \text{Recall}(T))$ along that curve. It represents an operational decision boundary for deployment (e.g. trading off false alarms against missed detections).

### Prediction Matching Protocol
For every evaluation threshold $T \in [0.05, 0.50]$:
1. Predictions with confidence score $s < T$ are discarded.
2. Retained predictions are grouped per class and sorted in descending order of confidence.
3. Each predicted box is greedily matched against ground-truth boxes of the identical class within that image.
4. If the intersection-over-union $\text{IoU} \ge 0.50$ against an unmatched ground-truth box, it is designated a **True Positive ($TP$)** and that ground-truth box is marked as matched.
5. If $\text{IoU} < 0.50$ or the ground-truth box was already claimed, the prediction is designated a **False Positive ($FP$)**.
6. Any ground-truth box remaining unmatched at the conclusion of matching is designated a **False Negative ($FN$)**.

---

## 4. Confidence Threshold Sweep Comparison Table

Evaluated across the 9 authorized confidence thresholds on all 15 validation images (143 ground-truth boxes):

| Confidence Threshold ($T$) | Retained Predictions | True Positives ($TP$) | False Positives ($FP$) | False Negatives ($FN$) | Precision ($\text{Prec}_{50}$) | Recall ($\text{Rec}_{50}$) | F1-Score | FP Reduction vs Baseline |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0.05** (Raw Baseline) | 528 | 55 | 473 | 88 | 0.1042 (10.42%) | **0.3846 (38.46%)** | 0.1639 | Baseline (0%) |
| **0.10** | 368 | 50 | 318 | 93 | 0.1359 (13.59%) | 0.3497 (34.97%) | 0.1957 | -32.77% |
| **0.15** | 293 | 49 | 244 | 94 | 0.1672 (16.72%) | 0.3427 (34.27%) | 0.2248 | -48.41% |
| **0.20** | 241 | 46 | 195 | 97 | 0.1909 (19.09%) | 0.3217 (32.17%) | 0.2396 | -58.77% |
| **0.25** | 204 | 45 | 159 | 98 | 0.2206 (22.06%) | 0.3147 (31.47%) | 0.2594 | -66.38% |
| **0.30** | 174 | 43 | 131 | 100 | 0.2471 (24.71%) | 0.3007 (30.07%) | 0.2713 | -72.30% |
| **0.35** | 153 | 40 | 113 | 103 | 0.2614 (26.14%) | 0.2797 (27.97%) | 0.2703 | -76.11% |
| **0.40** (Optimal F1) | 132 | 40 | 92 | 103 | **0.3030 (30.30%)** | 0.2797 (27.97%) | **0.2909** | **-80.55%** |
| **0.50** (Strict) | 108 | 35 | 73 | 108 | **0.3241 (32.41%)** | 0.2448 (24.48%) | 0.2789 | **-84.57%** |

Visualization artifact: [confidence_threshold_sweep_phase9_2c_1.png](file:///D:/Main_Project/cotton_leaf_ai/artifacts/metrics/detection/confidence_threshold_sweep_phase9_2c_1.png)

---

## 5. Precision-Recall-F1 Trade-off Analysis

The quantitative progression across thresholds reveals crucial operating characteristics:

1. **Suppression of False Positives:**
   - At the raw detector threshold ($T=0.05$), the network emits 528 candidate boxes across 15 images ($35.2$ boxes/image), producing 473 false positives and a low precision of $10.42\%$.
   - Increasing the threshold to $T=0.25$ eliminates 314 false positives (a **$66.38\%$ reduction**), reducing predictions to $13.6$ boxes/image.
   - At $T=0.40$, false positives plummet to 92 (an **$80.55\%$ reduction**), and at $T=0.50$, false positives drop to 73 (an **$84.57\%$ reduction**).
2. **True Positive Retention and Recall Stability:**
   - While false positives fall by over $80\%$, true positive detections drop much more slowly: from 55 ($T=0.05$) down to 45 ($T=0.25$) and 40 ($T=0.40$).
   - Between $T=0.05$ and $T=0.40$, the model retains **$72.73\%$ of all its true positive detections** ($40/55$), while purging **$80.55\%$ of its noise**.
3. **F1-Score Maximization:**
   - F1-score rises monotonically from $0.1639$ ($T=0.05$) to $0.2594$ ($T=0.25$), peaking at **$F1 = 0.2909$ at $T=0.40$** (a **$+77.49\%$ improvement** in harmonic balance).
   - Above $T=0.40$, recall attenuation begins to outpace precision gains, causing F1 to decline slightly to $0.2789$ at $T=0.50$.

---

## 6. Per-Class Threshold Results

### Class 0: Bacterial Blight (48 Ground-Truth Boxes)
Bacterial Blight represents the strongest detection capability of the trained detector:

| Threshold | Retained Preds | TP | FP | FN | Precision | Recall | F1-Score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 191 | 36 | 155 | 12 | 0.1885 | **0.7500** | 0.3013 |
| 0.10 | 155 | 35 | 120 | 13 | 0.2258 | 0.7292 | 0.3448 |
| 0.20 | 125 | 34 | 91 | 14 | 0.2720 | 0.7083 | 0.3931 |
| 0.30 | 106 | 34 | 72 | 14 | 0.3208 | 0.7083 | 0.4416 |
| **0.40** | **89** | **34** | **55** | **14** | **0.3820** | **0.7083** | **0.4964** |
| **0.50** | **81** | **33** | **48** | **15** | **0.4074** | **0.6875** | **0.5116** |

- **Key Finding:** True positive Bacterial Blight detections have high confidence. Between $T=0.05$ and $T=0.40$, recall drops by only **$4.17\%$** ($36 \to 34$ lesions detected), while precision doubles ($18.85\% \to 38.20\%$). At $T=0.50$, the detector still detects **$68.75\%$ of all Bacterial Blight lesions** ($33/48$) with precision exceeding $40\%$.

### Class 3: Leaf Hopper Jassids (20 Ground-Truth Boxes)

| Threshold | Retained Preds | TP | FP | FN | Precision | Recall | F1-Score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 48 | 11 | 37 | 9 | 0.2292 | **0.5500** | 0.3235 |
| 0.15 | 17 | 8 | 9 | 12 | 0.4706 | 0.4000 | **0.4324** |
| 0.25 | 11 | 6 | 5 | 14 | 0.5455 | 0.3000 | 0.3871 |
| **0.30** | **9** | **6** | **3** | **14** | **0.6667** | **0.3000** | **0.4138** |
| 0.40 | 6 | 4 | 2 | 16 | 0.6667 | 0.2000 | 0.3077 |
| 0.50 | 2 | 1 | 1 | 19 | 0.5000 | 0.0500 | 0.0909 |

- **Key Finding:** For Jassid lesions, precision reaches **$66.67\%$** ($2/3$ of predictions correct) at $T=0.30$, with only 3 false positives remaining. However, higher thresholds sharply reduce recall from $55\%$ ($T=0.05$) to $30\%$ ($T=0.30$) and $5\%$ ($T=0.50$). The optimal operational balance for Jassids is $T \in [0.15, 0.30]$.

### Class 1: Curl Virus (4 Ground-Truth Boxes)
- **Results across all thresholds:** $TP = 0$, $\text{Precision} = 0.0000$, $\text{Recall} = 0.0000$.
- **False Positive Elimination:** Raising the threshold from $0.05 \to 0.40$ reduces false positives from $67 \to 5$ (a **$92.54\%$ reduction**).

### Class 2: Herbicide Growth Damage (15 Ground-Truth Boxes)
- **Results across all thresholds:** $TP = 0$, $\text{Precision} = 0.0000$, $\text{Recall} = 0.0000$.
- **False Positive Elimination:** All 14 baseline false positives had confidence $< 0.20$. At $T \ge 0.20$, **$FP = 0$** across the entire validation split.

### Class 4: Leaf Redding (18 Ground-Truth Boxes)

| Threshold | Retained Preds | TP | FP | FN | Precision | Recall | F1-Score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 95 | 3 | 92 | 15 | 0.0316 | 0.1667 | 0.0531 |
| 0.25 | 27 | 3 | 24 | 15 | 0.1111 | 0.1667 | **0.1333** |
| 0.35 | 12 | 0 | 12 | 18 | 0.0000 | 0.0000 | 0.0000 |

- **Key Finding:** True positive detections for Leaf Redding carry confidence scores between $0.20$ and $0.28$. Thresholds above $0.30$ extinguish all true positive detections.

### Class 5: Leaf Variegation (38 Ground-Truth Boxes)

| Threshold | Retained Preds | TP | FP | FN | Precision | Recall | F1-Score |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 113 | 5 | 108 | 33 | 0.0442 | 0.1316 | 0.0662 |
| 0.20 | 48 | 3 | 45 | 35 | 0.0625 | 0.0789 | **0.0698** |
| 0.40 | 24 | 2 | 22 | 36 | 0.0833 | 0.0526 | 0.0645 |
| 0.50 | 17 | 1 | 16 | 37 | 0.0588 | 0.0263 | 0.0364 |

- **Key Finding:** Variegation exhibits severe false positive density (108 at baseline) due to light-colored venation and natural reflection. Thresholding to $0.40$ reduces false positives to 22, but recall is constrained to $5.26\%$.

---

## 7. False-Positive and False-Negative Failure Mode Analysis

Multi-threshold comparative panels were generated for forensic inspection:
- `artifacts/visual_qa/phase9_2c_1_threshold_analysis/threshold_comp_val_01_BBC00130.png`
- `artifacts/visual_qa/phase9_2c_1_threshold_analysis/threshold_comp_val_05_LHJ00093.png`
- `artifacts/visual_qa/phase9_2c_1_threshold_analysis/threshold_comp_val_13_LR00039.png`

### 1. Background and Vein-Shadow False Positives
- **Mechanism:** Natural shadow crevices along primary veins and leaf fold contours generate low-contrast boundaries that RPN proposals flag as potential necrotic lesions.
- **Confidence Profile:** These artifacts consistently register low confidence scores between $0.05$ and $0.18$.
- **Resolution via Thresholding:** Setting $T \ge 0.25$ effectively eliminates over $80\%$ of shadow-induced false positives across healthy control images (`HL00087.jpg` and `HL00039.jpg`).

### 2. Small Lesion Sensitivity vs Bounding Box Granularity
- **Mechanism:** Bacterial Blight lesions often appear as dense clusters of tiny angular spots ($< 15\text{ px}$ diameter).
- **Behavior:** Human annotators in Tasks #4 and #5 frequently grouped 3–5 proximate spots into a single encompassing bounding box. The Faster R-CNN detector frequently predicts individual tight boxes around each sub-spot.
- **Metric Impact:** Because individual predicted sub-boxes do not achieve $\text{IoU} \ge 0.50$ against the larger composite ground-truth box, they are penalized as $FP$, even though the network detected the genuine lesion.

### 3. Diffuse Disease Symptoms (Curl Virus & Herbicide Damage)
- **Mechanism:** Diseases causing physiological distortion (leaf curling, cupping, vein clearing) do not have localized focal margins.
- **Failure Mode:** Faster R-CNN's RoI heads are optimized for discrete object localization. When confronted with diffuse symptoms, the model produces broad, low-confidence boxes that fail IoU matching.

### 4. Cross-Class Misclassifications
- **Observation:** On red/purple leaf edges (`LR00530.jpg`), the model occasionally cross-predicts Bacterial Blight or Leaf Hopper Jassids.
- **Confidence Profile:** These misclassified proposals typically exhibit confidence $< 0.22$ and are filtered out at $T \ge 0.25$.

---

## 8. Recommended Operating Threshold

Based on measured validation trade-offs, two operating points are supported by empirical evidence:

```
                    Threshold Trade-Off Space
High Recall Focus                                   High Precision Focus
(T = 0.25)                                          (T = 0.40)
-------------------------------------------------------------------------
Precision: 22.06%                                   Precision: 30.30%
Recall:    31.47%                                   Recall:    27.97%
F1-Score:  0.2594                                   F1-Score:  0.2909 (PEAK)
Retained:  204 preds (13.6/img)                     Retained:  132 preds (8.8/img)
Blight Rec: 70.83%                                  Blight Rec: 70.83%
Jassid Prec: 54.55%                                 Jassid Prec: 66.67%
```

1. **Balanced Screening Operating Point ($T = 0.25$):**
   - Retains $31.47\%$ overall recall (45/143 lesions).
   - Retains **$70.83\%$ Bacterial Blight recall** ($34/48$) and **$30.00\%$ Jassid recall**.
   - Cuts total false alarms by **$66.38\%$** ($473 \to 159$).
   - Appropriate for preliminary field screening where missing a diseased plant is costlier than an alert.
2. **Optimal F1 Diagnostic Operating Point ($T = 0.40$):**
   - Achieves the global peak **$F1 = 0.2909$**.
   - Achieves the highest precision (**$30.30\%$**), purging **$80.55\%$ of false alarms** ($473 \to 92$).
   - Retains full **$70.83\%$ Bacterial Blight recall** ($34/48$ lesions).
   - Achieves **$66.67\%$ precision on Leaf Hopper Jassids**.
   - Appropriate for targeted automated bounding-box rendering in user interfaces.

---

## 9. Known Evaluation Limitations

1. **Small Sample Size:** The validation split consists of only 15 images. While standard for pilot exploration, confidence thresholds calibrated on 15 images have wide statistical confidence bounds.
2. **Single-Box Matching Assumption:** The Pascal VOC / COCO IoU 0.50 matching rule penalizes multiple tight predictions on a single grouped annotation.
3. **Thresholding Does Not Substitute for Training Data:** Increasing the confidence threshold purges false positives, but cannot recover true positives for under-represented classes (Curl Virus, Herbicide Damage). Only Phase 9.3 dataset expansion can address representation deficits.

---

## 10. Verification of Invariants

- **Zero Training Executed:** Exactly 0 epochs, 0 optimizer updates, and 0 backward passes occurred during Phase 9.2C.1. The model was strictly loaded in evaluation mode (`torch.no_grad()`).
- **Sealed Test Set Strictly Preserved:** Manifest `artifacts/manifests/splits/test.csv` (321 images, SHA-256 `6025D86F...`) was **not opened, not decoded, not evaluated, and not touched**.
- **Historical Assets Untouched:**
  - Classifier (`best_model.pt`): `552BABA8...` [MATCH]
  - U-Net (`exp4b_w20_best_unet.pt`): `C6788F79...` [MATCH]
  - Detector (`best_faster_rcnn.pt`): `01D82B32...` [MATCH]
  - Database (`cotton_leaf_ai.db`): Unmodified.

---

## 11. Final System Status

In accordance with strict project governance:
- **Detector Status:** `EXPERIMENTAL / NOT APPROVED FOR PRODUCTION`.
- **Pipeline Integration:** Prohibited until full dataset expansion (Phase 9.3) is executed and approved.
- **Phase 9.3 / CVAT Task #6:** Unmodified pending user authorization.
