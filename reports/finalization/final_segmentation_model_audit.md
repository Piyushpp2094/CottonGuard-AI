# CottonGuard AI — Final Segmentation Model Quality & Model Selection Audit

**Document Reference:** `reports/finalization/final_segmentation_model_audit.md`  
**Production Checkpoint:** `artifacts/checkpoints/segmentation/best_unet.pt`  
**Production SHA-256:** `612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43`  
**Experimental 5-Fold Checkpoints:** `artifacts/checkpoints/segmentation/5fold_cv/fold_{0..4}_best.pt`  
**Evaluation Standard:** Final B.Tech Model Selection & Agronomic Safety Audit  
**Date:** November 19, 2026  
**Auditor:** Senior Computer Vision Researcher & Machine Learning Engineer  
**Status:** **RETAIN PRODUCTION CHECKPOINT — HONEST ACADEMIC TRANSPARENCY**

---

## 1. Executive Summary

A critical engineering and scientific evaluation of semantic segmentation in CottonGuard AI was performed. Semantic segmentation performs a dual objective:
1. **Class 1 (Leaf Lamina):** Delineates the cotton leaf boundary from complex soil/foliage backgrounds.
2. **Class 2 (Diseased Lesion):** Identifies symptomatic necrotic, chlorotic, or variegated tissue on the lamina.

The production model `best_unet.pt` (U-Net with ResNet-34 encoder) demonstrates high-accuracy leaf segmentation (**Dice: 0.8656, IoU: 0.8056**) with exceptional specificity on healthy foliage (**1.11% false positive rate**). However, fine-grained lesion segmentation exhibits low macro overlap (**Macro Dice: 0.1498**).

An experimental 5-fold cross-validation study was executed across all 70 development images to determine whether retraining could improve lesion segmentation. Out-of-fold inference revealed that the retrained candidate caused a catastrophic surge in healthy leaf false positives (**19.79% false positive noise**, averaging 80,143 erroneous pixels per healthy leaf) and degraded Leaf Dice below acceptable safety thresholds (**0.7983**).

Following strict pre-established engineering decision rules, the candidate was **REJECTED**, and the original production checkpoint `best_unet.pt` is **RETAINED AS THE PRODUCTION STANDARD**.

---

## 2. Production Model Architecture & Pipeline

### A. Network Architecture
- **Model Topology:** U-Net with ResNet-34 Encoder (SMP implementation, `smp.Unet`)
- **Encoder Weights:** Pretrained ImageNet-1K (`resnet34`)
- **Decoder Channels:** $[256, 128, 64, 32, 16]$ with skip connections and transposed convolutions
- **Input Channels:** 3 (RGB), normalized via ImageNet standards
- **Output Classes:** 3 discrete classes:
  - `0`: Background (soil, air, agricultural ground clutter)
  - `1`: Healthy Leaf Tissue
  - `2`: Diseased / Symptomatic Lesion Tissue
- **Resolution:** Input resized to $256 \times 256$ for model forward pass, outputs upsampled back to original image dimensions ($W \times H$) using nearest-neighbor interpolation.

### B. Morphological Post-Processing Pipeline
To eliminate discretization artifacts and enforce biological plausibility, `PredictionService._post_process_segmentation_mask` executes three sequential post-processing operations:
1. **Hole Filling (`ndimage.binary_fill_holes`):** Closes internal voids or lacunae within the leaf lamina mask caused by specular reflection.
2. **Connected-Component Noise Suppression (`cv2.connectedComponentsWithStats`):** Eliminates tiny isolated lesion artifacts smaller than $A_{\min} = 50\text{ pixels}$.
3. **Strict Containment Enforcement:** Computes logical intersection $\text{Disease} = \text{Disease} \land \text{Leaf}$, ensuring no symptomatic pixel can ever exist outside the leaf canopy.

---

## 3. Production Baseline Empirical Metrics

Evaluated on the canonical validation split (`baseline_segmentation_metrics.json`):

### A. Primary Segmentation Overlap
| Target Structure | Macro Dice | Macro IoU | Micro Precision | Micro Recall | Micro Dice |
|---|:---:|:---:|:---:|:---:|:---:|
| **Leaf Lamina (Class 1)** | **0.8656** | **0.8056** | 0.9124 | 0.8841 | 0.8980 |
| **Diseased Region (Class 2)** | **0.1498** | **0.1096** | 0.4588 | 0.1312 | 0.2024 |

### B. Per-Disease Lesion Dice Breakdown
| Disease Category | Lesion Dice | Saliency & Visual Characteristics |
|---|:---:|---|
| **Leaf Variegation** | **0.5355** | Broad, continuous chlorotic/yellow blotches across leaf surface. |
| **Leaf Redding** | **0.3037** | Expansive marginal bronze/purple anthocyanin accumulation. |
| **Leaf Hopper Jassids** | **0.2334** | Marginal hopperburn browning along peripheral leaf sectors. |
| **Bacterial Blight** | **0.0490** | Tiny, angular, vein-bounded micro-water-soaked spots ($5\text{--}30\text{ px}$). |
| **Curl Virus** | **0.0225** | Systemic leaf distortion, curling, and vein swelling (few focal spots). |
| **Herbicide Damage** | **0.0050** | Phenotypic strap-leaf/cupping deformity with minimal necrotic mass. |

### C. Specificity & False Positive Rate
- **Healthy Leaf False Positive Rate:** **1.11%** (Only 4,820 px average spurious noise on pristine healthy foliage).
- **Clinical Significance:** Excellent specificity ensures healthy leaves are not falsely alarmed as diseased crops.

---

## 4. Experimental 5-Fold Cross-Validation Study

To investigate whether expanding training to all 70 development images could elevate lesion Dice without sacrificing leaf precision, a 5-fold stratified cross-validation was conducted (`scripts/run_5fold_segmentation_experiments.py`) on the NVIDIA RTX 3050 GPU.

### A. 5-Fold Out-of-Fold (OOF) Inference Results
Across all 5 folds ($k=0$ through $4$, 14 holdout images per fold):

| Evaluation Metric | Production Baseline (`best_unet.pt`) | 5-Fold OOF Candidate | Net Delta ($\Delta$) |
|---|:---:|:---:|:---:|
| **Leaf Macro Dice** | **0.8656** | 0.7983 | $-0.0673$ *(Degraded)* |
| **Leaf Macro IoU** | **0.8056** | 0.7241 | $-0.0815$ *(Degraded)* |
| **Lesion Macro Dice** | 0.1498 | **0.1639** | $+0.0141$ *(Marginal)* |
| **Lesion Micro Dice** | 0.2024 | **0.3003** | $+0.0979$ *(Apparent)* |
| **Healthy Leaf False Positive Rate** | **1.11%** | **19.79%** | **$+18.68\%$ *(Severe Degradation)*** |
| **Healthy Leaf Mean FP Pixels** | **4,820 px** | **80,143 px** | **$+75,323\text{ px}$ (16.6x Surge)** |
| **Bacterial Blight Lesion Dice** | **0.0490** | 0.0180 | $-0.0310$ *(Severely Dropped)* |

### B. Root Cause Discovery: The Batch 03 Annotation Disparity
Visual error analysis (`generate_segmentation_error_analysis.py`) and COCO polygon inspection uncovered the fundamental data root cause behind this phenomenon:

1. **Batches 01 and 02 Annotation Standard:** Annotators marked Curl Virus and Herbicide Damage leaves as *Healthy Leaf* (Class 1) and isolated only tiny necrotic flecks as *Disease* (Class 2), yielding disease area percentages of $0.5\% \text{ to } 4.0\%$.
2. **Batch 03 Annotation Standard:** Annotators re-annotated Curl Virus and Herbicide Damage by classifying the **entire deformed leaf** as *Diseased Region* (Class 2), yielding affected area percentages of $65\% \text{ to } 99\%$.
3. **Model Conflict:** When the 5-fold models were exposed to Batch 03 annotations, the loss gradient compelled the network to predict Class 2 over large areas of green lamina. Consequently, when presented with normal, healthy leaves, the retrained model hallucinated massive patches of disease (**19.79% of healthy leaves misidentified as diseased**).

---

## 5. Formal Model Selection Audit Against Pre-Established Rules

| Rule ID | Decision Criterion | Threshold | Production Baseline | 5-Fold Candidate | Rule Status |
|:---:|---|:---:|:---:|:---:|:---:|
| **Rule 1** | Healthy Leaf False Positive Rate | $\le 3.0\%$ | **1.11%** | 19.79% | **FAIL (Candidate Breaches Safety)** |
| **Rule 2** | Bacterial Blight Lesion Dice | $\ge 0.0490$ | **0.0490** | 0.0180 | **FAIL (Candidate Regressed)** |
| **Rule 3** | Leaf Lamina Macro Dice | $\ge 0.8500$ | **0.8656** | 0.7983 | **FAIL (Candidate Dropped Below 0.85)** |
| **Rule 4** | Lesion Macro Dice Improvement | $\ge +0.0500$ | 0.1498 | 0.1639 (+0.0141) | **FAIL (Insignificant Improvement)** |
| **Rule 5** | Reproducibility & Stability | Across all 5 folds | Preserved | Folds 0–4 preserved | PASS |
| **Rule 6** | Post-Processing Parameter | $A_{\min} = 50\text{ px}$ | Confirmed | Confirmed | PASS |
| **Rule 7** | Checkpoint Cryptographic Safety | Inviolable hash | Verified | Archived separately | PASS |

---

## 6. Model Selection Verdict & Engineering Decision

**VERDICT: REJECT CANDIDATE — RETAIN PRODUCTION BASELINE `best_unet.pt`**

### Technical Justification
In agricultural diagnostics, a model that falsely paints healthy crops with 20% diseased lesions destroys farmer trust and leads to unnecessary chemical pesticide applications. The production checkpoint `best_unet.pt`:
- Accurately segments cotton leaf canopies (0.8656 Dice).
- Preserves clean healthy leaves (only 1.11% false positive noise).
- Is coupled with Grad-CAM explainability and EfficientNet-B0 classification (98.43% accuracy) to provide reliable crop diagnosis.

All five experimental fold models remain preserved in `artifacts/checkpoints/segmentation/5fold_cv/` for research auditing. The production system strictly serves `best_unet.pt`.
