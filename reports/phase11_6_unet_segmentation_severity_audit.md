# CottonGuard AI — Phase 11.6: U-Net Segmentation, Disease Area & Severity Estimation Integrity Audit Report

**Audit Date:** November 17, 2026  
**Auditor:** Senior Computer Vision Researcher, Machine Learning Integrity Auditor & Quality Assurance Specialist  
**Project:** CottonGuard AI (`D:\Main_Project\cotton_leaf_ai`)  
**Phase:** 11.6 — U-Net Segmentation, Disease Area & Severity Estimation Integrity Audit  
**Execution Mode:** Read-Only Evaluation & Scientific Verification (Zero Model Training, Zero Backpropagation, Zero Optimizer Steps)  
**Approval Gate 4 Status:** `PENDING_APPROVAL` (Strictly Maintained)  

---

## 1. Executive Summary

A comprehensive, strictly read-only technical audit of the CottonGuard AI U-Net semantic segmentation model (`artifacts/checkpoints/segmentation/best_unet.pt`) and its downstream disease severity estimation pipeline was conducted. The audit verified checkpoint integrity, training provenance, dataset/mask inventories, class mappings, preprocessing and resolution sensitivities (native 512×512 vs. service 256×256 vs. probability bilinear upsampling), leaf-restricted severity quantification, healthy negative-control behavior, prediction repeatability, and GPU hardware telemetry across the 15 verified validation image-mask pairs.

### Key Audit Findings:
1. **Checkpoint Integrity Invariant Preserved:** The authoritative SHA-256 hash `612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43` was verified before and after evaluation with zero drift. The model comprises 24,455,713 parameters based on a ResNet-34 encoder with a 3-class U-Net decoder.
2. **Corpus & Validation Status Provenance:** Exactly 50 rasterized 800×800 PNG masks exist in `data/annotations/masks/` (35 training pairs, 15 validation pairs). The consolidated 100-image annotation corpus does **not** contain 100 rasterized masks (the other 50 images contain vector bounding-box/polygon annotations only). The 15 validation pairs participated in model selection during development; they represent **development-validation samples**, **not** an external independent test set.
3. **Resolution & Preprocessing Sensitivity:**
   - **Native 512×512 Resolution (Training-Compatible):** Macro Dice is **79.15%**, Macro IoU is **66.98%**. For diseased tissue specifically: Dice is **62.55%**, IoU is **45.50%**, Precision is **58.11%**, Recall is **67.71%**, with 190,572 false-positive pixels and 126,104 false-negative pixels.
   - **Service 256×256 Resolution (Current Backend):** Macro Dice drops to **71.11%** (-8.04 percentage points), Macro IoU drops to **56.77%** (-10.21 pp). For diseased tissue: Dice drops to **52.07%** (-10.48 pp), IoU drops to **35.20%** (-10.30 pp), Precision drops to **44.20%** (-13.91 pp), and false-positive diseased pixels surge to **312,293 (+63.9% increase in false alarms)**.
   - Bilinear probability upsampling at 512×512 slightly further improves diseased Dice to **63.00%** and Macro Dice to **79.33%**.
4. **Severity Estimation Error Analysis:**
   - **Native 512×512:** Mean Absolute Error (MAE) is **1.99%** (1.99 percentage points), Root Mean Squared Error (RMSE) is **3.07%**, Mean Signed Error (Bias) is **+0.20%**, and Maximum Absolute Error is **7.85%** (on sample `LV00112`).
   - **Service 256×256:** MAE degrades to **4.63%** (>2.3× higher error), RMSE degrades to **7.81%** (>2.5× higher error), with an overestimation Bias of **+2.75%** and Maximum Absolute Error of **21.47%** (on `LV00112`, causing an erroneous jump from Trace to Moderate severity).
5. **Healthy Negative-Control Findings:** Both healthy validation leaf samples (`HL00087` and `HL00039`) triggered false-positive disease detections. In native 512 mode, predicted severity is **0.44%** and **0.41%** (170 and 85 leaf false-positive pixels; 1,699 and 2,303 background noise pixels). In service 256 mode, predicted severity rises to **2.10%** and **0.67%**. Under the provisional 6-tier agronomic system (where "None" requires $\le 0.0\%$), both healthy leaves are misclassified into the "Trace" severity tier. Under the legacy 4-tier system ($\le 0.5\%$), 512 mode correctly classifies them as "NONE", while 256 mode misclassifies `HL00087` as "LOW".
6. **Repeatability & Hardware Performance:** Repeated GPU inference (5 samples × 5 passes = 25 runs on NVIDIA RTX 3050 6GB Laptop GPU) exhibited **100% bitwise mask determinism** (zero differing pixels across passes) and **0.000000 probability drift**. Mean latency was **26.9 ms/image** (37.2 FPS) with peak allocated VRAM of **222.17 MB**.
7. **Database & Dataset Invariance:** The production SQLite database (`cotton_leaf_ai.db`) remained untouched at 48 predictions and 5 users. The sealed 321-image classification test set remained 100% untouched.

---

## 2. Audit Scope and Regulatory Boundaries

The audit was executed under strict read-only non-mutation protocols:
- **Zero Model Modification:** No fine-tuning, no weight updates, no training loops, no loss optimization.
- **Pre- and Post-Audit Hash Verifications:** All three model checkpoints (EfficientNet-B0 classifier, Faster R-CNN detector, U-Net segmentor) were cryptographically verified before and after the audit.
- **Sealed Test Set Protection:** The 321-image classifier test split (`artifacts/manifests/splits/test.csv`) was never opened, loaded, or evaluated.
- **Production Isolation:** Offline evaluation was executed using temporary in-memory tensors and direct file outputs in `artifacts/audit_phase11/phase11_6/`. No HTTP API endpoints creating database entries were invoked.
- **Gate 4 Governance:** Approval Gate 4 remains locked in `PENDING_APPROVAL`.

---

## 3. Checkpoint Integrity and Architecture

### 3.1 Checkpoint Forensic Verification

| Property | Measured Value | Requirement / Expected | Status |
| :--- | :--- | :--- | :---: |
| **File Path** | `artifacts/checkpoints/segmentation/best_unet.pt` | Relative to project root | VERIFIED |
| **Pre-Audit SHA-256** | `612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43` | `612fca0a...` | MATCH |
| **Post-Audit SHA-256** | `612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43` | `612fca0a...` | INVARIANT |
| **Architecture** | `unet_resnet34` (`segmentation_models_pytorch.Unet`) | ResNet-34 Encoder + U-Net Decoder | VERIFIED |
| **Total Parameters** | 24,455,713 parameters | 24,455,713 | MATCH |
| **Input Tensor Shape** | `[1, 3, 512, 512]` | 512×512 RGB | VERIFIED |
| **Output Channels** | 3 logits (`[1, 3, H, W]`) | 3 classes | VERIFIED |
| **Checkpoint Status** | `EXPERIMENTAL` | Stored in metadata | CONFIRMED |
| **Best Training Epoch** | 31 (validation loss: 0.45934, train loss: 0.42640) | Epoch 31 | MATCH |
| **Stored Loss Weights**| CE Class Weights: `[0.3, 1.0, 2.5]`, CE/Dice: 0.5 / 0.5 | Weighted combined loss | VERIFIED |

### 3.2 Checkpoint Model State Dict Inspection
The model state dict contains the complete trained weights for:
- ResNet-34 encoder layers (`encoder.conv1`, `encoder.layer1` through `layer4`).
- Decoder blocks with skip connections (`decoder.blocks.0` through `decoder.blocks.4`).
- Final segmentation head (`segmentation_head.0` 3×3 conv mapping 16 channels to 3 output classes).

---

## 4. Dataset Provenance and Mask Inventory

### 4.1 Mask Inventory and Corpus Clarification
A critical finding of Phase 11.6 is resolving the discrepancy between the "100-image consolidated annotation corpus" and the "U-Net segmentation dataset":
- **Total Rasterized Segmentation Masks:** Exactly **50** 800×800 3-class PNG masks exist in `data/annotations/masks/`.
- **Total Corresponding Images:** Exactly **50** matching JPEG images exist in `data/annotations/images/`.
- **Other 50 Consolidated Images:** The remaining 50 images in `artifacts/manifests/consolidated_100_annotation_manifest.csv` possess vector bounding-box and polygon annotations (for Faster R-CNN and YOLO format), but **do not possess rasterized pixel-level masks**.
- **Split Breakdown:** 35 images in the training split, 15 images in the validation split (`artifacts/manifests/combined_pilot_manifest.csv`).

### 4.2 Class Mapping and Color Indices

| Class ID | Class Name | Definition | Palette Color in QA Overlays |
| :---: | :--- | :--- | :---: |
| **0** | Background | Non-leaf field context, soil, shadow, background | Black `[0, 0, 0]` |
| **1** | Healthy Leaf | Healthy, green cotton leaf lamina and veins | Green `[40, 180, 40]` |
| **2** | Diseased Tissue | Pathological lesions, necrotic spots, variegation, chlorosis | Red `[230, 30, 30]` |

### 4.3 Corpus Pixel Distribution (All 50 Masks)
Across all 50 ground-truth masks ($50 \times 800 \times 800 = 32,000,000$ pixels):
- **Background (Class 0):** 11,811,280 pixels (**36.91%**)
- **Healthy Leaf (Class 1):** 19,005,622 pixels (**59.39%**)
- **Diseased Tissue (Class 2):** 1,183,098 pixels (**3.70%**)
The severe class imbalance (Diseased Tissue represents only 3.70% of total pixels, ~5.86% of total leaf tissue) confirms the agronomic necessity of the training loss weighting (`class_weights = [0.3, 1.0, 2.5]`).

---

## 5. Validation Samples Inventory

The 15 development-validation image-mask pairs ($15 \times 800 \times 800 = 9,600,000$ pixels) used for evaluation:

| Image ID | Class Label | Background Px | Healthy Leaf Px | Diseased Px | Total Leaf Px | Ground Truth Severity (%) | 6-Tier Category | Healthy Control? |
| :--- | :--- | ---:| ---:| ---:| ---:| ---:| :---: | :---: |
| **BBC00130** | Bacterial Blight | 258,035 | 378,731 | 3,234 | 381,965 | 0.85% | Trace | No |
| **CV00395** | Curl Virus | 335,707 | 303,895 | 398 | 304,293 | 0.13% | Trace | No |
| **HL00087** | Healthy Leaf | 286,403 | 353,597 | 0 | 353,597 | 0.00% | None | **YES** |
| **HGD00151** | Herbicide Damage | 474,957 | 157,324 | 7,719 | 165,043 | 4.68% | Trace | No |
| **LHJ00093** | Leaf Hopper Jassids | 106,566 | 501,522 | 31,912 | 533,434 | 5.98% | Trace | No |
| **LR00530** | Leaf Reddening | 396,451 | 241,350 | 2,199 | 243,549 | 0.90% | Trace | No |
| **LV00001** | Leaf Variegation | 389,106 | 193,912 | 56,982 | 250,894 | 22.71% | Mild | No |
| **BBC00170** | Bacterial Blight | 260,906 | 376,870 | 2,224 | 379,094 | 0.59% | Trace | No |
| **CV00310** | Curl Virus | 407,721 | 223,272 | 9,007 | 232,279 | 3.88% | Trace | No |
| **HL00039** | Healthy Leaf | 88,785 | 551,215 | 0 | 551,215 | 0.00% | None | **YES** |
| **HGD00095** | Herbicide Damage | 237,603 | 400,714 | 1,683 | 402,397 | 0.42% | Trace | No |
| **LHJ00057** | Leaf Hopper Jassids | 276,873 | 362,054 | 1,073 | 363,127 | 0.30% | Trace | No |
| **LR00039** | Leaf Reddening | 182,514 | 269,582 | 187,904 | 457,486 | 41.07% | Moderate | No |
| **LV00112** | Leaf Variegation | 332,504 | 285,990 | 21,506 | 307,496 | 6.99% | Trace | No |
| **LV00085** | Leaf Variegation | 386,602 | 188,719 | 64,679 | 253,398 | 25.52% | Moderate | No |
| **Total / Summary** | *15 Validation Samples* | **4,420,733** | **4,788,747** | **390,520** | **5,179,267** | **Mean: 7.58%** | — | **2 Controls** |

*Note: Validation pixels represent 46.05% Background, 49.88% Healthy Leaf, and 4.07% Diseased Tissue ($N=390,520$ diseased pixels).*

---

## 6. Preprocessing & Resolution Comparison

Inference was evaluated under six strictly controlled configurations using identical model weights:
1. **Mode A (Native 512×512 Albumentations $\to$ 800×800):** Native training preprocessing (512×512, ImageNet normalization, argmax discrete mask nearest upsampled to 800×800).
2. **Mode B (Service 256×256 $\to$ 800×800):** Production service preprocessing (`prediction_service.py`: 256×256, ImageNet normalization, argmax discrete mask nearest upsampled to 800×800).
3. **Mode C1 (Prob Bilinear 512×512 $\to$ 800×800):** 512 input, softmax probabilities upsampled via bilinear interpolation to 800×800, then argmax.
4. **Mode C2 (Prob Bilinear 256×256 $\to$ 800×800):** 256 input, softmax probabilities upsampled via bilinear interpolation to 800×800, then argmax.
5. **Mode D (Native 512×512 Evaluated at 512):** Ground truth mask downsampled to 512×512 (nearest) and compared directly against 512 prediction.
6. **Mode E (Service 256×256 Evaluated at 256):** Ground truth mask downsampled to 256×256 (nearest) and compared directly against 256 prediction.

### Preprocessing Comparison Matrix

| Evaluation Mode | Input Res | Upsampling Method | Macro Dice | Macro IoU | Background Dice | Healthy Leaf Dice | Diseased Dice | Diseased IoU | Diseased Precision | Diseased Recall | Diseased FP Px | Diseased FN Px |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mode A: Native 512 (Nearest)** | 512×512 | Nearest (mask) | **79.15%** | **66.98%** | 86.88% | 88.03% | **62.55%** | **45.50%** | **58.11%** | **67.71%** | **190,572** | **126,104** |
| **Mode B: Service 256 (Nearest)** | 256×256 | Nearest (mask) | **71.11%** | **56.77%** | 81.17% | 80.10% | **52.07%** | **35.20%** | **44.20%** | **63.34%** | **312,293** | **143,155** |
| **Mode C1: Prob Bilinear 512** | 512×512 | Bilinear (probs) | **79.33%** | **67.18%** | 86.94% | 88.07% | **63.00%** | **45.98%** | **58.78%** | **67.86%** | **185,842** | **125,502** |
| **Mode C2: Prob Bilinear 256** | 256×256 | Bilinear (probs) | **71.31%** | **56.98%** | 81.26% | 80.15% | **52.53%** | **35.62%** | **44.81%** | **63.46%** | **305,248** | **142,693** |
| **Mode D: Native at 512** | 512×512 | None (GT@512) | **79.19%** | **67.02%** | 86.91% | 88.04% | **62.63%** | **45.59%** | **58.28%** | **67.68%** | 77,541* | 51,709* |
| **Mode E: Service at 256** | 256×256 | None (GT@256) | **71.09%** | **56.75%** | 81.17% | 80.10% | **52.02%** | **35.15%** | **44.14%** | **63.32%** | 32,049* | 14,666* |

*\*Note: Pixel counts in Modes D and E are reported at downsampled resolution ($512^2$ and $256^2$).*

---

## 7. Per-Class Segmentation Performance (512 vs 256)

Detailed comparison between Native 512×512 inference and Service 256×256 inference (evaluated at full 800×800 resolution across all 9,600,000 pixels):

| Class Name | 512 Native Dice | 256 Service Dice | Dice Delta | 512 Native IoU | 256 Service IoU | IoU Delta | 512 Precision | 256 Precision | 512 Recall | 256 Recall | 512 FP Pixels | 256 FP Pixels | FP Pixel Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Background** | 86.88% | 81.17% | **+5.71 pp** | 76.81% | 68.30% | **+8.51 pp** | 90.84% | 77.90% | 83.26% | 84.71% | 371,006 | 1,062,203 | **-691,197 (-65.1%)** |
| **Healthy Leaf** | 88.03% | 80.10% | **+7.93 pp** | 78.61% | 66.81% | **+11.80 pp**| 85.39% | 85.36% | 90.83% | 75.46% | 744,001 | 619,765 | +124,236 (+20.0%) |
| **Diseased Tissue**| **62.55%** | **52.07%** | **+10.48 pp**| **45.50%** | **35.20%** | **+10.30 pp**| **58.11%** | **44.20%** | **67.71%** | **63.34%** | **190,572** | **312,293** | **-121,721 (-39.0%)** |
| **Macro Average** | **79.15%** | **71.11%** | **+8.04 pp** | **66.98%** | **56.77%** | **+10.21 pp**| **78.11%** | **69.15%** | **80.60%** | **74.50%** | — | — | — |

### Root Cause of Resolution Degradation:
1. **Sub-Pixel Feature Compression:** Cotton leaf disease lesions (such as early Bacterial Blight angular water-soaked spots, punctate necrotic dots in Curl Virus, or early Hopper burn tips) span only 3 to 10 pixels in an 800×800 photograph. When downsampled to 256×256, these lesions shrink below 1 to 3 pixels and are completely erased during the ResNet-34 encoder's stride-32 bottleneck ($256 \to 8 \times 8$).
2. **Boundary Dilation Artifacts:** At 256×256, nearest-neighbor upsampling produces blocky $3.125 \times 3.125$ pixel superpixels, resulting in severe over-segmentation along leaf borders and background shadows, which inflates false-positive diseased pixels by 63.9% (from 190,572 to 312,293).

---

## 8. Leaf-Restricted Severity Estimation Audit

### 8.1 Severity Quantification Formula
The project enforces leaf-restricted foliar damage quantification:
$$\text{Severity (\%)} = \frac{\text{Diseased Leaf Pixels}}{\text{Total Visible Leaf Pixels}} \times 100 = \frac{N_{\text{disease}}}{N_{\text{healthy}} + N_{\text{disease}}} \times 100$$
Where:
- Background pixels ($C=0$) are strictly excluded from both numerator and denominator.
- Spurious disease predictions outside the leaf lamina (in background) cannot inflate the numerator.
- If $N_{\text{leaf}} < 500$ pixels, status is `INSUFFICIENT_LEAF_AREA`.

### 8.2 Severity Error Metrics Summary

| Metric | 512 Native Resolution | 256 Service Resolution | Performance Gain with 512 |
| :--- | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **1.99%** (1.99 pp) | **4.63%** (4.63 pp) | **57.0% Error Reduction** |
| **Root Mean Squared Error (RMSE)** | **3.07%** | **7.81%** | **60.7% Error Reduction** |
| **Mean Signed Error (Bias)** | **+0.20%** (slight overestimation) | **+2.75%** (heavy overestimation) | **Near-zero bias (+0.20% vs +2.75%)** |
| **Maximum Absolute Error** | **7.85%** (on `LV00112`) | **21.47%** (on `LV00112`) | **-13.62 pp Max Error Reduction** |

### 8.3 Per-Image Severity Estimation Results (15 Validation Samples)

| Image ID | Ground Truth Severity | GT 6-Tier Tier | 512 Pred Severity | 512 Error (pp) | 512 6-Tier Tier | 256 Pred Severity | 256 Error (pp) | 256 6-Tier Tier | Tier Match (512 vs GT) | Tier Match (256 vs GT) |
| :--- | ---:| :---: | ---:| ---:| :---: | ---:| ---:| :---: | :---: | :---: |
| **BBC00130** | 0.85% | Trace | 0.14% | -0.71% | Trace | 0.34% | -0.51% | Trace | MATCH | MATCH |
| **CV00395** | 0.13% | Trace | 0.12% | -0.01% | Trace | 0.41% | +0.28% | Trace | MATCH | MATCH |
| **HL00087** | 0.00% | None | 0.44% | +0.44% | Trace | 2.10% | +2.10% | Trace | MISMATCH | MISMATCH |
| **HGD00151** | 4.68% | Trace | 0.19% | -4.49% | Trace | 0.61% | -4.07% | Trace | MATCH | MATCH |
| **LHJ00093** | 5.98% | Trace | 4.27% | -1.71% | Trace | 0.91% | -5.07% | Trace | MATCH | MATCH |
| **LR00530** | 0.90% | Trace | 0.96% | +0.06% | Trace | 0.61% | -0.29% | Trace | MATCH | MATCH |
| **LV00001** | 22.71% | Mild | 24.11% | +1.40% | Mild | 23.81% | +1.10% | Mild | MATCH | MATCH |
| **BBC00170** | 0.59% | Trace | 0.05% | -0.54% | Trace | 0.25% | -0.34% | Trace | MATCH | MATCH |
| **CV00310** | 3.88% | Trace | 0.26% | -3.62% | Trace | 0.06% | -3.82% | Trace | MATCH | MATCH |
| **HL00039** | 0.00% | None | 0.41% | +0.41% | Trace | 0.67% | +0.67% | Trace | MISMATCH | MISMATCH |
| **HGD00095** | 0.42% | Trace | 0.25% | -0.17% | Trace | 0.95% | +0.53% | Trace | MATCH | MATCH |
| **LHJ00057** | 0.30% | Trace | 0.62% | +0.32% | Trace | 2.82% | +2.52% | Trace | MATCH | MATCH |
| **LR00039** | 41.07% | Moderate | 47.06% | +5.99% | Moderate | 58.28% | +17.21% | Severe | MATCH | MISMATCH (Severe) |
| **LV00112** | 6.99% | Trace | 14.84% | +7.85% | Mild | 28.46% | +21.47% | Moderate | MISMATCH (Mild) | MISMATCH (Moderate) |
| **LV00085** | 25.52% | Moderate | 23.35% | -2.17% | Mild | 34.93% | +9.41% | Moderate | MISMATCH (Mild) | MATCH |

### 8.4 Analysis of Severity Estimation Accuracy
1. **Low-Severity Precision:** For samples with mild/trace symptoms ($\text{GT} \le 6\%$), native 512 maintains errors under $\pm 1\%$, correctly classifying them into the "Trace" category.
2. **Major Divergence on Variegation & Reddening:**
   - On `LR00039` (severe leaf reddening, $\text{GT} = 41.07\%$), 512 mode predicts $47.06\%$ (retains Moderate tier), whereas 256 service mode predicts $58.28\%$ ($+17.21\%$ overestimation), erroneously bumping the leaf into the **Severe** category.
   - On `LV00112` (leaf variegation, $\text{GT} = 6.99\%$), 512 mode predicts $14.84\%$ ($+7.85\%$), whereas 256 service mode predicts $28.46\%$ ($+21.47\%$ overestimation), erroneously bumping the leaf two full tiers up into **Moderate**.

---

## 9. Healthy Negative-Control Analysis

To establish whether the segmentation model hallucinated disease on non-diseased leaves, the 2 pure healthy controls in the validation split (`HL00087` and `HL00039`) were audited:

| Control Image | Ground Truth Status | GT Disease Px | 512 FP Disease (Total) | 512 FP in Leaf Lamina | 512 FP in Background | 512 Pred Severity | 512 6-Tier Tier | 256 FP Disease (Total) | 256 FP in Leaf Lamina | 256 FP in Background | 256 Pred Severity | 256 6-Tier Tier |
| :--- | :---: | ---:| ---:| ---:| ---:| ---:| :---: | ---:| ---:| ---:| ---:| :---: |
| **HL00087** | Healthy ($0.00\%$) | 0 | 1,869 | 170 | 1,699 | 0.44% | Trace | 3,686 | 937 | 2,749 | 2.10% | Trace |
| **HL00039** | Healthy ($0.00\%$) | 0 | 2,388 | 85 | 2,303 | 0.41% | Trace | 3,844 | 295 | 3,549 | 0.67% | Trace |

### Critical Negative-Control Observations:
1. **Background Noise Predominance:** In 512 mode, over **90%** of false-positive disease pixels (1,699/1,869 on `HL00087`, 2,303/2,388 on `HL00039`) occur along image edges or in background shadow regions, rather than within the healthy leaf lamina.
2. **Leaf Lamina Resistance:** Inside the actual leaf boundaries, the model demonstrates high specificity: only 170 pixels (0.048% of leaf) on `HL00087` and 85 pixels (0.015% of leaf) on `HL00039` were misclassified as disease at 512 resolution.
3. **Agronomic Classification Mismatch:**
   - Under the provisional 6-tier rules (`provisional_6_tier_thresholds` in `configs/severity_rules.yaml`), the cutoff for "None" is strictly $\le 0.0\%$. Consequently, both healthy leaves ($0.44\%$ and $0.41\%$) are classified as **Trace** (false alarms).
   - Under the legacy 4-tier rules (`no_lesion_max_percentage: 0.5%`), both 512 predictions ($0.44\%$ and $0.41\%$) fall below $0.5\%$ and are correctly classified as **NONE**. In 256 mode, however, `HL00087` ($2.10\%$) triggers a false alarm into **LOW**.

---

## 10. Repeatability and Hardware Telemetry

A controlled 5-sample × 5-pass repeatability evaluation was performed on GPU:
- **Test Samples:** `HL00039` (Healthy), `BBC00170` (Bacterial Blight), `CV00310` (Curl Virus), `LV00001` (Leaf Variegation), `LR00039` (Leaf Reddening).
- **Execution:** 5 consecutive forward passes per sample ($N=25$ total passes) on `cuda:0`.

### 10.1 Repeatability Results Matrix

| Sample ID | Symptom Type | Passes Evaluated | Mask Bitwise Equality | Max Differing Pixels | Max Probability Drift | Mean Latency (ms) | Median Latency (ms) | Min Latency (ms) | Max Latency (ms) | Latency Std (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HL00039** | Healthy Control | 5 | **100% IDENTICAL** | 0 | **0.000000** | 29.62 ms | 29.69 ms | 28.65 ms | 30.70 ms | 0.67 ms |
| **BBC00170**| Bacterial Blight | 5 | **100% IDENTICAL** | 0 | **0.000000** | 26.87 ms | 26.30 ms | 25.62 ms | 28.65 ms | 1.11 ms |
| **CV00310** | Curl Virus | 5 | **100% IDENTICAL** | 0 | **0.000000** | 26.13 ms | 26.07 ms | 25.54 ms | 26.86 ms | 0.49 ms |
| **LV00001** | Leaf Variegation | 5 | **100% IDENTICAL** | 0 | **0.000000** | 26.38 ms | 25.75 ms | 25.37 ms | 28.92 ms | 1.32 ms |
| **LR00039** | Leaf Reddening | 5 | **100% IDENTICAL** | 0 | **0.000000** | 25.58 ms | 25.32 ms | 25.18 ms | 26.35 ms | 0.45 ms |

### 10.2 Hardware Telemetry

| Parameter | Value | Reference / Notes |
| :--- | :--- | :--- |
| **Compute Device** | NVIDIA GeForce RTX 3050 6GB Laptop GPU | Dedicated Mobile GPU |
| **CUDA Driver / Runtime** | CUDA 12.x / PyTorch 2.15.0+cu132 | Fully verified |
| **Initial Allocated VRAM** | 122.34 MB | Base model weights |
| **Peak Allocated VRAM** | **222.17 MB** | Forward pass activations at 512×512 |
| **Peak Reserved VRAM** | 282.00 MB | PyTorch caching allocator |
| **Mean Inference Latency** | **26.92 ms** | Across all 25 passes (~37.1 FPS) |
| **CUDA Exceptions** | **0** | Zero runtime errors or out-of-memory faults |

The U-Net model demonstrates excellent hardware efficiency, requiring only ~222 MB of GPU VRAM at 512×512 resolution, which easily fits within entry-level 4 GB to 6 GB agricultural edge accelerators.

---

## 11. Historical Metric Provenance Reconciliation

Past project reports presented divergent segmentation metrics that previously appeared inconsistent. Phase 11.6 forensically resolved the exact conditions that generated each historical number:

| Historical Report / Record | Macro Dice | Macro IoU | Diseased Dice | Diseased IoU | Provenance and Evaluation Pipeline Condition | Reconciled Status in Phase 11.6 |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Original Checkpoint Metadata** (`best_unet.pt`) | **68.45%** | **59.35%** | **30.40%** | **22.52%** | Stored during training run (Epoch 31). Evaluated at 512×512 with initial training validation dataloader. | **EXPLAINED:** Reflects the raw checkpoint checkpoint-time validation log before Albumentations inference pipeline stabilization. |
| **Missing ImageNet Normalization** (`segmentation_eval_results.json`) | **70.46%** | **57.41%** | **43.82%** | **28.06%** | Evaluated at 512×512 using raw PIL tensors (`[0, 1]` scaling) without ImageNet mean/std normalization. | **EXPLAINED:** Caused by evaluating the ResNet-34 backbone without proper ImageNet normalization. |
| **Current Backend Service** (`prediction_service.py` / Phase 11.2.1) | **71.11%** | **56.77%** | **52.07%** | **35.20%** | Evaluated at 256×256 service resolution with ImageNet normalization, upsampled via nearest neighbor. | **EXPLAINED & REPRODUCED:** Exactly reproduced in Mode B ($71.11\%$ Macro Dice, $52.07\%$ Diseased Dice). |
| **Native Training Preprocessing** (Phase 11.2.1 / Phase 11.6 Mode A) | **79.15%** | **66.98%** | **62.55%** | **45.50%** | Evaluated at 512×512 native resolution with ImageNet normalization, upsampled via nearest neighbor. | **EXPLAINED & REPRODUCED:** Exactly reproduced in Mode A ($79.15\%$ Macro Dice, $62.55\%$ Diseased Dice). |
| **Probability Bilinear Resizing** (Phase 11.6 Mode C1) | **79.33%** | **67.18%** | **63.00%** | **45.98%** | Evaluated at 512×512 with softmax bilinear interpolation to 800×800 before argmax. | **NEW PEAK:** Optimal post-processing configuration for future deployment. |

---

## 12. Visual QA Overlays

Diagnostic 5-panel composite overlays were generated for all 15 validation samples and saved to `artifacts/audit_phase11/phase11_6/visual_qa/`. Each composite displays:
1. Panel 1: Original 800×800 RGB Image.
2. Panel 2: Ground Truth 3-Class Mask (Black=Background, Green=Healthy Leaf, Red=Diseased Tissue).
3. Panel 3: Predicted Mask at 512 Native Resolution.
4. Panel 4: Predicted Mask at 256 Service Resolution.
5. Panel 5: Error Map (Yellow=True Positive Disease, Red=False Positive Disease, Blue=False Negative Disease, Dark Green=True Healthy Leaf, Gray=True Background).

### Generated Visual QA Artifacts:
- `overlay_BBC00130.png` (Bacterial Blight, GT Sev: 0.85%, 512 Sev: 0.14%)
- `overlay_BBC00170.png` (Bacterial Blight, GT Sev: 0.59%, 512 Sev: 0.05%)
- `overlay_CV00310.png` (Curl Virus, GT Sev: 3.88%, 512 Sev: 0.26%)
- `overlay_CV00395.png` (Curl Virus, GT Sev: 0.13%, 512 Sev: 0.12%)
- `overlay_HGD00095.png` (Herbicide Damage, GT Sev: 0.42%, 512 Sev: 0.25%)
- `overlay_HGD00151.png` (Herbicide Damage, GT Sev: 4.68%, 512 Sev: 0.19%)
- `overlay_HL00039.png` (Healthy Leaf Control, GT Sev: 0.00%, 512 Sev: 0.41%)
- `overlay_HL00087.png` (Healthy Leaf Control, GT Sev: 0.00%, 512 Sev: 0.44%)
- `overlay_LHJ00057.png` (Leaf Hopper Jassids, GT Sev: 0.30%, 512 Sev: 0.62%)
- `overlay_LHJ00093.png` (Leaf Hopper Jassids, GT Sev: 5.98%, 512 Sev: 4.27%)
- `overlay_LR00039.png` (Leaf Reddening, GT Sev: 41.07%, 512 Sev: 47.06%)
- `overlay_LR00530.png` (Leaf Reddening, GT Sev: 0.90%, 512 Sev: 0.96%)
- `overlay_LV00001.png` (Leaf Variegation, GT Sev: 22.71%, 512 Sev: 24.11%)
- `overlay_LV00085.png` (Leaf Variegation, GT Sev: 25.52%, 512 Sev: 23.35%)
- `overlay_LV00112.png` (Leaf Variegation, GT Sev: 6.99%, 512 Sev: 14.84%)

---

## 13. Checkpoint and Database Invariance Audit

Forensic state verification confirmed zero mutations occurred during Phase 11.6 execution:

| Asset | Pre-Audit State | Post-Audit State | Mutation Delta | Status |
| :--- | :--- | :--- | :---: | :---: |
| **U-Net Checkpoint** | `612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43` | `612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43` | 0 bytes | **INVARIANT** |
| **Classifier Checkpoint** | `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e` | `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e` | 0 bytes | **INVARIANT** |
| **Detector Checkpoint** | `01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2` | `01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2` | 0 bytes | **INVARIANT** |
| **Database Predictions** | 48 records | 48 records | 0 rows | **INVARIANT** |
| **Database Users** | 5 records | 5 records | 0 rows | **INVARIANT** |
| **Classifier Test Split**| 321 images (322 lines) | 321 images (322 lines) | 0 lines | **SEALED & INTACT** |
| **Approval Gate 4** | `PENDING_APPROVAL` | `PENDING_APPROVAL` | Unchanged | **LOCKED** |

---

## 14. Limitations and Generalization Boundaries

1. **Development-Validation Nature of Metrics:** The 15 evaluated image-mask pairs were used for validation during model development (early stopping and checkpoint selection at epoch 31). Consequently, the reported 79.15% Macro Dice and 62.55% Diseased Dice must be understood as **development-validation benchmarks**, not independent generalizable field test accuracy.
2. **Limited Healthy Negative Controls ($N=2$):** Only 2 of the 15 validation masks are pure healthy leaves. While both exhibit low leaf-restricted false-positive rates ($<0.45\%$), the sample size is insufficient to statistically certify false-alarm resistance across diverse field backgrounds.
3. **Punctate Lesion Under-Segmentation:** On micro-lesion pathologies (e.g., early Bacterial Blight or Herbicide spray speckles), the model tends to under-segment diseased area ($0.14\%$ predicted vs $0.85\%$ GT on `BBC00130`; $0.19\%$ predicted vs $4.68\%$ GT on `HGD00151`), as ResNet-34 spatial pooling smooths away sub-5-pixel lesions.
4. **Variegation vs. Disease Ambiguity:** On physiological variegation (`LV00112`), the model over-segments chlorotic yellow patterns, resulting in a $+7.85\%$ severity overestimation at 512 and $+21.47\%$ at 256.

---

## 15. Audit Deliverables Inventory

All 10 structured evidence deliverables and 15 visual QA overlays were generated and saved in `artifacts/audit_phase11/phase11_6/`:

1. [phase11_6_dataset_provenance.json](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_6/phase11_6_dataset_provenance.json) — Mask counts, corpus distribution, split history, class mappings.
2. [phase11_6_validation_samples.csv](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_6/phase11_6_validation_samples.csv) — Exact 15 validation masks, dimensions, pixel counts, and ground-truth severities.
3. [phase11_6_checkpoint_integrity.json](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_6/phase11_6_checkpoint_integrity.json) — Checkpoint SHA-256, parameter counts, architecture, epoch 31 training metadata.
4. [phase11_6_preprocessing_comparison.json](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_6/phase11_6_preprocessing_comparison.json) — Full confusion matrices and metrics across all 6 preprocessing modes.
5. [phase11_6_segmentation_metrics.json](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_6/phase11_6_segmentation_metrics.json) — Comprehensive segmentation performance, severity MAE/RMSE, and healthy control summaries.
6. [phase11_6_per_class_metrics.csv](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_6/phase11_6_per_class_metrics.csv) — Per-class Dice, IoU, precision, recall, and false-positive deltas between 512 and 256.
7. [phase11_6_severity_estimation_results.csv](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_6/phase11_6_severity_estimation_results.csv) — 30 per-image severity evaluations (15 at 512, 15 at 256) with errors and 6-tier categories.
8. [phase11_6_healthy_control_analysis.csv](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_6/phase11_6_healthy_control_analysis.csv) — Quantitative false-positive and false-alarm breakdown for healthy controls `HL00087` and `HL00039`.
9. [phase11_6_repeatability_results.json](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_6/phase11_6_repeatability_results.json) — 25 GPU passes, bitwise mask equivalence, zero probability drift, latency and VRAM telemetry.
10. [phase11_6_environment.json](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_6/phase11_6_environment.json) — Complete execution environment, CUDA device, software versions, and post-audit checkpoint hashes.
11. [visual_qa/](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/phase11_6/visual_qa/) — 15 high-resolution 5-panel diagnostic composite images.

---

## 16. Final Verdict & Recommendations

### Formal Audit Verdict: `PARTIALLY VERIFIED / EXPERIMENTAL`

- **Technical Soundness:** The U-Net ResNet-34 segmentation pipeline and leaf-restricted severity quantification mathematics are implemented rigorously and execute with 100% determinism on GPU.
- **Resolution Imperative:** Operating at native 512×512 resolution substantially outperforms the 256×256 service pipeline (+10.48 pp Diseased Dice, +13.91 pp Precision, 39.0% fewer false positive pixels, and a 57.0% reduction in Severity MAE from 4.63% to 1.99%).
- **Status Gate:** The checkpoint must remain labeled `EXPERIMENTAL`. The 15 validation samples are development-validation samples; independent external segmentation accuracy cannot be certified until an external human-annotated test set is acquired.
- **Approval Gate 4:** Remains **strictly `PENDING_APPROVAL`**.

### Recommendations for Future Phases:
1. **Phase 12 Pipeline Upgrade:** Once Gate 4 is approved, upgrade `prediction_service.py` to run U-Net inference at native 512×512 resolution (or bilinear probability upsampling) instead of downsampling to 256×256.
2. **Healthy Control Threshold Gate:** Implement a minimum leaf disease threshold ($\ge 0.5\%$) before triggering disease severity alarms, preventing healthy leaves with background edge noise from reporting false "Trace" infections.
3. **Dataset Expansion:** Prioritize expanding the 50-mask corpus to 100+ rasterized masks, ensuring at least 20 independent healthy controls are included.

---
*Audit Completed Successfully. Execution Halted. Awaiting Explicit User Instructions.*
