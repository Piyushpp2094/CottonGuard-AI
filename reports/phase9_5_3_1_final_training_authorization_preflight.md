# CottonGuard AI — Phase 9.5.3.1 Technical Report
## Final Training Authorization Preflight

**Date:** 2026-11-17  
**Project:** CottonGuard AI (`cotton_leaf_ai`)  
**Phase:** Phase 9.5.3.1 — Final Training Authorization Preflight  
**Author:** Autonomous Senior AI/ML & Computer Vision Engineer  
**Status / Executive Verdict:** **`A. READY FOR EXPLICIT TRAINING APPROVAL`**  
**Formal Safety Governance:** Strictly non-executing preflight verification. Zero training epochs executed. Zero optimizer updates. Zero checkpoint modifications. Zero database updates. CVAT and sealed test set untouched.

---

## 1. Executive Summary & Verdict

Phase 9.5.3.1 resolves all remaining numerical, dataset, geometric, and hardware-verification questions identified during the Phase 9.5.3 review. This preflight audit establishes the exact execution specification, safety boundaries, hardware feasibility, and rollback protections required for Faster R-CNN retraining with Tier 2 ignore-region supervision.

### Key Preflight Results:
1. **Numerical Reconciliation of All 1,009 Tier 1 Detection Boxes:**
   - Evaluated directly from raw CVAT XML archives (`task4_pilot_all21.zip`, `task5_pilot_batch02.zip`) and JSON payload (`candidate_annotations_task6.json`).
   - The consolidated 100-image corpus contains exactly **1,009 Tier 1 focal detection boxes** across 73 diseased images, **130 Tier 2 diffuse ignore regions** across 23 images, and **15 golden Healthy Leaf negative controls** (strictly 0 boxes, 0 ignore regions).
   - The historical **817-box figure** was conclusively resolved: it represented the published Tasks #4 and #5 baseline (514 boxes) plus an early 30-frame Task #6 pilot subset (303 boxes). The complete 50-frame Task #6 expansion yields 495 explicit detection boxes, reconciling exactly: $514 + 495 = \mathbf{1,009\text{ boxes}}$.
2. **Complete Geometric Outlier Audit of All 130 Tier 2 Ignore Regions:**
   - All 130 diffuse shapes were audited for pixel area, bounding box area, fill factor, and healthy lamina enclosure.
   - Identified and visually confirmed **5 Whole-Leaf Morphological Outliers** (>25% healthy leaf enclosed: `LR00003`, `CV00006`, `CV00102`, `CV00001`, `CV00211`), representing systemic leaf curl virus puckering and lobe-wide anthocyanin blush.
   - Audited **27 Intermediate Enclosure Regions** (0.49% to 24.16% healthy leaf enclosed) and **98 Tightly Bounded Regions** (median 0.02% healthy leaf enclosed).
   - Rendered 9 high-resolution diagnostic visual QA cards confirming proper separation between Cyan focal boxes and Orange ignore boxes.
3. **Warmup & Learning Rate Schedule Recalculation:**
   - For 70 training samples at batch size 2, steps per epoch is $\lceil 70 / 2 \rceil = 35\text{ steps}$.
   - Over a 3-epoch warmup, the schedule requires **105 warmup steps** (reconciled from the old 54 steps calibrated for 35 samples), followed by 945 cosine decay steps across the remaining 27 epochs (total 1,050 steps).
4. **Controlled RTX 3050 GPU (6 GB VRAM) Hardware Dry-Run:**
   - Under `torch.cuda.amp.autocast(float16)` with batch size 2 at native 800×800 resolution:
     - Model Loaded VRAM: **169.37 MB**
     - Peak Forward VRAM: **1,855.98 MB**
     - Peak Backward Allocated VRAM: **1,906.91 MB**
     - Peak Backward Reserved VRAM: **2,210.00 MB**
     - Available Safety Headroom: **4,236.6 MB (69.0% buffer)** on the 6,143.5 MB GPU.
   - Parameter hash invariant verified before and after the backward pass: **SHA-256 identical (`eccdc3ea...`)**, zero parameter drift, zero optimizer steps.
5. **Split Governance (Option A vs. Option B):**
   - Option A (70 train / 30 val) is formally recommended. Reserving an independent test set from 100 images would leave only ~15 images in test (~2 per class), making mAP statistically noisy. Transparent reporting documents these as 30-sample validation metrics.
6. **Isolated Destination & Rollback Protection:**
   - Retraining will write to `artifacts/checkpoints/detection/phase9_5_retrain/`, completely preserving the production baseline `artifacts/checkpoints/detection/best_faster_rcnn.pt` (SHA-256: `01d82b32...`).
7. **Preflight Outcome:** **`A. READY FOR EXPLICIT TRAINING APPROVAL`**.

---

## 2. Exhaustive Reconciliation of All 1,009 Detection Boxes

### 2.1 Traceability from Raw Payloads to Model Targets

The detection target dataset is derived from three historical CVAT annotation tasks across 100 images:

| CVAT Task ID | Batch Name | Image Count | Healthy Controls | Diseased Images | Raw Annotation Source | Emitted Tier 1 Focal Boxes | Emitted Tier 2 Ignore Boxes |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| **Task #4** | Pilot Batch 01 | 21 | 3 | 18 | `task4_pilot_all21.zip` (CVAT XML 1.1) | 208 | 0 |
| **Task #5** | Pilot Batch 02 | 29 | 4 | 25 | `task5_pilot_batch02.zip` (CVAT XML 1.1) | 306 | 0 |
| **Task #6** | Pilot Batch 03 | 50 | 8 | 42 | `candidate_annotations_task6.json` (CVAT JSON 1.1) | 495 | 130 |
| **Total** | **Consolidated Corpus** | **100** | **15** | **85** | **Multi-Source CVAT Corpus** | **1,009** | **130** |

### 2.2 Resolution of the Historical 817-Box Discrepancy

Earlier planning documents referenced an **817-box figure**. The exact derivation and discrepancy have been mathematically reconciled:
- In Phase 9.0/9.1, Tasks #4 and #5 provided:
  $$208\text{ (Task 4)} + 306\text{ (Task 5)} = 514\text{ published boxes}$$
- An interim pilot subset of Task #6 (the first 30 annotated images) contained **303 boxes**:
  $$514 + 303 = \mathbf{817\text{ boxes}}$$
- When Task #6 was fully expanded to 50 images in Phase 9.3/9.5 under the V3 dual-tier standard, the final Task #6 count reached **495 explicit rectangles** (matching 495 focal polygons):
  $$514 + 495 = \mathbf{1,009\text{ total Tier 1 focal boxes}}$$
- There is **zero double-counting** and **zero coordinate truncation**. Every box corresponds to an audited lesion.

### 2.3 Class-Wise Distribution across Tasks

| Disease Class | Class ID | Task #4 Boxes | Task #5 Boxes | Task #6 Focal Boxes | Total Focal Boxes | Task #6 Diffuse Ignore Boxes | Golden Negative Images |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bacterial Blight** | 1 | 48 | 62 | 114 | **224** | 78 | 0 |
| **Curl Virus** | 2 | 16 | 24 | 31 | **71** | 7 | 0 |
| **Healthy Leaf** | 0 | 0 | 0 | 0 | **0** | 0 | **15** |
| **Herbicide Growth Damage**| 3 | 12 | 11 | 8 | **31** | 7 | 0 |
| **Leaf Hopper Jassids** | 4 | 32 | 45 | 57 | **134** | 12 | 0 |
| **Leaf Redding** | 5 | 44 | 78 | 132 | **254** | 26 | 0 |
| **Leaf Variegation** | 6 | 56 | 86 | 153 | **295** | 0 | 0 |
| **Total** | — | **208** | **306** | **495** | **1,009** | **130** | **15** |

### 2.4 Negative Control Integrity
All **15 Healthy Leaf images** (3 in Task #4, 4 in Task #5, 8 in Task #6) were confirmed to have strictly:
- `target["boxes"].shape == torch.Size([0, 4])`
- `target["labels"].shape == torch.Size([0])`
- `target["ignore_boxes"].shape == torch.Size([0, 4])`
- Negative control supervision remains 100% active, ensuring Faster R-CNN learns background suppression on healthy cotton leaves.

---

## 3. Geometric Outlier Audit of All 130 Tier 2 Ignore Regions

### 3.1 Overview of Diffuse Shapes
The 130 Tier 2 diffuse regions represent non-focal, sprawling, or systemic symptom expressions (anthocyanin accumulation, chlorotic vein banding, viral leaf curl epinasty, and water-soaked coalescence).
- **Mean Polygon Area:** $46,291.4\text{ px}^2$
- **Mean Bounding Box Area:** $72,088.1\text{ px}^2$
- **Mean Fill Factor ($\text{Area}_{\text{poly}} / \text{Area}_{\text{box}}$):** $0.593$

### 3.2 Stratified Outlier Categorization

#### Category A: 5 Whole-Leaf Morphological Outliers (>25% Healthy Lamina Enclosed)
These 5 shapes represent severe systemic symptoms where the diffuse polygon spans across major leaf lobes:

| Frame Index | Filename | Disease Class | Poly Area ($\text{px}^2$) | Box Area ($\text{px}^2$) | Fill Factor | Enclosed Healthy Lamina % | Biological Justification |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Frame 43** | `LR00003.jpg` | Leaf Redding | 148,220 | 385,125 | 0.385 | **43.76%** | Entire lower leaf lobe exhibit diffuse anthocyanin reddening. Box spans interveinal gaps. |
| **Frame 37** | `CV00006.jpg` | Curl Virus | 132,450 | 321,400 | 0.412 | **40.54%** | Systemic curl virus causing cupped, distorted lamina. Diffuse shape covers curled margin. |
| **Frame 06** | `CV00102.jpg` | Curl Virus | 118,900 | 295,680 | 0.402 | **34.06%** | Severe viral curling with multiple wrinkled folds enclosing healthy central tissue. |
| **Frame 05** | `CV00001.jpg` | Curl Virus | 112,300 | 278,400 | 0.403 | **33.19%** | Severe downward curl distortion with diffuse marginal chlorosis. |
| **Frame 07** | `CV00211.jpg` | Curl Virus | 98,750 | 246,120 | 0.401 | **26.22%** | Moderate curl virus puckering with diffuse vein banding. |

**Detector Supervision Impact:**
In standard Faster R-CNN, these large areas would be sampled as false negative background, heavily penalizing the detector for firing on viral symptoms. Under Phase 9.5.2 ignore-aware RPN/RoI heads, the ignore mask converts background anchors within these regions to `-1.0` (ignored), preventing false penalties. Concurrently, genuine focal lesions within the frame remain fully protected by the `~is_positive` invariant.

#### Category B: 27 Intermediate Enclosure Regions (0.49% to 24.16% Healthy Lamina Enclosed)
- Represents regional symptom sectors: chlorotic patches in Leaf Hopper Jassids, coalesced Bacterial Blight necrosis, and herbicide epinasty.
- Mean fill factor: $0.612$; median healthy lamina enclosed: $6.82\%$.

#### Category C: 98 Tightly Bounded Regions (<0.49% Healthy Lamina Enclosed)
- 98 out of 130 diffuse shapes (75.4%) are tightly conformed to localized diffuse patches.
- **Median healthy lamina enclosed:** **$0.02\%$** (virtually zero healthy tissue enclosed).
- **Mean fill factor:** $0.708$.

### 3.3 Visual Verification Panels
Diagnostic visual QA cards were generated at $1200 \times 445$ resolution for all 5 whole-leaf outliers and 4 risk categories, showing:
1. Native Lamina + Green Leaf Contour
2. Ground-Truth Annotations (Cyan = Focal, Magenta = Diffuse)
3. Model Supervision Targets (Cyan = Focal Box, Orange = Ignore Box)

- [outlier_overlay_frame_43_LR00003.png](file:///d:/Main_Project/cotton_leaf_ai/artifacts/visual_qa/phase9_5_3_1_preflight/outlier_overlay_frame_43_LR00003.png)
- [outlier_overlay_frame_37_CV00006.png](file:///d:/Main_Project/cotton_leaf_ai/artifacts/visual_qa/phase9_5_3_1_preflight/outlier_overlay_frame_37_CV00006.png)
- [outlier_overlay_frame_06_CV00102.png](file:///d:/Main_Project/cotton_leaf_ai/artifacts/visual_qa/phase9_5_3_1_preflight/outlier_overlay_frame_06_CV00102.png)
- [outlier_overlay_frame_05_CV00001.png](file:///d:/Main_Project/cotton_leaf_ai/artifacts/visual_qa/phase9_5_3_1_preflight/outlier_overlay_frame_05_CV00001.png)
- [outlier_overlay_frame_07_CV00211.png](file:///d:/Main_Project/cotton_leaf_ai/artifacts/visual_qa/phase9_5_3_1_preflight/outlier_overlay_frame_07_CV00211.png)
- [outlier_overlay_frame_40_HGD00280.png](file:///d:/Main_Project/cotton_leaf_ai/artifacts/visual_qa/phase9_5_3_1_preflight/outlier_overlay_frame_40_HGD00280.png)
- [outlier_overlay_frame_22_LR00286.png](file:///d:/Main_Project/cotton_leaf_ai/artifacts/visual_qa/phase9_5_3_1_preflight/outlier_overlay_frame_22_LR00286.png)
- [outlier_overlay_frame_16_LHJ00059.png](file:///d:/Main_Project/cotton_leaf_ai/artifacts/visual_qa/phase9_5_3_1_preflight/outlier_overlay_frame_16_LHJ00059.png)
- [outlier_overlay_frame_35_BBC00008.png](file:///d:/Main_Project/cotton_leaf_ai/artifacts/visual_qa/phase9_5_3_1_preflight/outlier_overlay_frame_35_BBC00008.png)

---

## 4. Warmup Schedule & Learning Rate Mathematics

### 4.1 Resolution of Warmup Steps Discrepancy
The previous configuration in `configs/phase9_2a_detector_config.yaml` specified:
- `warmup_steps: 54`
- `total_steps: 540`

This was mathematically derived for a **35-sample training set**:
$$\text{steps\_per\_epoch} = \lceil 35 / 2 \rceil = 18\text{ steps}$$
$$\text{warmup\_steps} = 18 \times 3\text{ epochs} = 54\text{ steps}$$
$$\text{total\_steps} = 18 \times 30\text{ epochs} = 540\text{ steps}$$

### 4.2 Updated Schedule for Consolidated 70-Sample Split
The consolidated dataset split consists of **70 training samples** and **30 validation samples**. With batch size 2:
$$\text{steps\_per\_epoch} = \lceil 70 / 2 \rceil = 35\text{ steps/epoch}$$
$$\text{warmup\_steps} = 35 \times 3\text{ epochs} = \mathbf{105\text{ steps}}$$
$$\text{total\_steps} = 35 \times 30\text{ epochs} = \mathbf{1,050\text{ steps}}$$

### 4.3 Exact LR Progression
1. **Warmup Phase (Steps 1 to 105):**
   - Linear warmup starting from $\eta_{\text{warmup\_start}} = \frac{\eta_{\text{base}}}{3} = \frac{1.0 \times 10^{-4}}{3} \approx 3.33 \times 10^{-5}$ up to $\eta_{\text{base}} = 1.0 \times 10^{-4}$.
   $$\eta(t) = \eta_{\text{warmup\_start}} + \frac{t}{105} \cdot (\eta_{\text{base}} - \eta_{\text{warmup\_start}})$$
2. **Cosine Decay Phase (Steps 106 to 1,050):**
   - Cosine annealing schedule decaying from $1.0 \times 10^{-4}$ down to $\eta_{\text{min}} = 1.0 \times 10^{-6}$.
3. **Early Stopping:**
   - Evaluated at the end of each epoch (every 35 steps).
   - Patience = 5 epochs (175 steps) without validation mAP@0.50 improvement.

---

## 5. NVIDIA RTX 3050 (6 GB VRAM) Hardware & Parameter Invariant Verification

A rigorous, non-executing memory and parameter invariance dry-run was executed using `scripts/dry_run_gpu_memory.py` on the host hardware.

### 5.1 Hardware Configuration
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (Ampere architecture, GA107)
- **Total Dedicated VRAM:** 6,143.5 MB (6.00 GB)
- **Host Power Limit:** 50 W
- **CUDA Version:** 13.2 / PyTorch 2.15.0+cu132

### 5.2 Dry-Run Execution Metrics

```
================================================================================
COTTONGUARD AI — CONTROLLED GPU MEMORY & PARAMETER INVARIANT DRY-RUN
================================================================================
Device: NVIDIA GeForce RTX 3050 6GB Laptop GPU
Total Dedicated VRAM: 6,143.5 MB (6.00 GB)
Resolution: 800x800 | Batch Size: 2 | Precision: AMP float16

Baseline VRAM: Allocated = 0.00 MB, Reserved = 0.00 MB
Model Loaded VRAM: Allocated = 169.37 MB, Reserved = 170.00 MB
Peak Forward Allocated VRAM: 1,855.98 MB
Peak Backward Allocated VRAM: 1,906.91 MB
Peak Backward Reserved VRAM: 2,210.00 MB
Available Safety Headroom: 4,236.6 MB (69.0% buffer)

Parameters with Gradients Populated: 209/209
Optimizer Step Called: False (STRICTLY OMITTED)
Zero Parameter Drift: True

Initial Parameter SHA-256: eccdc3eaec67d4a5b5fb8167912f20b39de91e786d8abe70093de1e148ed6ee9
Post-Run Parameter SHA-256: eccdc3eaec67d4a5b5fb8167912f20b39de91e786d8abe70093de1e148ed6ee9
Parameter Hash Invariant Preserved: True (100% IDENTICAL)
================================================================================
```

### 5.3 Hardware Feasibility Verdict
- Peak VRAM consumption reached **1,906.91 MB allocated** and **2,210.00 MB reserved**, leaving **4,236.6 MB (69.0%)** of unallocated VRAM.
- Out-of-memory (OOM) risk on the 6 GB RTX 3050 at batch size 2 with 800×800 inputs is **virtually zero**.
- Parameter hash invariance confirmed that executing forward/backward passes for dry-run verification caused **zero weight updates** before formal authorization.

---

## 6. Detection Evaluation Split Decision (Option A vs. Option B)

### 6.1 Evaluation Options

#### Option A: 70 Train / 30 Validation Split (Recommended)
- **Structure:** 70 images for model training, 30 images held out strictly for validation and early stopping.
- **Reporting Standard:** Final detection metrics are transparently published as **30-image validation metrics**.
- **Academic Justification:** In deep learning detection pilots with limited annotated images ($N=100$), splitting into three sets (e.g. 70/15/15) creates an evaluation set of only 15 images (an average of only ~2 images per disease class). A 15-image test set introduces severe statistical variance, where single-lesion false positives cause massive swing in class mAP. Retaining 30 validation images provides a statistically stable sample (~4 to 5 images per class) for measuring precision, recall, and mAP@0.50.
- **Transparency Safeguard:** The technical report and dissertation will explicitly state that detection performance reflects a stratified 30-image validation split, while the 321-image classifier test set remains independently sealed.

#### Option B: 70 Train / 15 Validation / 15 Test Split (Rejected)
- **Structure:** 70 train, 15 validation for early stopping, 15 sealed test for final evaluation.
- **Limitation:** At 15 test images, rare classes like Herbicide Growth Damage or Curl Virus have only 1 to 2 images. If a single diffuse leaf curl lesion has IoU 0.49 instead of 0.50, class AP swings by 50–100%. This degrades academic reproducibility.

### 6.2 Formal Decision
**Option A is adopted.** All detection evaluations will be reported on the audited 30-image validation split, preserving statistical reliability and complete academic honesty.

---

## 7. Final Experiment Specification Run Card

Upon separate user approval, Faster R-CNN retraining will execute strictly under the following isolated run card:

```yaml
# CottonGuard AI — Experiment Run Card
experiment_name: "phase9_5_faster_rcnn_ignore_retrain"
run_id: "EXP-FRCNN-20261117-01"
timestamp: "2026-11-17"

model:
  architecture: "FasterRCNN_ResNet50_FPN_V2"
  pretrained_backbone: true
  trainable_backbone_layers: 3
  num_classes: 7  # 6 disease classes + background (0: healthy/background)
  enable_ignore_regions: true
  rpn_ignore_threshold: 0.30
  roi_ignore_threshold: 0.30

dataset:
  manifest_path: "artifacts/manifests/consolidated_100_annotation_manifest.csv"
  total_images: 100
  train_split: 70
  val_split: 30
  resolution: [800, 800]
  batch_size: 2
  num_workers: 0  # Safe Windows multiprocessing

optimization:
  optimizer: "AdamW"
  base_lr: 0.0001
  weight_decay: 0.0001
  warmup_steps: 105      # 3 epochs * 35 steps
  total_steps: 1050      # 30 epochs * 35 steps
  scheduler: "CosineAnnealingLR"
  min_lr: 0.000001
  mixed_precision: "amp_float16"
  max_epochs: 30
  early_stopping_patience: 5  # epochs without val mAP improvement

paths_and_isolation:
  output_checkpoint_dir: "artifacts/checkpoints/detection/phase9_5_retrain/"
  best_checkpoint_path: "artifacts/checkpoints/detection/phase9_5_retrain/best_faster_rcnn_ignore.pt"
  latest_checkpoint_path: "artifacts/checkpoints/detection/phase9_5_retrain/latest_faster_rcnn_ignore.pt"
  training_log_path: "artifacts/checkpoints/detection/phase9_5_retrain/training_metrics.json"

preservation_and_rollback:
  baseline_checkpoint_protected: "artifacts/checkpoints/detection/best_faster_rcnn.pt"
  baseline_sha256: "01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2"
  rollback_trigger: "If retrained model val mAP@0.50 regresses or fails stability checks, baseline checkpoint remains production default."
```

---

## 8. Final Safety Audit & Checkpoint Provenance

| Component | Path | Current SHA-256 Hash | Status |
| :--- | :--- | :--- | :---: |
| **Classification Checkpoint** | `artifacts/checkpoints/classification/best_model.pt` | `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e` | **Preserved / Verified** |
| **Baseline Detection Checkpoint** | `artifacts/checkpoints/detection/best_faster_rcnn.pt` | `01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2` | **Preserved / Verified** |
| **Segmentation Checkpoint** | `artifacts/checkpoints/segmentation/best_unet.pt` | `612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43` | **Preserved / Verified** |
| **Sealed Classification Test Set** | `data/splits/test.csv` | `6025d86f89b9ec4d3f95ad4a54f7d633478c9eb152c67057dfb021d8c00bdd9d` | **0 Leakage / Sealed** |
| **Consolidated Manifest** | `artifacts/manifests/consolidated_100_annotation_manifest.csv` | `bf404ecdc579fc9fc558bbf628623594b29f7f0df81e053a4805eb3ba3ea0ae6` | **Audited / Ready** |

---

## 9. Conclusion & Preflight Verdict

### Formal Outcome Recommendation:
# **`A. READY FOR EXPLICIT TRAINING APPROVAL`**

All technical prerequisites, coordinate reconciliations, geometric risk audits, warmup mathematics, hardware feasibility tests, parameter hash invariants, and rollback protections have been verified.

**Next Step:** Awaiting explicit user approval to execute Faster R-CNN training under Run Card `EXP-FRCNN-20261117-01`.
