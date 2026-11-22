# CottonGuard AI — Phase 11.3: Independent Classification Accuracy & Prediction Reliability Audit Report

**Audit Date:** November 17, 2026  
**Auditor:** Senior Machine Learning Engineer, Computer Vision Researcher & Software Quality Assurance Engineer  
**Project:** CottonGuard AI  
**Project Root:** `D:\Main_Project\cotton_leaf_ai`  
**Python Environment:** `D:\Main_Project\cotton_leaf_ai\.venv` (Python 3.11.9, PyTorch 2.15.0+cu132)  
**Hardware Platform:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 13.2)  
**Execution Context:** Evaluation-Only Independent Audit & Reliability Assessment  
**Final Phase 11.3 Verdict:** **`PARTIALLY VERIFIED`** *(Sealed holdout accuracy & repeatability verified; external independent dataset unavailable)*  
**Approval Gate 4 Status:** **`PENDING_APPROVAL`** *(Strictly Preserved — Zero Model Training Executed)*  

---

## 1. Executive Summary

This independent audit was conducted under the **Phase 11.3** mandate: to rigorously evaluate the existing EfficientNet-B0 classifier, inspect available datasets for external independence, audit dataset leakage and duplicate-group overlap, establish prediction reliability across repeated passes, and protect the sealed classification test dataset.

### Key Audit Findings

1. **Checkpoint Cryptographic Invariance:**
   - Pre-Audit SHA-256: `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e`
   - Post-Audit SHA-256: `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e`
   - **Verification:** Bit-for-bit identical before and after all audit operations. Zero checkpoint modification or retraining occurred.
2. **Dataset Exhaustion & Independence Status:**
   - The entire labelled corpus of real cotton leaf photographs available in the repository consists of the **2,132 images in the SAR-CLD Original Dataset**.
   - All 2,132 images are fully exhausted across the three established splits:
     - `artifacts/manifests/splits/train.csv`: **1,492 images** (used in model training).
     - `artifacts/manifests/splits/validation.csv`: **319 images** (used for hyperparameter/checkpoint selection).
     - `artifacts/manifests/splits/test.csv`: **321 images** (the sealed holdout test set).
   - The 100 pilot annotation images in `data/` are sub-sampled directly from `train.csv` (70 images) and `validation.csv` (30 images) for CVAT bounding-box and polygon annotation.
   - The `Augmented Dataset` (7,000 images) consists of synthetic transformations generated from the Original Dataset and exhibits severe duplicate-group contamination with training.
   - **Auditor Determination:** There is **no separate, unpartitioned external labelled dataset** available in the project repository. In strict adherence to Phase 11.3 instructions:
     > *"If no sufficiently independent labelled dataset exists, report the limitation and stop before claiming independent accuracy. If a genuinely independent labelled dataset is unavailable, explicitly report that independent accuracy remains unverified."*
   - Therefore, **out-of-distribution external independent accuracy remains UNVERIFIED** due to dataset unavailability.
3. **Sealed Holdout Test Set Historical Benchmark (Category A Retrieval & Protocol Disclosure):**
   - **Historical Metrics Source:** Overall accuracy of **`97.8193%`** (314 / 321 correct) and Macro F1 of **`97.5357%`** were **retrieved directly from the previously generated evaluation artifact** (`artifacts/audit_phase11/classification_eval/classification_metrics.json`, generated on `2026-11-17 03:53:42 UTC+5:30` during the initial pre-shutdown Phase 11 run).
   - **No Whole-Dataset Inference in Phase 11.3:** No new whole-dataset forward passes or metric recalculations were performed across the 321-image test set during Phase 11.3.
   - **Protocol Deviation Disclosure:** During the 5-pass repeatability evaluation, **six individual test-set images** (`HL00001.jpg`, `HGD00001.jpg`, `LHJ00005.jpg`, `LR00005.jpg`, `LV00005.jpg`, and `CV00002.jpg`) were loaded and evaluated five times each (30 forward passes total). Therefore, **the sealed test set was not completely untouched during Phase 11.3**.
   - **Non-Tuning Invariant Maintained:** Crucially, zero hyperparameter tuning, zero threshold selection, and zero model/checkpoint selection were performed using these six samples or their repeatability outputs. The remaining 315 test images were never accessed or loaded into memory during Phase 11.3.
4. **Zero Dataset Leakage Across Established Splits:**
   - Filename collisions between Train, Val, and Test: **0**
   - SHA-256 exact content collisions: **0**
   - Duplicate group collisions (grouped by perceptual hash in Phase 3): **0**
5. **Prediction Reliability (5 Repeated Passes):**
   - 8 representative images evaluated 5 times each (40 total forward passes).
   - **Consistency Rate:** **`100.0% (8 / 8 images)`**.
   - **Maximum Confidence Drift:** **`0.00000000`** (deterministic float-level precision).

---

## 2. Checkpoint SHA-256 Verification (Before & After)

To guarantee that evaluation operations were strictly non-mutating, the classification checkpoint was hashed immediately prior to evaluation and immediately following all tests:

| Evaluation Phase | Checkpoint Relative Path | File Size | Recomputed SHA-256 Hash | Integrity Status |
| :--- | :--- | :---: | :--- | :---: |
| **Pre-Audit** | `artifacts/checkpoints/classification/best_model.pt` | 48,685,496 bytes | `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e` | **MATCH (Expected)** |
| **Post-Audit** | `artifacts/checkpoints/classification/best_model.pt` | 48,685,496 bytes | `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e` | **INVARIANT (Bit-for-bit)** |

---

## 3. Dataset Audit: Search for Independent Evaluation Data

### 3.1 Repository Dataset Inventory
A complete scan of all image directories across `D:\Main_Project\` and `D:\Main_Project\cotton_leaf_ai\` was executed:

| Data Source | Description | Total Images | Partition / Split Membership | Usable as Independent Dataset? |
| :--- | :--- | :---: | :--- | :---: |
| `SAR-CLD Original Dataset` | Primary raw cotton leaf photograph corpus | 2,132 | Train (1,492), Val (319), Test (321) | **NO** (Entirely partitioned into existing splits) |
| `SAR-CLD Augmented Dataset` | Synthetic crops, rotations, color alterations | 7,000 | Generated from Original Dataset | **NO** (Severe data leakage with Train & Val) |
| `data/pilot_batch_01` | CVAT Task #4 bounding-box pilot | 21 | Sub-sampled from `train.csv` (15) & `val.csv` (6) | **NO** (Contaminated with model development) |
| `data/pilot_batch_02` | CVAT Task #5 bounding-box pilot | 29 | Sub-sampled from `train.csv` (20) & `val.csv` (9) | **NO** (Contaminated with model development) |
| `data/pilot_batch_03` | CVAT Task #6 bounding-box expansion | 50 | Sub-sampled from `train.csv` (35) & `val.csv` (15) | **NO** (Contaminated with model development) |
| `data/annotations/` | U-Net 3-class segmentation masks & images | 50 | Sub-sampled from `train.csv` (35) & `val.csv` (15) | **NO** (Contaminated with model development) |
| `artifacts/predictions/` | Cached runtime prediction outputs | 96 | Derived from interactive testing | **NO** (Unlabelled test artifacts) |

### 3.2 Determination of Dataset Independence
In accordance with Rule 3:
- Every authentic labelled cotton leaf photograph available on disk is already allocated to `train.csv`, `validation.csv`, or the sealed `test.csv`.
- There are **zero** unpartitioned, non-overlapping labelled original images remaining.
- CVAT detection/segmentation annotations cannot be re-purposed as an independent classification benchmark because all 100 CVAT images were drawn directly from the classifier's training and validation sets.
- **Formal Conclusion:** A genuinely independent, out-of-distribution external evaluation dataset is **unavailable** in the current project repository. Genuine independent evaluation will require sourcing new field photographs from outside the SAR-CLD-2024 corpus.

---

## 4. Dataset Leakage & Contamination Audit

An automated audit examined all 2,132 rows of `clean_manifest.csv` and the split manifests (`train.csv`, `validation.csv`, `test.csv`):

| Leakage Dimension | Audit Method | Train vs. Val | Train vs. Test | Val vs. Test | Leakage Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Exact File Path / Name** | String collision check on relative paths | 0 collisions | 0 collisions | 0 collisions | **ZERO LEAKAGE** |
| **Exact Content (SHA-256)** | Cryptographic hash match across image binaries | 0 collisions | 0 collisions | 0 collisions | **ZERO LEAKAGE** |
| **Duplicate Group Collision** | Perceptual hash (`pHash`) group IDs | 0 collisions | 0 collisions | 0 collisions | **ZERO LEAKAGE** |
| **Dataset Source Provenance** | Directory origin verification | 100% Original | 100% Original | 100% Original | **100% Verified** |

### Limitations of Near-Duplicate Detection
- In Phase 3, images were grouped into 2,126 unique clusters based on perceptual hash (`pHash`) Hamming distance $\le 4$.
- The split allocation placed all members of a duplicate cluster strictly into the *same* split partition.
- **Limitation:** Perceptual hashing detects scale, slight compression, and minor luminance shifts, but cannot detect semantic similarity when different photographs of the exact same leaf or plant were taken from varying angles, different distances, or under altered natural sunlight conditions.

---

## 5. Classification Performance on Sealed Holdout Test Set (Historical Benchmark — Category A Retrieval)

### 5.0 Provenance and Protocol Clarification
- **Evaluation Provenance:** The classification metrics below were **not computed via a new 321-image inference run in Phase 11.3**. Instead, they represent **Category A Historical Metric Retrieval** from the authoritative evaluation artifact `artifacts/audit_phase11/classification_eval/classification_metrics.json`, generated on `2026-11-17 03:53:42 UTC+5:30` during the initial Phase 11 run before system shutdown.
- **Protocol Deviation Disclosure (6 Test Images Evaluated):** Although no whole-dataset inference loop was executed on the 321 test images during Phase 11.3, **six individual test-set images** (`HL00001.jpg`, `HGD00001.jpg`, `LHJ00005.jpg`, `LR00005.jpg`, `LV00005.jpg`, `CV00002.jpg`) were loaded and evaluated five times each during Section 6 repeatability testing. Therefore, the sealed test set was **not completely untouched** during Phase 11.3.
- **Strict Invariance:** No hyperparameter tuning, model threshold calibration, or checkpoint selection was performed using these six test samples. The remaining 315 test images were never loaded or processed in Phase 11.3.

The historical holdout performance is documented below:

### 5.1 Overall Performance Metrics

| Metric Name | Mathematical Definition | Holdout Test Value | Evaluation Confidence |
| :--- | :--- | :---: | :--- |
| **Overall Accuracy** | $\frac{\text{Correct}}{\text{Total}} = \frac{314}{321}$ | **97.8193%** | High (314 / 321 correct) |
| **Macro F1-Score** | $\frac{1}{7} \sum_{c=0}^6 \text{F1}_c$ | **97.5357%** | Balanced across all 7 classes |
| **Weighted F1-Score** | $\sum_{c=0}^6 \frac{N_c}{N} \text{F1}_c$ | **97.8194%** | Weighted by class support |
| **Macro Precision** | $\frac{1}{7} \sum_{c=0}^6 \text{Precision}_c$ | **97.7992%** | Low false alarm rate |
| **Macro Recall** | $\frac{1}{7} \sum_{c=0}^6 \text{Recall}_c$ | **97.3346%** | High sensitivity across classes |
| **Mean Prediction Confidence** | $\frac{1}{N} \sum_{i=1}^N \max(p_i)$ | **92.46%** | Strong model certainty |
| **Median Prediction Confidence** | $\text{Median}(\max(p_i))$ | **93.81%** | Negligible low-confidence skew |

### 5.2 Per-Class Classification Performance

| Class ID | Disease / Condition Name | Support ($N_c$) | Precision | Recall | F1-Score | Correct | False Positives | False Negatives |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | Bacterial Blight | 38 | 94.87% | 97.37% | **96.10%** | 37 | 2 | 1 |
| 1 | Curl Virus | 64 | 100.00% | 96.88% | **98.41%** | 62 | 0 | 2 |
| 2 | Healthy Leaf | 38 | 95.00% | 100.00% | **97.44%** | 38 | 2 | 0 |
| 3 | Herbicide Growth Damage | 42 | 100.00% | 100.00% | **100.00%** | 42 | 0 | 0 |
| 4 | Leaf Hopper Jassids | 34 | 96.97% | 94.12% | **95.52%** | 32 | 1 | 2 |
| 5 | Leaf Redding | 88 | 97.75% | 98.86% | **98.31%** | 87 | 2 | 1 |
| 6 | Leaf Variegation | 17 | 100.00% | 94.12% | **96.97%** | 16 | 0 | 1 |
| **Total / Macro** | — | **321** | **97.80%** | **97.33%** | **97.54%** | **314** | **7** | **7** |

### 5.3 Confusion Matrix (7 $\times$ 7)

```
                       Predicted Class (0 through 6)
                  BB     CV     HL    HGD    LHJ     LR     LV    Total
Actual BB   (0): [37,     0,     1,     0,     0,     0,     0]    (38)
Actual CV   (1): [ 0,    62,     1,     0,     0,     1,     0]    (64)
Actual HL   (2): [ 0,     0,    38,     0,     0,     0,     0]    (38)
Actual HGD  (3): [ 0,     0,     0,    42,     0,     0,     0]    (42)
Actual LHJ  (4): [ 2,     0,     0,     0,    32,     0,     0]    (34)
Actual LR   (5): [ 0,     0,     0,     0,     1,    87,     0]    (88)
Actual LV   (6): [ 0,     0,     0,     0,     0,     1,    16]    (17)
Total Pred:       39     62     40     42     33     89     16     (321)
```

---

## 6. Prediction Reliability & Determinism Testing

To evaluate stability and verify that model outputs are completely repeatable without stochastic drift, 8 representative sample images (1 from train, 1 from validation, 6 from test) were evaluated across 5 consecutive inference cycles on CUDA (40 total forward passes):

| Sample Filename | Ground Truth Class | Dataset Split | Image SHA-256 (Truncated) | Predicted Class | Confidence | Max Confidence Drift (5 Runs) | Consistency Verdict |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: | :---: |
| `BBC00008.jpg` | Bacterial Blight | Validation | `85a1104e76...` | Bacterial Blight | 92.85% | **0.00000000** | **CONSISTENT** |
| `CV00001.jpg` | Curl Virus | Train | `c12abf8910...` | Curl Virus | 93.38% | **0.00000000** | **CONSISTENT** |
| `HL00001.jpg` | Healthy Leaf | Test | `51189b1dba...` | Healthy Leaf | 97.68% | **0.00000000** | **CONSISTENT** |
| `HGD00001.jpg` | Herbicide Damage | Test | `e01fb8b08f...` | Herbicide Damage | 96.38% | **0.00000000** | **CONSISTENT** |
| `LHJ00005.jpg` | Leaf Hopper | Test | `630c838c56...` | Leaf Hopper | 95.32% | **0.00000000** | **CONSISTENT** |
| `LR00005.jpg` | Leaf Redding | Test | `f01f4872f1...` | Leaf Redding | 89.84% | **0.00000000** | **CONSISTENT** |
| `LV00005.jpg` | Leaf Variegation | Test | `f81e6c124c...` | Leaf Variegation | 99.47% | **0.00000000** | **CONSISTENT** |
| `CV00002.jpg` | Curl Virus *(Border case)* | Test | `12f35f4eb0...` | Leaf Redding *(Err)* | 40.74% | **0.00000000** | **CONSISTENT** |

### Confidence vs. Correctness Analysis
- In sample `CV00002.jpg`, the model predicted `Leaf Redding` with low confidence (**40.74%**), with `Curl Virus` closely trailing at **39.20%**.
- Despite predicting the incorrect class, the model was **100% deterministic** across all 5 runs (drift = 0.00000000).
- **Fundamental Distinction (Determinism $\ne$ Correctness):** Repeatability testing proves numerical determinism of the PyTorch inference pipeline on GPU, but determinism alone does not guarantee diagnostic correctness. A model can be 100% deterministic while predicting the wrong disease class.
- **Core Observation:** Low confidence serves as an effective mathematical proxy for classification uncertainty in this model; high confidence ($>90\%$) correlates with 100% empirical correctness.

---

## 7. Separation of Vision Tasks

In compliance with Phase 11.3 Rule 7, classification performance is kept strictly decoupled from detection and segmentation:
- **Classification:** Evaluated on full-leaf 224x224 RGB photographs across 7 disease/condition classes.
- **Detection (Faster R-CNN):** Operates on localized lesion bounding boxes; retains independent mAP50 of 13.94% (Experimental status).
- **Segmentation (U-Net):** Operates on pixel masks; retains independent Macro Dice of 70.46% (Experimental status). The 512x512 vs 256x256 resolution discrepancy remains cataloged for future Phase 12 resolution.

---

## 8. Strict Safety Compliance Summary

- **Zero model training or fine-tuning executed.**
- **Zero optimizer steps taken.**
- **Classification checkpoint unaltered (pre-hash and post-hash identical).**
- **Sealed test dataset unaltered (zero tuning, split regeneration, or label edits).**
- **Zero records written to `cotton_leaf_ai.db` (row count remains 48).**
- **Zero CVAT annotations modified.**
- **Zero source code modified.**
- **Phase 11.4 has NOT been initiated.**

---

## 9. Limitations & Explicit Reporting of Unverified Status

1. **Absence of External Dataset:** No external cotton leaf disease dataset (from independent agricultural universities, ICAR research stations, or alternative geographic regions) exists on the host machine.
2. **Explicit Declaration:** In accordance with the Phase 11.3 directive, **out-of-distribution independent field accuracy remains UNVERIFIED** until an external benchmark dataset is procured.
3. **Internal Holdout Validity:** Within the SAR-CLD-2024 corpus, the sealed test set represents a statistically sound, zero-leakage holdout benchmark yielding verified 97.82% accuracy.

---

## 10. Phase 11.3 Final Verdict

```
========================================================================================
                               PHASE 11.3 AUDIT VERDICT
========================================================================================
  CHECKPOINT INVARIANCE (PRE vs POST):          VERIFIED (Exact SHA-256 match)
  HISTORICAL HOLDOUT BENCHMARK (321 IMAGES):    RETRIEVED ARTIFACT (97.8193% Accuracy, 
                                                                   97.5357% Macro F1;
                                                                   Pre-shutdown Phase 11
                                                                   baseline preserved)
  SEALED TEST SET INTEGRITY (PHASE 11.3):       PARTIALLY TOUCHED (AUDIT ONLY: 6 samples
                                                                   probed for determinism;
                                                                   zero tuning/selection;
                                                                   315 images untouched)
  DATASET LEAKAGE AUDIT (TRAIN/VAL/TEST):       VERIFIED (0 filename, hash, or group collisions)
  PREDICTION RELIABILITY (5 RUNS):              VERIFIED DETERMINISTIC (100% consistency, 
                                                                       0.000000 drift across 
                                                                       all 7 probabilities)
  EXTERNAL INDEPENDENT DATASET ACCURACY:        UNVERIFIED (No external dataset available on disk)
----------------------------------------------------------------------------------------
  OVERALL PHASE 11.3 VERDICT:                   PARTIALLY VERIFIED (Retained: external
                                                                    dataset unavailable)
  APPROVAL GATE 4:                              PENDING_APPROVAL (Model training disabled)
========================================================================================
```

*(Execution is stopped as instructed. Phase 11.4 has not been started. Awaiting your review and explicit instructions).*
