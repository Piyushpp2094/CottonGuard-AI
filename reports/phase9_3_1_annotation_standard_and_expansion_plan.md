# CottonGuard AI — Phase 9.3.1: Annotation Standardization and Dataset Expansion Plan
## Strategic Engineering Roadmap for Scaling the Faster R-CNN Detection Corpus from 50 to 100+ Verified Images

**Project Root:** `D:\Main_Project\cotton_leaf_ai`  
**Phase:** 9.3.1 — Annotation Standardization and Dataset Expansion Plan  
**Date:** 2026-11-15  
**Execution Authorization:** Granted explicitly by User for Phase 9.3.1 (Planning and Read-Only Audit Only)  
**Target Detector:** TorchVision Faster R-CNN ResNet-50 FPN V2  
**Baseline Model:** Checkpoint `best_faster_rcnn.pt` (Epoch 13, SHA-256 `01D82B32...`, $\text{val\_mAP}_{50} = 13.94\%$)  
**Current Detector Status:** `EXPERIMENTAL / NOT APPROVED FOR PRODUCTION`

---

## 1. Executive Summary & Objective

Phase 9.3.1 establishes a comprehensive, scientifically rigorous engineering roadmap to scale the CottonGuard AI object detection dataset from the existing **50-image pilot corpus** (35 train / 15 val, 514 bounding boxes) to a standardized **100-image balanced corpus** (70 train / 30 val, target $\approx 1,000\text{--}1,200$ bounding boxes).

This plan directly incorporates empirical lessons from:
1. **Phase 9.2C Full Training:** Observed training loss convergence ($4.52 \to 0.28$) with validation plateauing at Epoch 13 ($\text{mAP}_{50} = 13.94\%$), proving that further model generalization is fundamentally constrained by pilot sample size (35 training images).
2. **Phase 9.2C.1 Confidence Sweep:** Proved that true positive lesion detections carry high confidence, and operating thresholds of $T = 0.25$ (screening) and $T = 0.40$ (diagnostic UI rendering, peak $F1 = 0.2909$) effectively eliminate over $80\%$ of false alarms.
3. **Phase 9.2C.2 Error Analysis:** Identified the **Annotation Granularity Discrepancy** (where human annotators in Tasks #4 and #5 grouped proximate lesions into loose composite boxes, whereas the Faster R-CNN detector naturally predicts discrete individual spots, artificially inflating false alarms by $15\text{--}20\%$). Furthermore, micro-lesions ($< 900\text{ px}^2$) were revealed to account for $47.6\%$ of all missed detections.

### Core Strategic Decisions
- **Single-Lesion Bounding Box Standard:** Mandatory cessation of composite grouping boxes; every distinguishable focal lesion must receive an individual tight bounding box.
- **Tripartite Multi-Modal Division of Labor:** Confining Faster R-CNN detection to focal necrotic/chlorotic spot diseases (Bacterial Blight, Leaf Hopper Jassids), while delegating whole-leaf architectural morphopathies (Curl Virus, diffuse Herbicide strapping) to EfficientNet-B0 classification and U-Net semantic segmentation.
- **Expansion Vehicle:** Executing annotations across the 47 untouched frames of staged **CVAT Task #6** (`CottonGuard_Pilot_Batch_03_ShapesOnly`), which is already balanced, stratified, and verified leak-free.

---

## 2. Current Dataset Status and CVAT Task Inventory

A complete read-only audit of all existing CVAT tasks in Project #3 (`CottonGuard_AI_Annotation_v2`) was conducted:

| CVAT Task ID & Name | Job ID | Total Frames | Annotated Frames | Untouched Frames | Publication Status | Description & Role |
|---|---:|---:|---:|---:|---|---|
| **Task #3** (`CottonGuard_Pilot_Batch_01_21imgs`) | 3 | 21 | 21 (Corrupted) | 0 | **EXCLUDED** | Historical initial pilot; contains 12 `<track>` structures; permanently quarantined. |
| **Task #4** (`CottonGuard_Pilot_Batch_01_ShapesOnly`) | 4 | 21 | 21 | 0 | **PUBLISHED (Batch 01)** | Clean shapes-only replacement; yields 208 YOLO bounding boxes across 6 classes. |
| **Task #5** (`CottonGuard_Pilot_Batch_02_ShapesOnly`) | 5 | 29 | 29 | 0 | **PUBLISHED (Batch 02)** | Expansion batch; yields 306 YOLO bounding boxes across 6 classes. |
| **Task #6** (`CottonGuard_Pilot_Batch_03_ShapesOnly`) | 6 | 50 | 3 (Verified) | 47 | **STAGED / UNPUBLISHED** | Provisioned expansion task; 3 pilot frames verified (F0, F16, F30); 47 frames untouched. |

### Published Baseline vs Staged Corpus

```
Current Published Corpus (Tasks #4 + #5):
  Total Images: 50 images (35 Train / 15 Validation)
  Total Bounding Boxes: 514 boxes (371 Train / 143 Validation)
  Class Breakdown: Bacterial Blight (188), Leaf Variegation (121), Curl Virus (59), 
                   Leaf Hopper Jassids (50), Leaf Redding (69), Herbicide Damage (27).
  Healthy Controls: 7 images (0 boxes).

Staged Task #6 Corpus:
  Total Images: 50 images (35 Train / 15 Validation)
  Annotated & Verified: 3 images (Frame 0: BBC00001, Frame 16: LHJ00059, Frame 30: HL00002).
  Untouched & Ready for Annotation: 47 images (32 Train / 15 Validation).
  Class Breakdown: Perfectly balanced at 7 images per disease class (7x6=42) + 8 Healthy Controls.
```

When Task #6 is fully annotated and merged with Tasks #4 and #5, the expanded corpus will contain **exactly 100 images** (70 training, 30 validation), preserving the canonical $70\% / 30\%$ dataset split ratio with zero duplicate images.

---

## 3. Single-Lesion Annotation Standard

To resolve the annotation granularity discrepancy identified in Phase 9.2C.2, all future annotations in Task #6 and subsequent tasks must strictly adhere to the **Single-Lesion Annotation Protocol**:

### Standard Rules (Rules 1–10)

```
        INCORRECT (Legacy Composite Box)             CORRECT (Single-Lesion Standard)
       +-------------------------------+           +---------+     +---------+
       |   (o)                   (o)   |           |   (o)   |     |   (o)   |
       |         Healthy Tissue        |    ==>    +---------+     +---------+
       |   (o)                   (o)   |           +---------+     +---------+
       +-------------------------------+           |   (o)   |     |   (o)   |
                                                   +---------+     +---------+
   Penalizes discrete model predictions as FPs.      Aligns ground truth with detector heads.
```

1. **One Bounding Box per Distinguishable Lesion:**
   - Every individual lesion with a distinct necrotic core, puncture point, or chlorotic margin must receive its own separate bounding box.
   - Grouping multiple distinct lesions into a single large encompassing box is **strictly prohibited**.
2. **Handling Touching Lesions:**
   - When two adjacent lesions expand and touch at their borders:
     - If distinct focal centers or separate necrotic cores are visually distinguishable: draw two individual bounding boxes that touch or slightly overlap at the perimeter.
     - If necrotic tissue is completely fused into an indistinguishable single confluent necrotic patch: draw one encompassing bounding box around the confluent patch.
3. **Handling Overlapping Lesions:**
   - In dense symptom clusters, individual bounding boxes may overlap in image space.
   - Same-class bounding boxes are permitted to overlap with $\text{IoU} \le 0.40$. If two annotations overlap by $\text{IoU} > 0.40$ and share a single continuous focus, they must be consolidated into one box.
4. **Minimum Visible Lesion Dimensions:**
   - Minimum bounding box size: **$12 \times 12\text{ pixels}$ ($144\text{ px}^2$)**.
   - Pinhead specks smaller than $10\text{ px}$ diameter ($< 100\text{ px}^2$) should **not** be annotated unless they exhibit a high-contrast necrotic puncture with dark borders. Tiny ambiguous dots below $10\text{ px}$ are treated as sub-visual background noise.
5. **Box Tightness and Boundary Placement:**
   - Bounding boxes must tightly enclose the symptomatic tissue (necrotic center + immediate chlorotic halo).
   - Padding beyond the symptomatic perimeter must not exceed **$2\text{ pixels}$**. Enclosing healthy green leaf lamina within the box must be minimized.
6. **Handling Partially Visible / Truncated Lesions:**
   - Lesions truncated at the leaf margin or image boundary must be boxed tightly to the visible edge ($X=0, Y=0, X=799, Y=799$).
   - Imaginary boxes extrapolating outside the image canvas are strictly forbidden.
7. **Handling Uncertain Disease Labels:**
   - If lesion etiology is ambiguous (e.g. early Bacterial Blight vs early fungal spot): annotate under the primary probable class and set the mutable CVAT attribute `symptom_subtype = Uncertain` for secondary audit.
8. **Healthy-Control Annotation Rules:**
   - Healthy control leaves (`HL*.jpg`) must contain **strictly 0 bounding boxes**.
   - Zero disease targets are serialized as empty label files (0 bytes) and empty detection targets (`boxes: shape (0, 4)`).
9. **Handling Diffuse Architectural Symptoms:**
   - Non-focal whole-leaf symptoms (curling, puckering, diffuse chlorosis) must **not** be forced into arbitrary rectangular boxes. If no distinct focal necrotic lesions exist, 0 detection boxes are created.
10. **Class Taxonomy Invariance:**
    - The canonical 6 foreground disease classes are strictly preserved:
      `1: Bacterial Blight`, `2: Curl Virus`, `3: Herbicide Growth Damage`, `4: Leaf Hopper Jassids`, `5: Leaf Redding`, `6: Leaf Variegation` (with `0: background`).

---

## 4. Class-Specific Annotation Guidance & Architectural Division of Labor

The Phase 9.2C.2 error analysis demonstrated that attempting to force all cotton pathologies into bounding boxes produces metric distortion for whole-leaf architectural conditions. CottonGuard AI enforces a principled multi-modal division of labor:

```
+---------------------------------------------------------------------------------------------------+
|                                  COTTONGUARD AI TRIPARTITE ARCHITECTURE                           |
+------------------------------------+----------------------------------+---------------------------+
| Faster R-CNN ResNet-50 FPN V2      | EfficientNet-B0 Classifier       | U-Net ResNet-34           |
| (Focal Lesion Detection)           | (Whole-Image Diagnosis)          | (Pixel-Level Severity)    |
+------------------------------------+----------------------------------+---------------------------+
| - Bacterial Blight (Angular spots) | - Overall Disease Diagnosis      | - Exact Lesion Boundaries |
| - Leaf Hopper Jassids (V-wedges)   | - Whole-Leaf Morphopathies:      | - Diseased Area (px)      |
| - Discrete Necrotic Lesions        |   * Curl Virus (Lamina curling)  | - Leaf Lamina Area (px)   |
| - Lesion Counting & Spatial Spread |   * Herbicide Damage (Strapping) | - Quantitative Severity:  |
|                                    |   * Leaf Redding (Lamina color)  |   S = (Disease/Leaf)*100% |
+------------------------------------+----------------------------------+---------------------------+
```

### Class-Specific Annotation Protocols

| Disease Class | Primary Visual Symptom | Faster R-CNN Bounding Box Protocol | Delegated Model Role |
|---|---|---|---|
| **Bacterial Blight** | Angular, water-soaked, dark brown to black necrotic spots bounded by veinlets. | **Primary Detection Target:** Annotate individual spots ($12\text{--}80\text{ px}$). Do not group adjacent spots. Exclude healthy green veins. | **Faster R-CNN:** High recall ($75\%+$) spot detection and lesion count. |
| **Leaf Hopper Jassids** | Marginal V-shaped chlorosis and marginal necrosis ("hopperburn"). | **Primary Detection Target:** Annotate individual necrotic marginal wedges and distinct localized chlorotic bands. | **Faster R-CNN:** Marginal wedge localization; **U-Net:** Severity percentage. |
| **Curl Virus** | Upward leaf curling, vein thickening, leaf enation, severe distortion. | **Restricted Protocol:** Only annotate discrete necrotic secondary lesions if present. Do NOT box the curled leaf blade. | **EfficientNet-B0:** Primary diagnostic classification ($98\%+$ accuracy). |
| **Herbicide Growth Damage** | Leaf strapping, puckering, cupping, parallel vein feathering. | **Restricted Protocol:** Only annotate distinct chemical burn focal spots. Do NOT box broad chlorotic interveinal lamina. | **EfficientNet-B0:** Image classification; **U-Net:** Chlorosis segmentation. |
| **Leaf Redding** | Anthocyanin accumulation, purple/red pigmentation on upper lamina. | **Selective Protocol:** Annotate only discrete, dense dark-red focal patches ($> 25\text{ px}$). Ignore light pink continuous lamina blush. | **U-Net:** Measures red pigmentation surface area; **EfficientNet-B0:** Classification. |
| **Leaf Variegation** | Genetic/chimeric white-yellow chlorotic marbling along margins. | **Selective Protocol:** Annotate distinct, well-defined chlorotic patches ($20\text{--}100\text{ px}$). Do not enclose healthy green centers. | **U-Net:** Accurately delineates complex non-convex variegated contours. |

---

## 5. Dataset Expansion Strategy: 50 to 100+ Images

### Target Configuration: 100-Image Balanced Dataset

To achieve statistical validity and alleviate the small-sample overfitting observed in Phase 9.2C, the expanded dataset (designated **Dataset Version 2.0**) will incorporate the staged 50 images of CVAT Task #6:

```
+-------------------------------------------------------------------------------------------------+
|                               EXPANDED DATASET VERSION 2.0 (100 IMAGES)                         |
+--------------------------+--------------------+---------------------+---------------------------+
| Source Partition         | Published Images   | Staged Task #6      | Total Version 2.0         |
+--------------------------+--------------------+---------------------+---------------------------+
| Training Split           | 35 images          | 35 images           | **70 images (70.0%)**     |
| Validation Split         | 15 images          | 15 images           | **30 images (30.0%)**     |
| **Total Corpus**         | **50 images**      | **50 images**       | **100 images (100.0%)**   |
| Estimated Bounding Boxes | 514 boxes          | ~550–650 boxes      | **~1,050–1,150 boxes**    |
+--------------------------+--------------------+---------------------+---------------------------+
```

### Class Balancing in Version 2.0

The 50 images in Task #6 were sampled from the SAR-CLD-2024 original corpus to ensure strict class balancing:

| Disease Class | Tasks #4 + #5 (Published) | Task #6 (Staged) | Total Images in v2.0 | Target Representation |
|---|---:|---:|---:|---:|
| **Bacterial Blight** | 7 images | 7 images | **14 images** | 14.0% |
| **Curl Virus** | 7 images | 7 images | **14 images** | 14.0% |
| **Herbicide Growth Damage** | 7 images | 7 images | **14 images** | 14.0% |
| **Leaf Hopper Jassids** | 7 images | 7 images | **14 images** | 14.0% |
| **Leaf Redding** | 7 images | 7 images | **14 images** | 14.0% |
| **Leaf Variegation** | 8 images | 7 images | **15 images** | 15.0% |
| **Healthy Leaf Controls** | 7 images | 8 images | **15 images** | 15.0% |
| **Total** | **50 images** | **50 images** | **100 images** | **100.0%** |

### Group Isolation and Zero Test-Split Leakage
- **Test Set Protection:** All 100 images have been cross-checked against `artifacts/manifests/splits/test.csv`. Exactly **0 images** from the 321-image test split are present.
- **Group Isolation:** The SAR-CLD-2024 dataset contains duplicate shots of identical leaves assigned to `group_XXXXX`. In Task #6, every candidate image was audited to ensure that no group in the training split appears in the validation split (zero group collision).

---

## 6. Two-Stage Quality Assurance (QA) Process

Before any newly annotated frame in Task #6 can be published, it must pass a formal two-stage QA pipeline:

```
[Annotator Output] ──> [Stage 1: Automated Script Validator] ──> [Stage 2: Independent Visual QA] ──> [Approved for v2.0]
                                 │                                          │
                                 └── FAIL: Reject with log                  └── FAIL: Return for correction
```

### Stage 1: Automated Invariant Validator (`scripts/validate_expansion_annotations.py`)
An automated Python validator evaluates the raw exported annotations against strict geometric constraints:
- **Rule A (Coordinate Sanity):** $0 \le x_1 < x_2 \le 800$ and $0 \le y_1 < y_2 \le 800$.
- **Rule B (Degenerate Rejection):** Rejects any box with $\text{width} < 10\text{ px}$ or $\text{height} < 10\text{ px}$.
- **Rule C (Oversized Box Flag):** Flags for manual review any box with $\text{area} > 60,000\text{ px}^2$ ($> 9.375\%$ of image).
- **Rule D (Healthy Control Gate):** Verifies that all `HL*.jpg` images have **strictly 0 bounding boxes**.
- **Rule E (Duplicate Rejection):** Rejects identical box coordinates ($\text{IoU} = 1.0$) or near-identical boxes ($\text{IoU} > 0.85$ of same class).
- **Rule F (Zero Track Structures):** Confirms that all annotations are `<box>` elements with 0 `<track>` elements.

### Stage 2: Independent Visual QA
- **Audit Sample:** $100\%$ review of the first 10 frames + $25\%$ random stratified sampling of remaining frames.
- **Granularity Inspection:** Audits for composite box violations (forcing annotator to split multi-spot groupings).
- **Vein Exclusion Check:** Confirms that normal dark green primary veins are not enclosed as Bacterial Blight lesions.

---

## 7. Future Evaluation Protocol for Version 2.0 Detector

To ensure reproducible, scientifically rigorous comparisons with the Phase 9.2C baseline:

1. **Continuous PR Integration AP Metric:**
   - The primary metric remains continuous numerical integration of the monotonic Precision-Recall envelope:
     $$\text{AP} = \sum_{k=1}^{N} (R_k - R_{k-1}) \cdot P_{\text{interp}}(R_k)$$
   - Evaluated at primary threshold $\alpha = 0.50$ ($\text{mAP}_{50}$) and averaged over $[0.50:0.05:0.95]$ ($\text{mAP}_{50:95}$).
   - The academic thesis will explicitly document that this metric reflects continuous curve integration and is mathematically distinct from the 101-point discrete COCO sampling approximation.
2. **Dual Operational Operating Points:**
   - **Screening Operating Point ($T = 0.25$):** Prioritizes high recall ($30\text{--}35\%$) for automated alert generation.
   - **Diagnostic UI Rendering Point ($T = 0.40$):** Prioritizes high precision ($30\text{--}40\%+$) and peak F1 ($0.29\text{--}0.35+$) for rendering clean bounding boxes in the user interface.
3. **Scale-Aware Evaluation:**
   - Compute separate AP metrics on:
     - Small Lesions: $\text{area} < 32^2\text{ px}^2$ ($< 1,024\text{ px}^2$)
     - Medium Lesions: $32^2 \le \text{area} \le 96^2\text{ px}^2$
     - Large Lesions: $\text{area} > 96^2\text{ px}^2$ ($> 9,216\text{ px}^2$)
4. **Test Set Preservation:**
   - The sealed 321-image test split (`test.csv`, SHA-256 `6025D86F...`) remains **100% sealed** during all Version 2.0 training, validation, and hyperparameter tuning.

---

## 8. Dataset Versioning Architecture

To maintain reproducible provenance and prevent mixing experimental stages:

```
data/
  annotations/
    v1_published/             # Tasks #4 + #5 (50 image-mask pairs)
    v2_expanded/              # Tasks #4 + #5 + #6 (100 image-mask pairs)
  detection/
    v1_published/             # Tasks #4 + #5 (50 YOLO label files, 514 boxes)
    v2_expanded/              # Tasks #4 + #5 + #6 (100 YOLO label files, ~1,100 boxes)
artifacts/
  manifests/
    combined_pilot_manifest_v1.csv   # 50 images (35 train / 15 val)
    combined_pilot_manifest_v2.csv   # 100 images (70 train / 30 val)
```

---

## 9. Future Train/Validation Split Strategy

The 100-image Version 2.0 dataset preserves exact group isolation:
- **Training Partition (70 images):**
  - 35 images from Published Batch 01 & 02.
  - 35 images from Staged Task #6 (`split == 'train'`).
- **Validation Partition (30 images):**
  - 15 images from Published Batch 01 & 02.
  - 15 images from Staged Task #6 (`split == 'validation'`).
- **Statistical Benefit:** Doubling the validation set from 15 to 30 images narrows confidence intervals, raises Curl Virus validation support from 4 to 11+ boxes, and provides a much more robust empirical signal for early stopping.

---

## 10. Risks and Mitigation Strategies

| Strategic Risk | Probability | Severity | Mitigation Strategy |
|---|---|---|---|
| **Annotator Fatigue & Grouping Lapses** | High | High | Automated Rule C validator flags boxes $> 60,000\text{ px}^2$ and triggers mandatory rejection. |
| **Vein False Positives in Expansion** | Medium | Medium | Strict spectrophotometric contrast guidelines; vein shadows must not be annotated without necrotic discoloration. |
| **CUDA OOM on 70-Image Dataset** | Very Low | Low | Faster R-CNN uses batch size 2; dataset size increases epoch iteration count, not per-step VRAM. Peak VRAM remains $\approx 1.65\text{ GB}$. |
| **Test Set Accidental Leakage** | Low | Critical | Automated preflight hash checks verify that no candidate SHA-256 matches `test.csv`. |

---

## 11. Estimated Annotation Workload

- **Remaining Frames in Task #6:** 47 frames.
  - 8 Healthy Control Frames: $\approx 1\text{ minute/frame}$ (Leaf polygon only, 0 boxes) $= 8\text{ minutes}$.
  - 12 Bacterial Blight & Jassid Frames (Dense focal spots): $\approx 8\text{--}12\text{ minutes/frame} \approx 120\text{ minutes}$.
  - 27 Other Frames (Curl, Redding, Variegation, Herbicide): $\approx 4\text{--}6\text{ minutes/frame} \approx 135\text{ minutes}$.
- **Total Human Annotation Time:** $\approx 4.5\text{ hours}$.
- **Automated Validation & Merge Time:** $\approx 15\text{ minutes}$.
- **Feasibility:** Entire 47-frame expansion can be executed in a single structured operational session.

---

## 12. Recommended Implementation Sequence

```
+---------------------------------------------------------------------------------------------------+
|                               PHASE 9.3 RECOMMENDED IMPLEMENTATION STAGES                         |
+--------+------------------------------------+-----------------------------------------------------+
| Stage  | Phase Identifier                   | Key Actions & Deliverables                          |
+--------+------------------------------------+-----------------------------------------------------+
| **1**  | **Phase 9.3.2 (Annotation Pilot)** | Annotate 5 representative frames in Task #6 using   |
|        |                                    | the Single-Lesion Standard; run Stage 1 QA.         |
| **2**  | **Phase 9.3.3 (Full Annotation)**  | Annotate remaining 42 frames in Task #6; run Stage 1 |
|        |                                    | automated validation and Stage 2 visual audit.      |
| **3**  | **Phase 9.3.4 (Dataset Release)**  | Export clean XML; compile `manifest_v2.csv` (100    |
|        |                                    | images); publish Version 2.0 dataset assets.        |
| **4**  | **Phase 9.3.5 (Retraining v2)**    | Retrain Faster R-CNN on 70 train images (30 epochs);|
|        |                                    | evaluate on 30 val images; benchmark vs Phase 9.2C. |
+--------+------------------------------------+-----------------------------------------------------+
```

---

## 13. Strict Protection Rules & Governance Status

- **Read-Only Verification:** Zero annotations were created or modified during Phase 9.3.1.
- **Task Integrity:** Tasks #3, #4, #5, and #6 remain 100% untouched.
- **Model Checkpoints:** Classifier (`552BABA8...`), U-Net (`C6788F79...`), and Detector (`01D82B32...`) remain strictly preserved.
- **Sealed Test Set:** Holdout split (`test.csv`, SHA-256 `6025D86F...`) remains sealed.
- **Detector Status:** Maintained strictly as **`EXPERIMENTAL / NOT APPROVED FOR PRODUCTION`**.

---

## 14. Conclusion & Next Step

Phase 9.3.1 provides the complete, scientifically grounded blueprint to transition CottonGuard AI from a 50-image pilot to a robust 100-image production-grade dataset. 

Execution is halted awaiting user review and authorization to proceed with **Stage 1 (Phase 9.3.2 Annotation Pilot in Task #6)**.
