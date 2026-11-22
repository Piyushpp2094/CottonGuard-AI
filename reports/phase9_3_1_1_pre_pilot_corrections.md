# CottonGuard AI — Phase 9.3.1.1: Pre-Pilot Annotation Standard Corrections
## Comprehensive Consistency Review, Policy Harmonization, and Five-Image Pilot Specification

**Project Root:** `D:\Main_Project\cotton_leaf_ai`  
**Phase:** 9.3.1.1 — Pre-Pilot Annotation Standard Corrections  
**Date:** 2026-11-15  
**Authorization:** Granted by User for Phase 9.3.1.1 (Read-Only Planning and Consistency Review)  
**Target Detector:** TorchVision Faster R-CNN ResNet-50 FPN V2  
**Baseline Checkpoint:** `artifacts/checkpoints/detection/best_faster_rcnn.pt` (Epoch 13, SHA-256 `01D82B32...`, $\text{val\_mAP}_{50} = 13.94\%$)  
**Current Detector Status:** `EXPERIMENTAL / NOT APPROVED FOR PRODUCTION`  
**Scope Boundary:** Planning correction only. Zero CVAT edits, zero annotations created, zero dataset exports, zero training, sealed test set strictly untouched.

---

## 1. Class-Count Reconciliation and 100-Image Target Distribution

### 1.1 The Arithmetic Discrepancy in the Previous Generalization
The Phase 9.3.1 high-level prose summary stated:
> *"Six disease classes, 14 images per disease class, 15 healthy controls, total target: 100 images."*

Carrying out the direct arithmetic:
$$\text{Calculated Total} = (6 \times 14) + 15 = 84 + 15 = 99 \text{ images}$$

This calculation reveals that the generalized prose statement was missing exactly **1 image** ($100 - 99 = 1$). 

### 1.2 Ground-Truth Audit of Manifests and Task #6 Metadata
To determine the exact ground-truth distribution without inventing images or altering splits, an audit was executed across:
1. `artifacts/manifests/combined_pilot_manifest.csv` (Published Baseline, Tasks #4 + #5, 50 images)
2. `artifacts/manifests/pilot_batch_03_selection_manifest.csv` (Staged Task #6 Expansion Corpus, 50 images)

The audit revealed the exact per-class, per-split breakdown:

```
=== PUBLISHED MANIFEST (combined_pilot_manifest.csv — 50 Images) ===
  Class Label                 Train Split    Validation Split    Total Images
  Bacterial Blight                 5                 2                 7
  Curl Virus                       5                 2                 7
  Herbicide Growth Damage          5                 2                 7
  Leaf Hopper Jassids              5                 2                 7
  Leaf Redding                     5                 2                 7
  Leaf Variegation                 5                 3                 8  <-- Note: 8 images
  Healthy Leaf (Control)           5                 2                 7
  -----------------------------------------------------------------------------
  Published Total                 35                15                50

=== STAGED TASK #6 MANIFEST (pilot_batch_03_selection_manifest.csv — 50 Images) ===
  Class Label                 Train Split    Validation Split    Total Images
  Bacterial Blight                 5                 2                 7
  Curl Virus                       5                 2                 7
  Herbicide Growth Damage          5                 2                 7
  Leaf Hopper Jassids              5                 2                 7
  Leaf Redding                     5                 2                 7
  Leaf Variegation                 5                 2                 7
  Healthy Leaf (Control)           5                 3                 8  <-- Note: 8 images
  -----------------------------------------------------------------------------
  Task #6 Staged Total            35                15                50
```

### 1.3 Consolidated 100-Image Ground-Truth Distribution (Version 2.0)
When Published (50 images) and Staged Task #6 (50 images) are merged:

| Canonical Class Name | Published (T4+T5) | Staged (T6) | Total v2.0 Images | Class % | Train Count | Val Count |
|---|---:|---:|---:|---:|---:|---:|
| **Bacterial Blight** | 7 | 7 | **14** | 14.0% | 10 | 4 |
| **Curl Virus** | 7 | 7 | **14** | 14.0% | 10 | 4 |
| **Herbicide Growth Damage** | 7 | 7 | **14** | 14.0% | 10 | 4 |
| **Leaf Hopper Jassids** | 7 | 7 | **14** | 14.0% | 10 | 4 |
| **Leaf Redding** | 7 | 7 | **14** | 14.0% | 10 | 4 |
| **Leaf Variegation** | 8 | 7 | **15** | 15.0% | 10 | 5 |
| **Healthy Leaf (Controls)** | 7 | 8 | **15** | 15.0% | 10 | 5 |
| **Total Corpus** | **50** | **50** | **100** | **100.0%** | **70 (70.0%)** | **30 (30.0%)** |

### 1.4 Reconciliation Finding
* **The "Missing" Image:** The 100th image belongs to **Leaf Variegation**, which contains **15 images** ($15.0\%$), not 14 images.
* **Symmetry:** Exactly five disease classes have 14 images ($5 \times 14 = 70$). Leaf Variegation has 15 images. Healthy Leaf controls have 15 images. Total: $70 + 15 + 15 = 100$ images.
* **Balanced Training Partition:** In the 70-image training split, **every single class has exactly 10 images** ($7 \times 10 = 70$). There is zero class imbalance in the training partition.
* **Validation Partition:** The 30-image validation split contains 4 images for each of the five focal/systemic disease classes ($5 \times 4 = 20$), 5 images for Leaf Variegation, and 5 images for Healthy Leaf ($20 + 5 + 5 = 30$).
* **Mathematical Invariant:** The canonical $70.0\% / 30.0\%$ split ratio is preserved exactly without altering manifests, inventing images, or shifting any sample.

---

## 2. Harmonized Bounding-Box Size Policy

### 2.1 The Policy Conflict
In Phase 9.3.1, the annotation guideline recommended a minimum lesion dimension of $12 \times 12\text{ pixels}$ ($144\text{ px}^2$), while the proposed Stage 1 automated validator rejected boxes below $10\text{ pixels}$. This created an unhandled grey area for boxes measuring $10$ or $11\text{ pixels}$.

### 2.2 Three-Tier Bounding-Box Size Policy
To eliminate ambiguity and ensure strict mathematical consistency between annotator guidelines and automated QA validation, the following three-tier policy is established:

```
[Box Size (Width or Height)]
      │
      ├── < 10 px  ──────────────────> [TIER 1: UNCONDITIONAL HARD REJECT]
      │                                 - Sub-visual noise / double-click slip
      │                                 - Cannot support RoIAlign feature extraction (stride 4)
      │                                 - Mandatory excision
      │
      ├── 10 to 11 px  ──────────────> [TIER 2: CONDITIONAL WARNING / MANUAL REVIEW]
      │                                 - Automated validator emits WARNING: BORDERLINE_SMALL_LESION
      │                                 - Approved ONLY if 100% zoom confirms high-contrast necrotic core
      │                                 - Excluded if faint, diffuse, or ambiguous
      │
      └── >= 12 px (Area >= 144 px²) ─> [TIER 3: ELIGIBLE FOR AUTOMATED ACCEPTANCE]
                                        - Standard operational scale for Faster R-CNN
                                        - Accepted subject to semantic correctness and single-lesion rule
```

### 2.3 Detailed Tier Specifications

| Policy Tier | Dimension & Area Constraints | Validator Code & Status | Human Review Requirement | Rationale & Architectural Grounding |
|---|---|---|---|---|
| **Tier 1: Degenerate Box** | $\text{width} < 10\text{ px}$ **OR** $\text{height} < 10\text{ px}$ **OR** $\text{area} < 100\text{ px}^2$ | `ERROR: DEGENERATE_BOX_TOO_SMALL` (Rejection) | Automatic rejection; no manual override permitted. | At ResNet-50 FPN stride 4 (P2), a box $< 10\text{ px}$ occupies $< 2.5$ feature map bins. RoIAlign pooling cannot extract discriminative representation, injecting gradient noise. Represents dust specks or annotation slips. |
| **Tier 2: Borderline Small** | $10 \le \text{width} \le 11\text{ px}$ **AND** $10 \le \text{height} \le 11\text{ px}$ ($\text{area} \ge 100\text{ px}^2$) | `WARNING: BORDERLINE_SMALL_LESION` (Flagged) | **Mandatory 100% zoom visual audit.** Approved only if an expert confirms a dark necrotic puncture with sharp margins. | Early-stage Bacterial Blight pinhead punctures may present at 10–11px. Legitimate lesions must not be discarded arbitrarily, but faint background artifacts must be excised. |
| **Tier 3: Standard Accepted** | $\text{width} \ge 12\text{ px}$ **AND** $\text{height} \ge 12\text{ px}$ ($\text{area} \ge 144\text{ px}^2$) | `PASS: VALID_DIMENSIONS` (Accepted) | Standard stratified visual QA ($25\%$ sample). | Matches Faster R-CNN anchor scale ($32^2, 64^2, 128^2, 256^2, 512^2$) with FPN P2 interpolation. |
| **Oversized Box Flag** | $\text{area} > 60,000\text{ px}^2$ ($> 9.375\%$ of $800 \times 800$) | `WARNING: POTENTIAL_COMPOSITE_GROUPING` (Flagged) | **Mandatory 100% visual QA.** Check for multi-spot grouping. | In Phase 9.2C.2, composite grouping boxes artificially inflated false alarms. Large boxes must be audited to ensure they represent single confluent lesions. |

---

## 3. Revised Overlapping Box Policy

### 3.1 Overlap as an Inspection Trigger, Not an Automatic Rejection
The crude rule $\text{IoU} \le 0.40$ previously proposed risked rejecting legitimate biological lesion configurations. In field pathology, fungal and bacterial spots frequently cluster along leaf veins. Faster R-CNN RoIAlign heads can localize distinct lesion centers even when their bounding boxes intersect in 2D Cartesian space.

Therefore, **bounding-box overlap is established as a review trigger rather than an unconditional rejection criterion**.

### 3.2 Five Spatial Scenarios and Handling Protocols

```
  SCENARIO A: Distinct Adjacent          SCENARIO B: Touching              SCENARIO C: Genuinely Overlapping
       +-----+      +-----+                   +-----+-----+                     +-----------+
       | (o) |      | (o) |                   | (o) | (o) |                     |  (o)  +---|---+
       +-----+      +-----+                   +-----+-----+                     |       |   |   |
      IoU = 0.0 (Gap >= 1 px)                0.0 < IoU <= 0.10                  +-------|---+   |
      PASS: Auto-accept.                     PASS: Auto-accept.                         +-------+
                                                                             0.10 < IoU <= 0.50 (Distinct cores)
                                                                             REVIEW TRIGGER: Audit distinct foci.

  SCENARIO D: Confluent / Fused          SCENARIO E: Duplicate Box
       +-------------------+                  +=============+
       |   ( oooooooo )    |                  |  (o)  (o)   |
       | (oooooooooooooo)  |                  |  Duplicate  |
       +-------------------+                  +=============+
      Fused necrotic patch.                  IoU >= 0.70 (Same class)
      MUST BE ONE SINGLE BOX.                HARD REJECT: Remove duplicate.
```

### 3.3 Overlap Protocol Matrix

| Scenario | Spatial Definition & Metric | Biological Reality | Annotation Rule | Automated Validator Action |
|---|---|---|---|---|
| **A. Distinct Adjacent** | $\text{IoU} = 0.0$, inter-box distance $d \ge 1\text{ px}$. | Separate infection events separated by healthy green leaf blade. | Draw separate tight box around each lesion. | `PASS: DISCRETE_BOXES` — Accepted automatically. |
| **B. Touching Lesions** | $0.0 < \text{IoU} \le 0.10$, boxes abut or graze edges. | Adjacent expanding lesions whose margins touch but centers remain separate. | Draw two separate touching boxes if separate necrotic centers are visible. | `PASS: TOUCHING_BOXES` — Accepted automatically. |
| **C. Genuinely Overlapping** | $0.10 < \text{IoU} \le 0.50$. | Dense cluster of distinct necrotic spots whose rectangular hulls intersect. | Draw separate boxes around each necrotic center. | $\text{IoU} \le 0.30$: `INFO: MODERATE_OVERLAP` (auto-accepted if centers distinct).<br>$0.30 < \text{IoU} \le 0.50$: `WARNING: SIGNIFICANT_OVERLAP_SAME_CLASS` (triggers manual review; accepted if separate cores confirmed). |
| **D. Completely Fused (Confluent)** | Lesions coalesced into an indistinguishable necrotic sheet. | Secondary necrosis merging multiple initial lesions into a continuous dead area. | **Draw ONE encompassing box** around the entire confluent patch. Drawing artificial overlapping boxes is prohibited. | Stage 2 visual QA enforces single box. If multiple boxes are drawn over a single continuous lesion, reject and consolidate. |
| **E. Duplicate Annotations** | $\text{IoU} \ge 0.70$ (or identical coordinates $\text{IoU} = 1.0$). | Human annotator double-click slip or accidental re-drawing. | Prohibited. Exactly one box per lesion. | `ERROR: DUPLICATE_BOX_DETECTED (IoU >= 0.70)` — **Hard rejection.** Annotator must delete the duplicate shape. |

---

## 4. Coordinate and Padding Standard

### 4.1 Canonical Coordinate Space: Native $800 \times 800$ Resolution
An empirical inspection of image assets across `data/` and the original SAR-CLD-2024 repository was conducted:
* **Disk Verification:** Every image in the published dataset (`data/annotations/images/`, `data/detection/images/`) and staged Task #6 (`pilot_batch_03_selection_manifest.csv`) has dimensions of **$800 \times 800\text{ pixels}$**.
* **Zero Resizing Artifacts:** Because the native dataset images are already $800 \times 800$, the CVAT annotation canvas ($800 \times 800$), the exported XML coordinates, the YOLO label normalization basis, and the Faster R-CNN input tensor dimensions ($800 \times 800$) are **identical**.
* **Resolution Rule:** All annotation coordinates, bounding box boundaries, and padding rules apply directly and strictly to the **canonical $800 \times 800$ pixel grid**.

### 4.2 The 2-Pixel Padding Rule
* **Definition:** Bounding boxes must enclose the visible diseased tissue (the central necrotic core plus any surrounding chlorotic halo) with a maximum margin of **$2\text{ pixels}$** of healthy green leaf tissue.
* **Why $2\text{ Pixels}$?**
  - **Zero Padding ($0\text{ px}$):** Drawing boxes strictly inside or flush against necrotic borders frequently clips chlorotic transition zones, depriving the RoI head of critical edge contrast cues.
  - **Excessive Padding ($> 5\text{ px}$):** Enclosing wide margins of healthy green leaf lamina dilutes foreground feature statistics, confusing the detector and causing false-positive alarms on normal leaf veins (which accounted for $44.6\%$ of FPs in Phase 9.2C.2).
  - **The 2-Pixel Optimum:** Provides a clean $1\text{--}2\text{ pixel}$ margin around the lesion halo, maximizing foreground feature density while preserving local edge context.

### 4.3 Coordinate Transformation and Invariant Pipeline
The coordinate lifecycle across the CottonGuard AI pipeline is formalized as follows:

```
[CVAT Canvas: 800x800 px]
       │
       ▼ (Export CVAT for images 1.1 XML)
<box xtl="120.5" ytl="340.0" xbr="145.2" ybr="368.5" label="Diseased_Region">
  <attribute name="disease_type">Bacterial Blight</attribute>
</box>
       │
       ▼ (scripts/prepare_cvat_annotations.py — Ingestion & Normalization)
YOLO Format (.txt): [class_id  x_center/800  y_center/800  w/800  h/800]
  x_c = (xtl + xbr) / (2 * 800)
  y_c = (ytl + ybr) / (2 * 800)
  w_n = (xbr - xtl) / 800
  h_n = (ybr - ytl) / 800
       │
       ▼ (src/data/detection_dataset.py — TorchVision Detector Input)
target['boxes'] = torch.tensor([[xtl, ytl, xbr, ybr]], dtype=torch.float32)
target['labels'] = torch.tensor([class_id], dtype=torch.int64)
```

### 4.4 Boundary Invariants
1. **Coordinate Clamping:** All bounding box coordinates must lie strictly within the canvas:
   $$0.0 \le x_{\text{tl}} < x_{\text{br}} \le 800.0, \quad 0.0 \le y_{\text{tl}} < y_{\text{br}} \le 800.0$$
2. **Truncated Edge Lesions:** If a lesion extends to the physical leaf edge or image border, the box coordinates must be clamped to the edge ($0.0$ or $800.0$). Never extrapolate imaginary coordinates outside the image canvas.
3. **No Retroactive Coordinate Changes:** Existing published annotation coordinates in Batch 01 and Batch 02 (`data/detection/labels/`) remain strictly frozen.

---

## 5. Five-Image Pilot Scope and Selection Criteria

### 5.1 Confirmation of Pilot Scope
The pilot scope is strictly confined to **exactly five images**:
- Review and refine the three already annotated frames in Task #6.
- Select and annotate two additional suitable frames from Task #6.
- Run automated Stage 1 QA validation.
- Perform 100% visual Stage 2 QA review on all five frames.
- **Strict Boundary:** The remaining 42 frames of Task #6 will **not** be annotated or modified during the pilot.

### 5.2 Status of the Three Existing Annotated Frames in Task #6
A read-only audit of `artifacts/reports/cvat_task6_phase9_0f_exported_annotations_20261114_235937.xml` confirmed:

```
Frame 00: BBC00001.jpg | Bacterial Blight | Train Split | Group group_00001
  Current Shapes: 1 Leaf polygon + 19 Diseased_Region polygons (0 tracks)
  Severity: S = 4.14% | 0 px outside leaf | 0 polygon overlaps
  Role in Pilot: Benchmark for discrete focal necrotic spots. Verify whether all 19 derived 
                 bounding boxes satisfy the Single-Lesion Standard and the >= 12px rule.

Frame 16: LHJ00059.jpg | Leaf Hopper Jassids | Train Split | Group group_01270
  Current Shapes: 1 Leaf polygon + 9 Diseased_Region polygons (0 tracks)
  Severity: S = 24.52% | 0 px outside leaf | 0 polygon overlaps
  Role in Pilot: Benchmark for marginal chlorosis/necrosis ("hopperburn"). Verify marginal 
                 box tightness, check for touching/overlapping wedge boxes, and test 2px padding.

Frame 30: HL00002.jpg | Healthy Leaf | Train Split | Group group_00683
  Current Shapes: 1 Leaf polygon + 0 Diseased_Region polygons (0 tracks)
  Severity: S = 0.00% | 0 px outside leaf | 0 diseased shapes
  Role in Pilot: Negative control benchmark. Verify that automated validation enforces 
                 strictly 0 bounding boxes and produces an empty YOLO label file (0 bytes).
```

### 5.3 Selection Criteria for the Two Additional Pilot Frames
To maximize the diagnostic value of the 5-image pilot, the two additional frames must satisfy five objective criteria:
1. **Validation Split Inclusion:** The three existing frames are all in the `train` split. At least one additional frame must belong to the `validation` split to verify validation data flow and directory routing.
2. **Dense Spot Clustering:** Must contain dense clusters of small lesions to stress-test the touching vs overlapping vs confluent rules (Scenarios B, C, D) and the 10–11px borderline size policy.
3. **Multi-Modal Boundary Testing:** Must test a class prone to cross-class confusion (Leaf Redding or Leaf Variegation) to enforce the selective protocol (annotating discrete dark necrotic/pigmented foci while ignoring diffuse pink lamina blush).
4. **Anatomical Cleanliness (Single-Leaf Invariant):** Must be a clean single-leaf image free from the multi-leaf rosette defect that disqualified `CV00211` or the genetic chimera issue that disqualified `LV00017` in Phase 9.0C.
5. **Group Isolation & Test Split Protection:** Must have 0 hash matches against `artifacts/manifests/splits/test.csv` and zero group overlap with other splits.

### 5.4 Recommended Additional Pilot Frames

| Frame # | Filename | Canonical Class | Split | Group ID | Selection Justification |
|---|---|---|---|---|---|
| **Frame 35** | `BBC00008.jpg` | **Bacterial Blight** | **Validation** | `group_00008` | **Validation Focal Spot Benchmark:** Features multiple dense angular necrotic spots across veinlets in the validation partition. Stress-tests single-lesion separation, touching box rules, and the 10–11px borderline review trigger. Confirms validation pipeline integrity. |
| **Frame 20** | `LR00001.jpg` | **Leaf Redding** | **Train** | `group_01437` | **Selective Protocol Benchmark:** Features anthocyanin reddening with discrete darker focal patches. Tests annotator discipline in boxing only distinct focal lesions ($> 25\text{ px}$) while ignoring diffuse background pink lamina blush, directly addressing the #2 cause of false positives identified in Phase 9.2C.2. |

### 5.5 Consolidated 5-Image Pilot Composition

```
+-------------------------------------------------------------------------------------------------+
|                                    FIVE-IMAGE PILOT SUITE                                       |
+-------+---------------+----------------------+------------+------------------+------------------+
| Frame | Filename      | Disease Class        | Split      | Lesion Topology  | Primary QA Role  |
+-------+---------------+----------------------+------------+------------------+------------------+
| 00    | BBC00001.jpg  | Bacterial Blight     | Train      | Discrete spots   | Focal Necrosis   |
| 16    | LHJ00059.jpg  | Leaf Hopper Jassids  | Train      | Marginal wedges  | Hopperburn / Halos|
| 30    | HL00002.jpg   | Healthy Leaf         | Train      | Zero lesions     | Negative Control |
| 35    | BBC00008.jpg  | Bacterial Blight     | Validation | Dense clusters   | Val Split / Size |
| 20    | LR00001.jpg   | Leaf Redding         | Train      | Pigment patches  | Diffuse Blushing |
+-------+---------------+----------------------+------------+------------------+------------------+
```

---

## 6. Pilot Acceptance Criteria

Before the 5-image pilot can be marked `PASS` and full annotation of the remaining 42 frames authorized, all five images must satisfy the following criteria:

### 6.1 Automated Stage 1 QA Gates
* **Gate 1 (Zero Degenerate Boxes):** Exactly 0 boxes with $\text{width} < 10\text{ px}$ or $\text{height} < 10\text{ px}$ or $\text{area} < 100\text{ px}^2$.
* **Gate 2 (Small Box Accounting):** Any box with dimension $10\text{--}11\text{ px}$ must be logged with an explicit reviewer verification code in the QA audit log.
* **Gate 3 (Zero Duplicate Boxes):** Exactly 0 box pairs of the same class with $\text{IoU} \ge 0.70$.
* **Gate 4 (Overlap Verification):** Any box pair with $0.30 < \text{IoU} < 0.70$ must have verified distinct centroid coordinates and visual confirmation of separate necrotic cores.
* **Gate 5 (Healthy Negative Control):** Frame 30 (`HL00002.jpg`) must export **strictly 0 bounding boxes** and produce an empty YOLO label file.
* **Gate 6 (Coordinate Sanity):** 100% of box coordinates must satisfy $0.0 \le x_1 < x_2 \le 800.0$ and $0.0 \le y_1 < y_2 \le 800.0$.
* **Gate 7 (Zero Track Structures):** CVAT export must contain strictly 0 `<track>` elements.

### 6.2 Visual Stage 2 QA Gates (100% Review of all 5 Images)
* **Gate 8 (Single-Lesion Compliance):** Zero composite multi-spot grouping boxes. Every discrete necrotic center must have its own box.
* **Gate 9 (Vein Discoloration Protection):** Normal dark green primary/secondary leaf veins must not be enclosed as Bacterial Blight lesions.
* **Gate 10 (Boundary Padding):** Bounding boxes must adhere to the $2\text{ px}$ padding limit (no excess healthy lamina enclosed).
* **Gate 11 (Diffuse Boundary Discipline):** In Frame 20 (`LR00001.jpg`), general pink blush is excluded; only discrete dark pigmented/necrotic patches are boxed.

### 6.3 Cryptographic and Data Isolation Gates
* **Gate 12 (Zero Test Leakage):** SHA-256 hashes of all five images verified against `artifacts/manifests/splits/test.csv` (0 matches).
* **Gate 13 (Model & Database Invariant):** EfficientNet-B0 (`552BABA8...`), U-Net (`C6788F79...`), Faster R-CNN (`01D82B32...`), and SQLite DB verified 100% untouched.

---

## 7. Remaining Unresolved Questions and Design Decisions

1. **Polygon-First vs Native-Box Workflow in CVAT:**
   - *Question:* Should annotators continue drawing high-precision `<polygon>` elements for `Diseased_Region` (which are converted to bounding boxes during ingestion), or should they draw native CVAT `<box>` shapes?
   - *Recommendation:* **Maintain the polygon-first workflow.** Drawing high-precision polygons provides the dual-use ground truth needed for both U-Net semantic segmentation masks and Faster R-CNN bounding boxes. Computing axis-aligned bounding boxes from polygons is deterministic, reproducible, and mathematically exact.
2. **Confluent Lesion Consolidation Threshold:**
   - *Question:* At what exact geometric threshold should two expanding adjacent lesions be treated as confluent (one box) versus touching (two boxes)?
   - *Resolution:* If a bridge of necrotic tissue connecting two lesions is wider than $50\%$ of the narrower lesion's diameter, the lesions are defined as confluent and must receive **one single bounding box**. If the connecting neck is $< 50\%$ or consists of chlorotic/healthy lamina, they must receive **two separate touching boxes**.
3. **Annotator Throughput on Dense Frames:**
   - *Question:* Will drawing 20–40 single-lesion polygons on dense Bacterial Blight frames cause annotator fatigue and introduce inconsistency?
   - *Mitigation:* Focus high-density annotation strictly on lesions $\ge 12\text{ px}$. Tiny ambiguous specks $< 10\text{ px}$ are rejected automatically by Tier 1. Fused clusters are consolidated under Rule 2/Scenario D.

---

## 8. Governance and Phase Boundary Confirmation

* **Current Status:** Phase 9.3.1.1 is complete.
* **Detector Governance Status:** Maintained strictly as **`EXPERIMENTAL / NOT APPROVED FOR PRODUCTION`**.
* **Zero Modification Invariant:** Zero CVAT annotations were created or modified. Checkpoints, database, manifests, and the sealed 321-image test set remain 100% untouched.
* **Next Gate:** Awaiting explicit user authorization to proceed with **Phase 9.3.2 (Five-Image Annotation Pilot in Task #6)**.
