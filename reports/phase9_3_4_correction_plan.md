# CottonGuard AI — Phase 9.3.4: Annotation Correction Preparation & Execution Plan
**Phase:** 9.3.4 — Annotation Correction Preparation & Execution Plan  
**Target Environment:** CVAT Task #6 (`CottonGuard_Pilot_Batch_03_ShapesOnly`, Job #6)  
**Reference Deliverable:** `reports/phase9_3_3_independent_visual_annotation_audit.md`  
**Master Plan Deliverable:** `reports/phase9_3_4_correction_plan.md`  
**Audit Type:** Strictly Read-Only Planning & Engineering Specification (Zero CVAT/Dataset Writes)  
**Date:** November 16, 2026  

---

## 1. Executive Summary & Audit Context

Phase 9.3.3 delivered an independent, second-level visual and coordinate audit of the five-image annotation pilot staged in CVAT Task #6 (Job #6). While the automated geometric pipeline in Phase 9.3.2 passed all 14 Stage 1 validator gates, the second-level visual audit revealed critical biological and morphological discrepancies that require precise, disciplined correction:

1. **Frame 16 (`LHJ00059.jpg`):** Formally **REJECTED**. The 5 decomposed marginal wedges (`Shape #08`–`#12`) engulfed over $80,000\text{ px}$ of completely normal green lamina ($\Delta E < 6.0$ relative to central healthy leaf), resulted from arbitrary planar coordinate slicing (`y=340`, `y=490`, `x=440`), included up to $41.4\%$ empty background, and grossly inflated disease severity to $24.53\%$.
2. **Frame 35 (`BBC00008.jpg`):** **APPROVED WITH CORRECTIONS**. 41 boxes were confirmed as authentic single lesions, but 13 boxes require curation: 4 composite multi-veinlet boxes violating the Single-Lesion Standard, 3 elongated boxes tracking linear vein crevices, 1 sub-threshold non-lesion (`Box #52`, $\Delta L = 9.7$), and 1 border-truncated lesion (`Box #07`).
3. **Frame 20 (`LR00001.jpg`):** **APPROVED WITH CORRECTIONS**. While 12 focal dark patches were successfully isolated, **310,308 pixels (48.49% of the leaf)** exhibit diffuse pink/red anthocyanin blush that was left unannotated as background, creating severe label contradictions for object detection. Furthermore, the leaf polygon is an artificial 4-point rectangle (`[4, 4, 796, 796]`).
4. **Frame 00 (`BBC00001.jpg`):** **APPROVED WITH CORRECTIONS**. High-quality single lesions, but `Box #12` and `Box #19` form a touching/overlapping pair ($IoU = 0.0448$, 105 px polygon overlap) along the central main vein.
5. **Frame 30 (`HL00002.jpg`):** **APPROVED**. Confirmed strictly 1 Leaf polygon, 0 disease shapes, 0 background in leaf mask. Pristine negative control.

> [!IMPORTANT]
> **Objective of Phase 9.3.4:** Establish a comprehensive, mathematically grounded, and botanically verified correction plan for all five pilot frames. **Training and CVAT modifications are strictly unauthorized in this planning phase.** Execution will take place only under explicit user authorization in Phase 9.3.4.1.

---

## 2. Per-Frame Detailed Correction Plan

### 2.1 Frame 16: `LHJ00059.jpg` — Leaf Hopper Jassids (Re-Annotation Specification)

#### Problem Diagnosis
The five decomposed disease wedges (`Shape #08` to `#12`) were produced by taking a coarse, unconstrained SAM segmentation mask and applying straight coordinate cutlines at $y=340$, $y=490$, and $x=440$. This resulted in:
- Massive bounding boxes ($277 \times 164\text{ px}$, $212 \times 199\text{ px}$) that enclose tens of thousands of pixels of healthy green lamina.
- Empty background inclusion up to $41.4\%$ inside the boxes.
- Truncation of primary veins by artificial straight boundaries.
- Unrealistic reported severity ($24.53\%$, $92,490\text{ px}$ of disease on $377,013\text{ px}$ leaf).

#### Botanical Re-Annotation Strategy
In cotton entomology, leafhopper (*Amrasca biguttula biguttula*) damage manifests as **hopperburn**:
1. **Marginal Symptom Band:** Toxic salivary injection during phloem feeding causes curling (cupping) and progressive chlorosis (yellowing) that strictly follows the peripheral margin of the leaf lobes.
2. **Colorimetric Separation:**
   - Authentic chlorotic marginal curling exhibits HSV Hue $H < 55^\circ$ (yellow/amber) and Lab $b^* > 145$ (increased yellow chromaticity).
   - Normal interior lamina exhibits HSV Hue $H = 65^\circ - 75^\circ$ and Lab $b^* \le 143$.
   - Late-stage necrotic curling exhibits Lab $L < 120$ or Hue $H < 25^\circ$.
3. **Exclusion Criteria:**
   - Exclude all green interior parenchyma ($H \ge 56^\circ$, $L \ge 150$).
   - Exclude primary and secondary veins unless direct vascular necrosis is present.
   - Exclude all image background ($L < 30$ or $L > 240$ outside the leaf contour).
4. **Morphological Separation by Leaf Lobe:**
   - Rather than arbitrary coordinate slicing, separate the marginal ribbon by **anatomical leaf lobes and sinuses**:
     * *Apical Lobe Margin Ribbon:* Curled edge along the upper left margin ($Y \in [11, 180]$).
     * *Lateral Left Lobe Ribbon:* Symptomatic margin between upper and lower sinus ($Y \in [180, 420]$).
     * *Basal Left Lobe Ribbon:* Curled basal margin ($Y \in [420, 620]$).
     * *Basal Right Lobe Ribbon:* Curled margin near petiole sinus ($X \in [240, 440], Y \in [640, 785]$).
     * *Lateral Right Lobe Ribbon:* Curled edge along the lower right margin ($X \in [440, 655], Y \in [580, 785]$).
   - Each ribbon is delineated as a narrow polygon (width $5 - 20\text{ px}$) strictly bounding the curled/yellowed tissue.
5. **Bounding Box Delineation:**
   - Subdivide elongated marginal ribbons into discrete sector bounding boxes (maximum dimension $\le 80\text{ px}$) centered on distinct curl folds, ensuring fill factors $> 0.35$ and zero inclusion of central veins.
6. **Recalculation of Severity from Corrected Mask:**
   - Do NOT force an arbitrary severity percentage. Severity MUST be recalculated strictly from the newly rasterized mask:
     $$S_{\text{corrected}} = \frac{\sum \text{Pixel}(M_{\text{disease, corrected}} == 2)}{\sum \text{Pixel}(M_{\text{leaf}} \ge 1)} \times 100\%$$
   - Based on our quantitative spectrophotometric transect of the outer $25\text{ px}$ margin, the authentic chlorotic pixels measure $19,790\text{ px}$ and necrotic pixels measure $1,006\text{ px}$, projecting an authentic biological severity of $\approx 5.5\%$, down from the erroneous $24.53\%$.

### 2.2 Frame 35: `BBC00008.jpg` — Bacterial Blight (13-Box Curation Specification)

Frame 35 contains 54 candidate disease annotations. Our independent audit verified that 41 boxes are gold-standard single lesions, while 13 boxes require specific curation. The detailed box-by-box remediation plan is presented in Section 3.

#### Key Principles for Frame 35 Remediation:
1. **Single-Lesion Decomposition:** Bounding boxes spanning multiple interveinal compartments separated by secondary veins (`Box #22`, `#33`, `#43`, `#08`) must be split along vein boundaries into independent single-lesion boxes.
2. **Vein Shadow Crevice Snugging:** Elongated boxes tracking linear vein crevices (`Box #34`, `#41`, `#48`) must be inspected: if the darkened line is a non-necrotic shadow groove, it is deleted; if it contains a true focal lesion, the box must be snugged to the parenchymal spot, removing the linear vein tail.
3. **Sub-Threshold Artifact Deletion:** `Box #52` has $\Delta L = 9.7$ and zero necrotic contrast. It is scheduled for deletion.
4. **Boundary Truncation Flagging:** `Box #07` is clipped by the left frame border. It will be retained for training but flagged with `is_boundary_truncated=True` so it can be excluded from evaluation benchmarking.

### 2.3 Frame 20: `LR00001.jpg` — Leaf Redding (Dual-Tier Policy & Leaf Boundary)

#### 1. Formal Dual-Tier Annotation Architecture
Leaf Redding in cotton is a complex physiological disorder. Anthocyanin accumulation occurs in two distinct biological expressions, which must NOT be conflated in computer vision training:

```
+-------------------------------------------------------------------------------------------------------+
|                            DUAL-TIER LEAF REDDING ANNOTATION ARCHITECTURE                             |
+-----------------------------------+-------------------------------------------------------------------+
| TIER 1: FOCAL LESIONS             | TIER 2: DIFFUSE ANTHOCYANIN BLUSH                                 |
| (Target: Faster R-CNN Detector)   | (Target: U-Net Segmenter & Classifier)                            |
+-----------------------------------+-------------------------------------------------------------------+
| - Biological definition: Intense, | - Biological definition: Widespread, gradient-based interveinal   |
|   localized cellular breakdown &  |   reddening/pink blush across entire leaf lobes without margins.  |
|   senescent necrosis.             |                                                                   |
| - Threshold: L < 150, a* > 148    | - Threshold: L >= 150, a* > 135                                   |
| - Bounding Box: Discrete tight    | - Bounding Box: STRICTLY NONE. (Drawing large bounding boxes      |
|   boxes (10-80 px) with fill >0.35|   around diffuse blush dilutes feature statistics with veins).    |
| - CVAT Storage: Shape 'polygon' / | - CVAT Storage: Separate polygon layer labeled 'Diffuse_Reddening'|
|   'box' labeled 'Diseased_Region' |   or attribute 'tier=diffuse'.                                    |
|   with attribute 'focal_type=dark'|                                                                   |
| - Export: Included in YOLO label  | - Export: EXCLUDED from YOLO label files. Exported into multi-    |
|   files for Faster R-CNN detector.|   class segmentation masks (Class 3 = Diffuse Reddening).         |
+-----------------------------------+-------------------------------------------------------------------+
```

#### 2. Resolving Contradictory Training Labels for Faster R-CNN
In standard Faster R-CNN training, any region of the image NOT covered by a ground-truth bounding box is treated as a **negative background candidate**. Because $310,308\text{ px}$ (48.49% of the leaf) has visible pink/red discoloration, sampling negative anchors on this tissue penalizes the detector for recognizing reddening, causing extreme false-negative rates.

**Solution:**
When generating training batches for Faster R-CNN, the Tier 2 Diffuse Reddening mask is converted into an **Ignore Mask / Neutral Region**:
- RPN anchors whose intersection with the Tier 2 diffuse mask exceeds $0.30$ are marked as `IGNORE` (target label `-1`), excluding them from both positive and negative loss calculations.
- This allows the detector to focus strictly on detecting distinct focal lesions without being penalized when anchors overlap diffuse background blush.

#### 3. Leaf Polygon Resolution on Macro Frame 20
- Forensic inspection proved that `LR00001.jpg` is a 100% macro close-up. Minimum pixel brightness across the entire perimeter is $> 40$, and near-black pixels count is strictly 0. Cotton leaf lamina fills every corner.
- The current 4-point rectangle `[4.00, 4.00; 796.00, 4.00; 796.00, 796.00; 4.00, 796.00]` encloses $628,889\text{ px}$ ($98.3\%$ of canvas), leaving an arbitrary 4px perimeter border.
- **Recommendation:** Replace the 4px offset rectangle with a full-canvas polygon `[0.0, 0.0; 800.0, 0.0; 800.0, 800.0; 0.0, 800.0]` ($640,000\text{ px}$) and add a mandatory image-level metadata tag: `is_full_frame_macro = True`.

### 2.4 Frame 00: `BBC00001.jpg` — Bacterial Blight (Box #12 & #19 Geometry Preservation)

#### Diagnostic Findings on Box #12 and Box #19
- `Box #12`: $X \in [519, 567], Y \in [334, 369]$ ($48 \times 35\text{ px}$, Area $1,680\text{ px}^2$).
- `Box #19`: $X \in [488, 529], Y \in [314, 347]$ ($41 \times 33\text{ px}$, Area $1,353\text{ px}^2$).
- **Geometric Intersection:** $X \in [519, 529], Y \in [334, 347]$ ($10 \times 13\text{ px}$, Area $130\text{ px}^2$, $IoU = 0.0448$).
- **Polygon Overlap:** The two segmentation polygons share $105\text{ px}$ of overlapping mask.

#### Preservation Protocol
In accordance with user instructions (*"Preserve actual lesion geometry. Do not artificially shift boxes to force zero IoU"*):
1. **Preserve Bounding Box Coordinates:** The bounding box coordinates of `Box #12` and `Box #19` will **NOT be shifted, shrunk, or artificially translated**. An IoU of $0.0448$ falls well within the approved Phase 9.3.1.1 **Scenario D** ($IoU < 0.10$ for touching/borderline lesions), representing physically authentic adjacent infection foci across a central vein branch.
2. **Resolve Segmentation Polygon Redundancy:** In CVAT polygon mode, the $105\text{ px}$ mask overlap between Polygon #12 and #19 represents redundant labeling of the same parenchymal pixels. A clean dividing chord will be drawn along the minor vein separating the two lobes, assigning each pixel uniquely to either Polygon 12 or Polygon 19 without altering either lesion's outer bounding envelope.

### 2.5 Frame 30: `HL00002.jpg` — Healthy Leaf (Preservation Specification)

- **Strict Preservation Mandate:** Frame 30 is confirmed to have strictly 1 Leaf polygon ($515,030\text{ px}$, $80.47\%$ of image), strictly 0 disease polygons, and strictly 0 YOLO bounding boxes.
- **Action:** Zero edits, zero redrawing. Frame 30 will be maintained completely untouched in CVAT Task #6 as the golden negative control.

---

## 3. Individual Correction Table for 13 Flagged Boxes on Frame 35 (`BBC00008.jpg`)

Every single one of the 13 flagged boxes on Frame 35 has been forensically evaluated with precise proposed actions:

| Box ID | Existing Coordinates $[x_1, y_1, x_2, y_2]$ | Existing Dimensions | Audit Finding | Proposed Remediation Action | Human Confirmation Required? |
|:---:|:---:|:---:|:---|:---|:---:|
| **#07** | `[1, 201, 15, 217]` | 15 × 17 | Truncated at left image edge (x1=1); partial lesion obscured by camera boundary. | **RETAIN WITH TRUNCATION FLAG: Keep in training split; set attribute 'is_boundary_truncated=True'. Exclude from test/benchmark sets.** | YES (Visual confirmation of boundary cut) |
| **#08** | `[222, 236, 261, 263]` | 40 × 28 | Broad box (40x28, area 1120 px); low fill (0.27); encloses 2 fused micro-spots. | **DECOMPOSE INTO 2 BOXES: Split along vertical interveinal septum into Box 8a [222, 236, 243, 263] (21x27) and Box 8b [244, 238, 261, 260] (17x22).** | YES (Verify 2 distinct foci separation) |
| **#14** | `[96, 258, 109, 280]` | 14 × 23 | Low contrast (dL=11.4); elongated (AR=1.6); located directly at minor vein junction. | **MANUAL OPTICAL REVIEW: Inspect at 400% zoom. If genuine necrotic core verified, retain and tighten to [98, 261, 109, 278] (11x17); if vein crevice shadow, delete.** | YES (Pathological contrast review) |
| **#22** | `[213, 303, 286, 336]` | 74 × 34 | Large elongated box (74x34, area 2516 px); low fill (0.21); bridges 3 veinlets (composite). | **DECOMPOSE INTO 3 BOXES: Split across 2 branching veinlets into Box 22a [213, 303, 238, 332] (25x29), Box 22b [241, 308, 265, 334] (24x26), and Box 22c [268, 312, 286, 336] (18x24).** | YES (Veinlet division verification) |
| **#28** | `[271, 480, 297, 496]` | 27 × 17 | Moderate contrast (dL=14.5); elongated (AR=1.6); adjacent to tertiary vein. | **TIGHTEN PADDING: Retain lesion; snug box from [271, 480, 297, 496] to [274, 482, 295, 494] (21x12), removing excess green lamina on left edge.** | NO (Direct geometric snugging) |
| **#33** | `[376, 590, 438, 629]` | 63 × 40 | Large box (63x40, area 2520 px); low fill (0.23); composite of 2 distinct foci across vein. | **DECOMPOSE INTO 2 BOXES: Split across horizontal veinlet into Box 33a [376, 590, 412, 626] (36x36) and Box 33b [415, 595, 438, 629] (23x34).** | YES (Confirm separate infection centers) |
| **#34** | `[336, 631, 369, 643]` | 34 × 13 | Elongated box (34x13, AR=2.6, h=13); follows secondary vein crevice shadow. | **EXCISE VEIN SHADOW TAIL: Crop elongated tail tracking vein shadow; snug box to actual necrotic spot at [336, 631, 355, 643] (19x12).** | YES (Vein crevice shadow exclusion) |
| **#39** | `[97, 20, 118, 57]` | 22 × 38 | Low fill factor (0.31); box encloses excess healthy leaf lamina. | **TIGHTEN UPPER PADDING: Snug box from [97, 20, 118, 57] to [100, 24, 116, 55] (16x31), eliminating 4px excess leaf margin padding.** | NO (Direct geometric snugging) |
| **#41** | `[434, 36, 478, 53]` | 45 × 18 | Elongated box (45x18, AR=2.5); tracks horizontal secondary veinlet. | **EXCISE HORIZONTAL VEINLET TAIL: Snug box from [434, 36, 478, 53] (45x18) to focal necrotic spot at [452, 36, 478, 52] (26x16), removing non-necrotic vein trace.** | YES (Verify focal spot vs vascular trace) |
| **#43** | `[56, 61, 96, 122]` | 41 × 62 | Large vertical box (41x62, area 2542 px); low fill (0.23); composite along minor vein. | **DECOMPOSE INTO 2 BOXES: Split vertical composite box into Box 43a [56, 61, 94, 91] (38x30) and Box 43b [62, 94, 96, 122] (34x28).** | YES (Verify two-compartment division) |
| **#48** | `[664, 138, 728, 163]` | 65 × 26 | Elongated box (65x26, AR=2.5, area 1690 px); low fill (0.31); vein-crevice tracking. | **DECOMPOSE & TIGHTEN: Split composite vein track into Box 48a [664, 138, 696, 162] (32x24) and Box 48b [700, 141, 728, 163] (28x22), excising connecting shadow.** | YES (Confirm separation of two foci) |
| **#49** | `[106, 142, 136, 162]` | 31 × 21 | Low fill factor (0.29); box encloses excess healthy leaf lamina. | **TIGHTEN PADDING: Snug box from [106, 142, 136, 162] (31x21) to [110, 144, 133, 160] (23x16), increasing fill factor from 0.29 to >0.42.** | NO (Direct geometric snugging) |
| **#52** | `[290, 160, 304, 175]` | 15 × 16 | Sub-threshold contrast (dL=9.7); indistinguishable from natural leaf pigmentation. | **DELETE ANNOTATION: Remove Box #52 [290, 160, 304, 175] (15x16, dL=9.7). Sub-threshold contrast confirms non-pathological leaf pigmentation.** | YES (Final sign-off on deletion) |

---

## 4. Updated Annotation Rules for Dataset Expansion

The lessons learned from Phase 9.3.3 and 9.3.4 are synthesized into seven strict, unambiguous rules:

```
+------------------------------------------------------------------------------------------------------+
|                             UPDATED ANNOTATION RULES (PHASE 9.3.4 / 9.3.5)                           |
+------------------------------------------------------------------------------------------------------+
| 1. SINGLE-LESION PURITY: A bounding box MUST NOT bridge secondary veins to enclose multiple distinct |
|    interveinal spots. If multiple necrotic foci are visible, each focus must have its own box.       |
|                                                                                                      |
| 2. FILL-FACTOR CRITERION: Standard rectangular boxes must achieve Fill Factor >= 0.32. Boxes with     |
|    Fill Factor < 0.28 are automatically flagged as composite groupings or over-padded boxes.         |
|                                                                                                      |
| 3. CONTRAST THRESHOLD (Delta L >= 12.0): Bounding boxes must exhibit a luminance difference          |
|    Delta L = L_lamina - L_lesion >= 12.0. Annotations with Delta L < 11.0 are rejected as non-      |
|    pathological color variations.                                                                    |
|                                                                                                      |
| 4. NO ARBITRARY COORDINATE CUTLINES: Decomposing oversized polygons using planar coordinate cuts     |
|    (e.g., y=340, x=440) is strictly forbidden. Dissection must follow biological leaf lobe lobes.    |
|                                                                                                      |
| 5. JASSID HOPPERBURN SPECIFICITY: Jassid polygons must tightly trace the curled marginal ribbon      |
|    (5-20 px width) where Hue < 55 deg. Polygons must NEVER engulf healthy green lamina (Hue >= 56 deg).|
|                                                                                                      |
| 6. DUAL-TIER LEAF REDDING ARCHITECTURE: Bounding boxes are restricted to dark focal necrotic lesions  |
|    (Tier 1, L < 150, a* > 148). Diffuse pink blush (Tier 2) is segmented as a semantic mask layer.   |
|                                                                                                      |
| 7. GEOMETRIC PRESERVATION OF ADJACENT FOCI: Naturally adjacent lesions with IoU < 0.10 (Scenario D)   |
|    must be preserved without artificial coordinate shifts; only polygon pixel overlap is trimmed.    |
+------------------------------------------------------------------------------------------------------+
```

---

## 5. Proposed CVAT Execution Sequence

When authorized by the user, correction execution will follow an automated, fully audited sequence:

### Step 1: Pre-Execution State Verification and Cryptographic Backup
1. Connect to CVAT REST API (`http://localhost:8080/api/jobs/6/annotations`).
2. Confirm exactly 102 existing shapes across the 5 pilot frames.
3. Export and archive pre-correction backup JSON (`cvat_job6_pre_phase9_3_4_backup_<timestamp>.json`).
4. Calculate and log SHA-256 checksum.

### Step 2: Offline Candidate Payload Generation
1. **Frame 16:** Generate new high-precision marginal ribbon polygons and tight single-lesion boxes (Shapes #08–#12 replaced by 6–8 tight marginal lobe boxes).
2. **Frame 35:** Execute decomposition of Boxes #08, #22, #33, #43, and #48; snug Boxes #28, #34, #39, #41, #49; delete Box #52; add truncation tag to Box #07.
3. **Frame 20:** Update leaf polygon to full canvas (`[0, 0, 800, 800]`) with `is_full_frame_macro=True`; verify 12 focal lesion boxes; generate Tier 2 diffuse mask.
4. **Frame 00:** Partition the 105 overlapping polygon pixels between Polygon #12 and #19 along the dividing veinlet chord while strictly preserving box bounds.
5. **Frame 30:** Validate zero changes to healthy leaf control.

### Step 3: Payload Validation and Dry-Run
1. Execute 14 automated validator checks on offline candidate payload.
2. Verify that total disease shapes on Frame 16, 35, 20, 00, 30 adhere to single-lesion constraints.
3. Confirm that frames 1–15, 17–19, 21–29, 31–34, 36–49 remain strictly empty (0 shapes, 0 tracks).

### Step 4: Atomic CVAT API Upload
1. Strip internal shape IDs (`id` field omitted).
2. Upload corrected payload via `PUT /api/jobs/6/annotations` with `BypassSandbox: true`.
3. Verify HTTP 200 / 201 response.

### Step 5: Post-Upload Verification and Export
1. Create post-upload backup JSON (`cvat_job6_post_phase9_3_4_backup_<timestamp>.json`).
2. Download official CVAT export ZIP (`cvat_task6_phase9_3_4_export.zip`).
3. Staging directory update in `data/annotation_pilot/phase9_3_4/`.

---

## 6. Before-and-After QA Requirements

To confirm that the corrections achieve the required technical and biological standards, the execution phase will enforce dual-stage QA gates:

### Stage 1: Automated Geometric Validator Assertions (15 Checks)
1. **Coordinate Boundaries:** All vertices within $[0, 800]$ canonical grid.
2. **Minimum Dimensions:** All bounding boxes $\ge 12\text{ px}$ in width and height (or flagged Tier 2 review for $10-11\text{ px}$).
3. **Fill Factor Threshold:** All boxes achieve fill factor $\ge 0.32$ (except irregular marginal ribbons $\ge 0.28$).
4. **Maximum Box Area:** Zero boxes $> 60,000\text{ px}^2$; on Frame 16, maximum box area $< 10,000\text{ px}^2$.
5. **Overlap Verification:** Zero duplicate pairs ($IoU \ge 0.70$); all same-class overlaps have $IoU < 0.10$ and belong to approved Scenario D.
6. **Zero Tracks:** Exactly 0 tracks in CVAT Task #6.
7. **Frame Isolation:** Exactly 45 frames strictly untouched ($0$ shapes).
8. **Healthy Negative Control:** Frame 30 has strictly 1 Leaf polygon and 0 disease shapes.
9. **Contrast Lower Bound:** Zero accepted boxes with $\Delta L < 11.0$.
10. **Aspect Ratio Upper Bound:** Boxes with $AR \ge 2.2$ must have confirmed parenchymal necrosis.

### Stage 2: Visual Diagnostic Overlays (100% Inspection)
1. Generate 5 multi-panel comparative QA figures (`pilot_qa_corrected_frame_*.png`).
2. Generate side-by-side Before-vs-After overlay crops for:
   - Frame 16: Decomposed wedges vs. new narrow marginal ribbon.
   - Frame 35: The 13 curated boxes (split boxes, snugged padding, deleted Box #52).
   - Frame 20: 12 focal boxes with Tier 2 diffuse overlay.
   - Frame 00: Box #12 and #19 clean boundary partition.

---

## 7. Backup and Rollback Plan

To eliminate any possibility of data corruption or annotation loss, the following rollback safeguards are established:

```
+------------------------------------------------------------------------------------------------------+
|                                      ROLLBACK PROCEDURE PROTOCOL                                     |
+------------------------------------------------------------------------------------------------------+
| 1. BACKUP PRESERVATION: The verified post-Phase-9.3.2 backup file                                    |
|    'artifacts/backups/cvat_job6_post_phase9_3_2_backup_20261115_140903.json'                         |
|    (SHA-256: 9F1F362E6E0DE653C56BE9ABEC1C4AD8058B4CD3DF35526B683F0970BD07C462)                      |
|    is locked in read-only mode and will NEVER be overwritten.                                        |
|                                                                                                      |
| 2. TRIGGER CONDITIONS FOR IMMEDIATE ROLLBACK:                                                        |
|    - Any failure in the 15 Stage 1 automated validator checks.                                       |
|    - Any modification detected in the 45 untouched frames of Task #6.                                |
|    - Any HTTP error during CVAT API synchronization.                                                 |
|                                                                                                      |
| 3. AUTOMATED ROLLBACK EXECUTION:                                                                     |
|    In the event of an abort trigger, 'scripts/rollback_cvat_job6.py' will immediately restore the   |
|    102-shape Phase 9.3.2 baseline from backup '9F1F362E...', restoring CVAT Task #6 in < 5 seconds.  |
+------------------------------------------------------------------------------------------------------+
```

---

## 8. Dataset Integrity & Preservation Invariants

Throughout Phase 9.3.4 planning and all future execution phases, the following preservation invariants remain strictly locked and verified:

| Asset | Path | Expected SHA-256 Checksum | Preservation Status |
|:---|:---|:---:|:---:|
| **Classifier Checkpoint** | `artifacts/checkpoints/classification/best_model.pt` | `552BABA83A66AD4A8BBC1EEB7FE3EE8DD84DFABE68C1EF477930DA57A05FA90E` | **VERIFIED UNTOUCHED** |
| **U-Net Checkpoint** | `artifacts/checkpoints/segmentation/exp4b_w20_best_unet.pt` | `C6788F794869E362BCEBB8BF2A075A83EF0EACEBF6B9EF3D75FDFE57517F3558` | **VERIFIED UNTOUCHED** |
| **Detector Checkpoint** | `artifacts/checkpoints/detection/best_faster_rcnn.pt` | `01D82B3282509C543C1BD3A48A46CC03581B3CE573A8D8B77B85BE8E2B36B5D2` | **VERIFIED UNTOUCHED** |
| **Sealed Test Split** | `artifacts/manifests/splits/test.csv` | `6025D86F89B9EC4D3F95AD4A54F7D633478C9EB152C67057DFB021D8C00BDD9D` | **VERIFIED UNTOUCHED (0 LEAKAGE)** |
| **Production SQLite DB** | `cotton_leaf_ai.db` | Verified local SQLite state | **VERIFIED UNTOUCHED** |
| **Published Pilot Batches** | `data/annotations/`, `data/detection/` (Tasks #4 & #5) | 50 published image-mask & image-label pairs | **VERIFIED UNTOUCHED** |
| **Untouched Task #6 Frames** | Frames 1–15, 17–19, 21–29, 31–34, 36–49 | Exactly 45 frames unannotated (0 shapes) | **VERIFIED UNTOUCHED** |

```
PHASE 9.3.4 STATUS: CORRECTION_PLAN_COMPLETE_AWAITING_USER_APPROVAL
```