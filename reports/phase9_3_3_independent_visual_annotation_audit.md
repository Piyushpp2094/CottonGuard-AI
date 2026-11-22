# CottonGuard AI — Phase 9.3.3: Independent Visual Annotation Quality Audit
**Phase:** 9.3.3 — Independent Visual & Coordinate Annotation Quality Audit  
**Target Environment:** CVAT Task #6 (`CottonGuard_Pilot_Batch_03_ShapesOnly`, Job #6)  
**Audit Scope:** Full Second-Level Forensic and Botanical Audit of the 5-Image Annotation Pilot  
**Audit Type:** Strictly Read-Only Investigation (Zero CVAT/Dataset Modifications)  
**Date:** November 15, 2026  
**Master Report Deliverable:** `reports/phase9_3_3_independent_visual_annotation_audit.md`  

---

## 1. Executive Summary

An unsparing, second-level forensic and botanical quality audit was conducted on the five-image Phase 9.3.2 annotation pilot in CVAT Task #6. While Phase 9.3.2 succeeded mechanically in converting multi-spot human groupings into single-lesion annotations and passing 14/14 automated geometric validator checks, this independent visual and coordinate audit reveals critical botanical discrepancies that must be rectified prior to expanding the dataset to the remaining 45 Task #6 images or training future detector models.

### Summary of Audit Verdicts by Image

| Frame ID | Image Filename | Disease Class | Total Boxes / Shapes | Primary Audit Finding | Decision Label |
|:---:|:---:|:---:|:---:|:---|:---:|
| **Frame 30** | `HL00002.jpg` | Healthy Leaf | 1 Leaf, 0 Disease | Pristine negative control. Zero disease false alarms, zero background in leaf mask. | **`APPROVED`** |
| **Frame 00** | `BBC00001.jpg` | Bacterial Blight | 1 Leaf, 19 Disease | High-quality single-lesion boxes. 1 touching pair (Box #12 & #19, IoU ~0.038) along central vein. | **`APPROVED WITH CORRECTIONS`** |
| **Frame 20** | `LR00001.jpg` | Leaf Redding | 1 Leaf, 12 Disease | Artificial 4-point leaf rectangle; 12 dark focal patches well-isolated, but 48.5% diffuse pink blush left unannotated. | **`APPROVED WITH CORRECTIONS`** |
| **Frame 35** | `BBC00008.jpg` | Bacterial Blight | 1 Leaf, 54 Disease | 41 confirmed lesions; 11 flagged for review (4 composite multi-veinlet boxes, 3 vein-crevice tracks); 1 non-lesion (Box #52); 1 truncated (Box #7). | **`APPROVED WITH CORRECTIONS`** |
| **Frame 16** | `LHJ00059.jpg` | Leaf Hopper Jassids | 1 Leaf, 12 Disease | **Critical failure:** 5 decomposed wedges contain >80% healthy green lamina; arbitrary coordinate slicing; inflated 24.53% severity. | **`REJECTED`** |

> [!IMPORTANT]
> **Overall Phase 9.3.3 Governance Recommendation:** **`APPROVED WITH CORRECTIONS (ACTION REQUIRED BEFORE EXPANSION)`**  
> Frame 16 (`LHJ00059.jpg`) is formally **REJECTED** and must be re-annotated botanically. Frame 35 (`BBC00008.jpg`) requires resolution of 13 flagged boxes. Frame 20 (`LR00001.jpg`) requires adoption of a clear Dual-Tier Policy for Leaf Redding before proceeding to Phase 9.3.4 (Dataset Expansion). Zero modifications were applied during this read-only phase.

---

## 2. Image-by-Image Forensic Findings

### 2.1 Frame 16: `LHJ00059.jpg` — Leaf Hopper Jassids (`REJECTED`)

Frame 16 represents the most severe discrepancy in the pilot. Although Phase 9.3.2 subdivided two legacy SAM segmentation shapes (`mask5` and `mask0`) to pass the `< 60,000 px²` bounding box limit, forensic inspection reveals that the underlying segmentation is fundamentally ungrounded in plant pathology:

1. **Purple Disease Polygons Cover Healthy Green Lamina:**
   - Quantitative colorimetry demonstrates that the pixel distribution inside the five decomposed wedges (`Shape #08`, `#09`, `#10`, `#11`, `#12`) is indistinguishable from the healthy central leaf blade:
     * **Healthy Central Lamina:** Mean RGB = `(119.6, 168.7, 128.8)`, Lab $L=165.2, a^*=103.5, b^*=143.6$, HSV Hue = $65.3^\circ$.
     * **Shape #08 (22,840 px):** Mean RGB = `(131.9, 173.1, 133.2)`, Lab $L=170.7, a^*=106.1, b^*=144.2$, HSV Hue = $61.2^\circ$.
     * **Shape #09 (18,044 px):** Mean RGB = `(122.7, 167.3, 126.8)`, Lab $L=164.6, a^*=104.7, b^*=144.3$, HSV Hue = $63.0^\circ$.
     * **Shape #11 (12,786 px):** Mean RGB = `(110.3, 163.4, 114.9)`, Lab $L=159.3, a^*=100.6, b^*=147.7$, HSV Hue = $63.4^\circ$.
     * **Shape #12 (18,109 px):** Mean RGB = `(110.3, 164.3, 112.7)`, Lab $L=159.8, a^*=99.7, b^*=149.3$, HSV Hue = $62.5^\circ$.
   - Across all these shapes, $\Delta E < 6.0$ relative to healthy green tissue. Over **88.5% ($81,886\text{ px}$)** of the purported disease mask on Frame 16 covers normal, photosynthetic green lamina.

2. **Bounding Boxes Do Not Contain Genuine Individual Lesions:**
   - The resulting bounding boxes are massive rectangular hulls spanning broad sectors of the leaf margin:
     * `Box #08`: $277 \times 164\text{ px}$ ($45,428\text{ px}^2$), enclosing $14,219\text{ px}$ ($31.3%$) of healthy leaf and $8,358\text{ px}$ ($18.4%$) of empty background.
     * `Box #12`: $212 \times 199\text{ px}$ ($42,188\text{ px}^2$), enclosing $15,694\text{ px}$ ($37.2%$) of healthy leaf and $8,353\text{ px}$ ($19.8%$) of empty background.
     * `Box #09`: $204 \times 150\text{ px}$ ($30,600\text{ px}^2$), enclosing $6,303\text{ px}$ ($20.6%$) of healthy leaf and $6,242\text{ px}$ ($20.4%$) of empty background.
     * `Box #11`: $192 \times 145\text{ px}$ ($27,840\text{ px}^2$), enclosing $9,855\text{ px}$ ($35.4%$) of healthy leaf and $5,178\text{ px}$ ($18.6%$) of empty background.
     * `Box #10`: $138 \times 129\text{ px}$ ($17,802\text{ px}^2$), enclosing $7,370\text{ px}$ ($41.4%$) of empty background.

3. **Excessive Vein and Background Inclusion:**
   - The bounding boxes cut straight across primary and secondary leaf veins, forcing the Faster R-CNN detector to associate prominent central veins with jassid symptoms.

4. **Severity Calculation Review ($24.53\%$ Is Unsound):**
   - The reported severity of $24.53\%$ ($92,490\text{ px}$ disease on $377,013\text{ px}$ leaf) is an artifact of the unconstrained SAM segmentation mask. True botanical hopperburn on this leaf is restricted to the extreme curled margin (estimated at $< 15,000\text{ px}$, or $< 4.0\%$ severity).

5. **Biological Invalidation of Geometric Slicing:**
   - Slicing `mask5` at `y=340` and `y=490` and `mask0` at `x=440` was an engineering heuristic to bypass validator size gates. It has **no botanical meaning** and creates artificial planar boundaries that confuse convolutional feature extractors.

### 2.2 Frame 35: `BBC00008.jpg` — Bacterial Blight (`APPROVED WITH CORRECTIONS`)

Frame 35 represents a heavily infected validation image with 54 automated disease annotations. A comprehensive audit of all 54 boxes classifies them into four strict categories:

- **Confirmed Lesions:** 41 boxes (75.9%) — Authentic, angular, dark necrotic lesions bounded by minor veins with high contrast ($\Delta L > 16.0$) and high fill factors.
- **Possible Lesions (Manual Review):** 11 boxes (20.4%) — Require human curation before publication:
  * *Composite multi-veinlet groupings:* `Box #22` ($74 \times 34$, Area 2516 px, Fill 0.21), `Box #33` ($63 \times 40$, Area 2520 px, Fill 0.23), `Box #43` ($41 \times 62$, Area 2542 px, Fill 0.23), `Box #08` ($40 \times 28$, Area 1120 px, Fill 0.27). These four boxes enclose multiple distinct necrotic spots across veinlets, directly violating the Single-Lesion Standard.
  * *Vein-crevice tracking:* `Box #34` ($34 \times 13$, AR 2.62), `Box #41` ($45 \times 18$, AR 2.50), `Box #48` ($65 \times 26$, AR 2.50). These elongated boxes track secondary vein shadows or linear vascular necrosis.
  * *Low contrast / veinlet junctions:* `Box #14` ($14 \times 23$, $\Delta L = 11.4$), `Box #28` ($27 \times 17$, $\Delta L = 14.5$), `Box #39` ($22 \times 38$, Fill 0.31), `Box #49` ($31 \times 21$, Fill 0.29).
- **Likely Non-Lesion:** 1 box (1.9%) — `Box #52` ($[290, 160, 304, 175]$, $15 \times 16\text{ px}$, $\Delta L = 9.7$). The spot has sub-threshold contrast indistinguishable from normal lamina color variations.
- **Insufficient Image Evidence:** 1 box (1.9%) — `Box #07` ($[1, 201, 15, 217]$, $15 \times 17\text{ px}$). Truncated right at the left frame perimeter ($x_1=1$), cutting off the lesion boundary.

### 2.3 Frame 20: `LR00001.jpg` — Leaf Redding (`APPROVED WITH CORRECTIONS`)

Frame 20 presents a macro close-up of Leaf Redding with 12 annotated disease shapes:

1. **Leaf Polygon Integrity:**
   - The leaf polygon is an artificial 4-point rectangle: `4.00,4.00; 796.00,4.00; 796.00,796.00; 4.00,796.00` ($628,889\text{ px}$, $98.3\%$ of canvas).
   - Forensic inspection of all four corners confirms that leaf tissue occupies the full frame (Mean RGB top-left `(137, 170, 114)`, top-right `(150, 115, 87)`, bottom-left `(135, 171, 121)`, bottom-right `(170, 142, 110)`). However, treating an unsegmented rectangular canvas as a leaf polygon is inconsistent with the polygon segmentation standard.

2. **Focal Lesions vs. Diffuse Blush:**
   - The 12 annotated boxes follow the tightened threshold ($L < 150, a^* > 148$), capturing dark, intense focal anthocyanin/necrotic lesions (mean Lab $L \approx 141, a^* \approx 183$, RGB $\approx [220, 75, 125]$).
   - **Critical Omission:** Beyond these 12 focal patches, **310,308 pixels (48.49% of the leaf)** exhibit pronounced pink/red anthocyanin blush ($a^* > 135$). An additional **9,315 pixels** of dark focal tissue were discarded by morphological filters.
   - In object detection training, the RPN samples unannotated pink regions as negative background anchors. Penalizing the detector for predicting reddening on intense pink blush causes high false negative rates. A formal Dual-Tier Annotation Policy is required.

### 2.4 Frame 00: `BBC00001.jpg` — Bacterial Blight (`APPROVED WITH CORRECTIONS`)

- All 19 disease boxes adhere to the Single-Lesion Standard with high fill factors (mean 0.72) and robust contrast ($\Delta L = 13.0 - 38.7$).
- **Vein Proximity & Touching Pair:** `Box #12` ($[519, 334, 567, 369]$, $49 \times 36\text{ px}$) and `Box #19` ($[488, 314, 529, 347]$, $42 \times 34\text{ px}$) form a touching/overlapping pair (IoU 0.038) directly straddling the main central vein. They require minor coordinate adjustment to prevent merged anchor proposals.
- No small lesion violations; minimum box width is $13\text{ px}$ (`Box #03` and `Box #18`).

### 2.5 Frame 30: `HL00002.jpg` — Healthy Leaf (`APPROVED`)

- **Exemplary Negative Control:** Strictly 1 Leaf polygon (74 vertices, $515,030\text{ px}$, $80.47\%$ of image), strictly 0 disease polygons, 0 YOLO boxes (0-byte file).
- **Mask Cleanliness:** Only 200 pixels (<0.04%) inside the leaf mask have gray intensity $< 30$ (shadow clefts at petiole insertion). Zero tabletop/background inclusion.
- Confirms that veins, leaf hairs, and surface reflections do not trigger false-positive annotations.

---

## 3. Questionable Annotation IDs and Coordinates

The following table details every questionable annotation identified across the five pilot frames:

| Frame ID | Image Filename | Shape / Box ID | Coordinates $[x_1, y_1, x_2, y_2]$ | Dimensions ($w \times h$) | Area ($px^2$) | Specific Flaw / Evidence | Recommended Correction |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|
| **Frame 16** | `LHJ00059.jpg` | Shape #08 | `[57, 176, 333, 339]` | $277 \times 164$ | 45,428 | SAM mask captures healthy green lamina (Hue 61.2°, ΔE<6); box is 31.3% healthy leaf, 18.4% bg. | **REJECT / REDRAW:** Tightly bound actual chlorotic margin only. |
| **Frame 16** | `LHJ00059.jpg` | Shape #09 | `[52, 340, 255, 489]` | $204 \times 150$ | 30,600 | Arbitrary horizontal slice cut at y=340; 20.6% healthy lamina, 20.4% bg inside box. | **REJECT / REDRAW:** Delete artificial slice boundary. |
| **Frame 16** | `LHJ00059.jpg` | Shape #10 | `[61, 490, 198, 618]` | $138 \times 129$ | 17,802 | Arbitrary slice cut at y=490; box contains 41.4% empty background outside leaf. | **REJECT / REDRAW:** Retrace narrow marginal symptom band. |
| **Frame 16** | `LHJ00059.jpg` | Shape #11 | `[248, 640, 439, 784]` | $192 \times 145$ | 27,840 | Arbitrary vertical cut at x=440; 35.4% healthy lamina, 18.6% bg; traverses primary veins. | **REJECT / REDRAW:** Re-annotate symptomatic margin only. |
| **Frame 16** | `LHJ00059.jpg` | Shape #12 | `[440, 584, 651, 782]` | $212 \times 199$ | 42,188 | Arbitrary cut at x=440; 37.2% healthy leaf, 19.8% bg; captures healthy green lamina (Hue 62.5°). | **REJECT / REDRAW:** Restrict to curled/chlorotic margin. |
| **Frame 35** | `BBC00008.jpg` | Box #07 | `[1, 201, 15, 217]` | $15 \times 17$ | 255 | Truncated at left image border ($x_1=1$); incomplete lesion feature evidence. | **FLAG (Insufficient Evidence):** Exclude from evaluation test sets. |
| **Frame 35** | `BBC00008.jpg` | Box #08 | `[222, 236, 261, 263]` | $40 \times 28$ | 1,120 | Low fill factor (0.27); composite box spanning two distinct micro-foci. | **DECOMPOSE:** Split into two single-lesion boxes. |
| **Frame 35** | `BBC00008.jpg` | Box #14 | `[96, 258, 109, 280]` | $14 \times 23$ | 322 | Weak contrast ($\Delta L = 11.4$); positioned on minor vein junction. | **MANUAL REVIEW:** Validate lesion viability at full optical zoom. |
| **Frame 35** | `BBC00008.jpg` | Box #22 | `[213, 303, 286, 336]` | $74 \times 34$ | 2,516 | Low fill factor (0.21); bridges 3 distinct veinlets (composite grouping violation). | **DECOMPOSE:** Split into 2 or 3 single-lesion boxes. |
| **Frame 35** | `BBC00008.jpg` | Box #28 | `[271, 480, 297, 496]` | $27 \times 17$ | 459 | Moderate contrast ($\Delta L = 14.5$), elongated along tertiary veinlet. | **MANUAL REVIEW:** Inspect under cross-polarized magnification. |
| **Frame 35** | `BBC00008.jpg` | Box #33 | `[376, 590, 438, 629]` | $63 \times 40$ | 2,520 | Low fill factor (0.23); composite box enclosing two separate necrotic foci across vein. | **DECOMPOSE:** Split into two independent single-lesion boxes. |
| **Frame 35** | `BBC00008.jpg` | Box #34 | `[336, 631, 369, 643]` | $34 \times 13$ | 442 | Highly elongated (AR 2.62, $h=13$); follows secondary vein crevice shadow. | **MANUAL REVIEW / ADJUST:** Trim vein shadow or eliminate. |
| **Frame 35** | `BBC00008.jpg` | Box #39 | `[97, 20, 118, 57]` | $22 \times 38$ | 836 | Low fill factor (0.31); encloses excess healthy leaf lamina along upper margin. | **TIGHTEN:** Trim 2px padding to strictly encompass necrotic core. |
| **Frame 35** | `BBC00008.jpg` | Box #41 | `[434, 36, 478, 53]` | $45 \times 18$ | 810 | Highly elongated (AR 2.50); tracks horizontal secondary veinlet. | **MANUAL REVIEW:** Confirm necrotic parenchymal involvement. |
| **Frame 35** | `BBC00008.jpg` | Box #43 | `[56, 61, 96, 122]` | $41 \times 62$ | 2,542 | Low fill factor (0.23); composite vertical group spanning multiple vein compartments. | **DECOMPOSE:** Split into 2 single-lesion boxes. |
| **Frame 35** | `BBC00008.jpg` | Box #48 | `[664, 138, 728, 163]` | $65 \times 26$ | 1,690 | Elongated (AR 2.50), low fill (0.31); tracks vein crevice near right margin. | **MANUAL REVIEW / SPLIT:** Isolate distinct necrotic foci. |
| **Frame 35** | `BBC00008.jpg` | Box #49 | `[106, 142, 136, 162]` | $31 \times 21$ | 651 | Low fill factor (0.29); loose bounding box enclosing excess green lamina. | **TIGHTEN:** Snug box coordinates to lesion boundary. |
| **Frame 35** | `BBC00008.jpg` | Box #52 | `[290, 160, 304, 175]` | $15 \times 16$ | 240 | Sub-threshold contrast ($\Delta L = 9.7$); indistinguishable from background pigmentation. | **DELETE (Likely Non-Lesion):** Remove non-pathological artifact. |
| **Frame 20** | `LR00001.jpg` | Leaf Poly | `[4, 4, 796, 796]` | $793 \times 793$ | 628,889 | Artificial 4-corner rectangle rather than a botanical leaf contour. | **STANDARDIZE:** Document macro full-frame policy or trace margins. |
| **Frame 00** | `BBC00001.jpg` | Box #12 & #19 | `[519, 334]` / `[488, 314]` | $49 \times 36$ / $42 \times 34$ | 1,764 / 1,428 | Touching pair (IoU 0.038) along central vein; mutual proximity may cause proposal merge. | **ADJUST:** Ensure 1-2px separation between bounding hulls. |

---

## 4. Frame 35 (`BBC00008.jpg`) — Complete 54-Box Classification Audit

Every single one of the 54 automated disease annotations on Frame 35 was independently audited:

| Box ID | Coordinates $[x_1, y_1, x_2, y_2]$ | Dim ($w \times h$) | Area ($px^2$) | Aspect Ratio | Fill Factor | $\Delta L$ Contrast | Assigned Category | Forensic Audit Rationale |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **#01** | `[258, 183, 278, 201]` | 21 × 19 | 399 | 1.11 | 0.47 | 19.6 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#02** | `[320, 190, 348, 214]` | 29 × 25 | 725 | 1.16 | 0.38 | 24.2 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#03** | `[243, 193, 255, 216]` | 13 × 24 | 312 | 1.85 | 0.54 | 19.5 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#04** | `[300, 193, 318, 207]` | 19 × 15 | 285 | 1.27 | 0.51 | 19.2 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#05** | `[46, 194, 71, 216]` | 26 × 23 | 598 | 1.13 | 0.42 | 20.6 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#06** | `[505, 198, 521, 218]` | 17 × 21 | 357 | 1.24 | 0.42 | 18.7 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#07** | `[1, 201, 15, 217]` | 15 × 17 | 255 | 1.13 | 0.58 | 15.1 | **`INSUFFICIENT IMAGE EVIDENCE`** | Truncated at left image edge (x1=1); partial lesion obscured by camera boundary. |
| **#08** | `[222, 236, 261, 263]` | 40 × 28 | 1120 | 1.43 | 0.27 | 21.3 | **`POSSIBLE LESION – MANUAL REVIEW`** | Broad box (40x28, area 1120 px); low fill (0.27); encloses 2 fused micro-spots. |
| **#09** | `[253, 163, 267, 181]` | 15 × 19 | 285 | 1.27 | 0.55 | 20.3 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#10** | `[137, 335, 151, 352]` | 15 × 18 | 270 | 1.20 | 0.56 | 26.4 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#11** | `[306, 339, 323, 353]` | 18 × 15 | 270 | 1.20 | 0.56 | 23.4 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#12** | `[337, 343, 365, 357]` | 29 × 15 | 435 | 1.93 | 0.54 | 27.8 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#13** | `[258, 243, 269, 263]` | 12 × 21 | 252 | 1.75 | 0.58 | 16.8 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#14** | `[96, 258, 109, 280]` | 14 × 23 | 322 | 1.64 | 0.50 | 11.4 | **`POSSIBLE LESION – MANUAL REVIEW`** | Low contrast (dL=11.4); elongated (AR=1.6); located directly at minor vein junction. |
| **#15** | `[198, 265, 219, 290]` | 22 × 26 | 572 | 1.18 | 0.41 | 17.7 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#16** | `[115, 268, 131, 285]` | 17 × 18 | 306 | 1.06 | 0.48 | 16.2 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#17** | `[306, 272, 333, 294]` | 28 × 23 | 644 | 1.22 | 0.58 | 52.6 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#18** | `[239, 275, 281, 302]` | 43 × 28 | 1204 | 1.54 | 0.36 | 20.6 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#19** | `[287, 275, 313, 305]` | 27 × 31 | 837 | 1.15 | 0.34 | 18.9 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#20** | `[310, 355, 329, 377]` | 20 × 23 | 460 | 1.15 | 0.48 | 24.6 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#21** | `[747, 376, 768, 388]` | 22 × 13 | 286 | 1.69 | 0.62 | 31.0 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#22** | `[213, 303, 286, 336]` | 74 × 34 | 2516 | 2.18 | 0.21 | 23.9 | **`POSSIBLE LESION – MANUAL REVIEW`** | Large elongated box (74x34, area 2516 px); low fill (0.21); bridges 3 veinlets (composite). |
| **#23** | `[298, 312, 311, 338]` | 14 × 27 | 378 | 1.93 | 0.52 | 22.4 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#24** | `[543, 385, 559, 398]` | 17 × 14 | 238 | 1.21 | 0.65 | 46.1 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#25** | `[680, 405, 698, 418]` | 19 × 14 | 266 | 1.36 | 0.71 | 42.0 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#26** | `[125, 458, 141, 472]` | 17 × 15 | 255 | 1.13 | 0.71 | 36.3 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#27** | `[299, 474, 318, 490]` | 20 × 17 | 340 | 1.18 | 0.48 | 23.1 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#28** | `[271, 480, 297, 496]` | 27 × 17 | 459 | 1.59 | 0.46 | 14.5 | **`POSSIBLE LESION – MANUAL REVIEW`** | Moderate contrast (dL=14.5); elongated (AR=1.6); adjacent to tertiary vein. |
| **#29** | `[175, 543, 189, 555]` | 15 × 13 | 195 | 1.15 | 0.74 | 30.3 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#30** | `[356, 553, 392, 573]` | 37 × 21 | 777 | 1.76 | 0.50 | 31.9 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#31** | `[367, 577, 380, 608]` | 14 × 32 | 448 | 2.29 | 0.54 | 24.7 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#32** | `[192, 588, 206, 603]` | 15 × 16 | 240 | 1.07 | 0.65 | 21.4 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#33** | `[376, 590, 438, 629]` | 63 × 40 | 2520 | 1.57 | 0.23 | 30.5 | **`POSSIBLE LESION – MANUAL REVIEW`** | Large box (63x40, area 2520 px); low fill (0.23); composite of 2 distinct foci across vein. |
| **#34** | `[336, 631, 369, 643]` | 34 × 13 | 442 | 2.62 | 0.50 | 15.7 | **`POSSIBLE LESION – MANUAL REVIEW`** | Elongated box (34x13, AR=2.6, h=13); follows secondary vein crevice shadow. |
| **#35** | `[168, 647, 185, 666]` | 18 × 20 | 360 | 1.11 | 0.68 | 42.6 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#36** | `[581, 665, 597, 681]` | 17 × 17 | 289 | 1.00 | 0.74 | 24.1 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#37** | `[483, 672, 502, 690]` | 20 × 19 | 380 | 1.05 | 0.80 | 34.9 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#38** | `[502, 736, 523, 756]` | 22 × 21 | 462 | 1.05 | 0.65 | 38.6 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#39** | `[97, 20, 118, 57]` | 22 × 38 | 836 | 1.73 | 0.31 | 20.5 | **`POSSIBLE LESION – MANUAL REVIEW`** | Low fill factor (0.31); box encloses excess healthy leaf lamina. |
| **#40** | `[213, 26, 225, 52]` | 13 × 27 | 351 | 2.08 | 0.52 | 17.4 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#41** | `[434, 36, 478, 53]` | 45 × 18 | 810 | 2.50 | 0.43 | 21.8 | **`POSSIBLE LESION – MANUAL REVIEW`** | Elongated box (45x18, AR=2.5); tracks horizontal secondary veinlet. |
| **#42** | `[223, 55, 234, 75]` | 12 × 21 | 252 | 1.75 | 0.59 | 25.4 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#43** | `[56, 61, 96, 122]` | 41 × 62 | 2542 | 1.51 | 0.23 | 23.3 | **`POSSIBLE LESION – MANUAL REVIEW`** | Large vertical box (41x62, area 2542 px); low fill (0.23); composite along minor vein. |
| **#44** | `[121, 64, 139, 93]` | 19 × 30 | 570 | 1.58 | 0.42 | 20.0 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#45** | `[608, 73, 628, 90]` | 21 × 18 | 378 | 1.17 | 0.39 | 16.0 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#46** | `[216, 94, 231, 125]` | 16 × 32 | 512 | 2.00 | 0.47 | 22.4 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#47** | `[29, 139, 53, 167]` | 25 × 29 | 725 | 1.16 | 0.37 | 16.9 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#48** | `[664, 138, 728, 163]` | 65 × 26 | 1690 | 2.50 | 0.31 | 34.0 | **`POSSIBLE LESION – MANUAL REVIEW`** | Elongated box (65x26, AR=2.5, area 1690 px); low fill (0.31); vein-crevice tracking. |
| **#49** | `[106, 142, 136, 162]` | 31 × 21 | 651 | 1.48 | 0.29 | 16.1 | **`POSSIBLE LESION – MANUAL REVIEW`** | Low fill factor (0.29); box encloses excess healthy leaf lamina. |
| **#50** | `[333, 142, 353, 158]` | 21 × 17 | 357 | 1.24 | 0.53 | 16.7 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#51** | `[230, 146, 241, 165]` | 12 × 20 | 240 | 1.67 | 0.62 | 23.6 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#52** | `[290, 160, 304, 175]` | 15 × 16 | 240 | 1.07 | 0.70 | 9.7 | **`LIKELY NON-LESION`** | Sub-threshold contrast (dL=9.7); indistinguishable from natural leaf pigmentation. |
| **#53** | `[259, 324, 295, 358]` | 37 × 35 | 1295 | 1.06 | 0.34 | 24.8 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |
| **#54** | `[231, 168, 247, 186]` | 17 × 19 | 323 | 1.12 | 0.51 | 22.8 | **`CONFIRMED LESION`** | Angular water-soaked/necrotic spot bounded by minor veins; high contrast. |

---

## 5. Severity Calculation Review

A rigorous review of the disease severity metrics ($S = \frac{A_{\text{disease}}}{A_{\text{leaf}}} \times 100\%$) reveals varying degrees of biological fidelity across the five frames:

| Frame ID | Image Filename | Disease Class | Leaf Area ($px$) | Staged Disease Area ($px$) | Staged Severity ($S\%$) | Audited Biological Validity | True Estimated Biological Severity |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| **Frame 30** | `HL00002.jpg` | Healthy Leaf | 515,030 | 0 | **0.0000%** | **Valid:** Zero disease pixels; true negative control. | **0.00%** |
| **Frame 00** | `BBC00001.jpg` | Bacterial Blight | 605,221 | 6,477 | **1.0702%** | **Valid:** Tight individual angular spots accurately captured. | **1.05% – 1.10%** |
| **Frame 35** | `BBC00008.jpg` | Bacterial Blight | 603,722 | 13,138 | **2.1762%** | **Slightly Overstated:** Includes 4 composite boxes, vein shadows, and 1 non-lesion. | **1.75% – 1.95%** |
| **Frame 20** | `LR00001.jpg` | Leaf Redding | 628,889 | 8,037 | **1.2780%** | **Severely Understated:** Captures only 12 focal dark spots; excludes 310,308 px of pink blush. | **45.0% – 50.0% (Overall Leaf)** |
| **Frame 16** | `LHJ00059.jpg` | Leaf Hopper Jassids | 377,013 | 92,490 | **24.5255%** | **Grossly Overstated (5x-6x):** SAM mask swamped >80,000 px of normal green lamina. | **3.0% – 5.0% (True Hopperburn)** |

> [!WARNING]
> **Impact on Downstream U-Net Severity Regression:**  
> If the U-Net severity branch trains on Frame 16 with a ground-truth label of $24.53\%$, the network will learn to predict catastrophic jassid defoliation on mild marginal curling, leading to massive false-positive severity alarms in field deployment. Severity ground truths must be re-baselined to reflect authentic chlorotic/necrotic tissue.

---

## 6. Recommended Corrections

To ensure that Task #6 provides a gold-standard benchmark for the Faster R-CNN detector and U-Net segmenter, the following corrections are recommended:

### Correction 1: Frame 16 Botanical Re-Annotation (`LHJ00059.jpg`)
1. **Discard the 5 Decomposed Wedges:** Remove Shapes #08, #09, #10, #11, and #12.
2. **Retrace Marginal Symptom Ribbon:** Re-annotate the leaf margin using a narrow polygon (5–15 px width) that strictly traces the curled, chlorotic edge where $H < 35^\circ$ and $a^* > 115$.
3. **Single-Lesion Bounding Boxes:** Draw tight bounding boxes (maximum dimension $< 80\text{ px}$) around distinct marginal chlorotic sectors or curl folds rather than enclosing half the leaf.
4. **Recalculate Severity:** Baseline the true disease area to $\approx 12,000 - 16,000\text{ px}$ ($S \approx 3.2 - 4.2\%$).

### Correction 2: Frame 35 Annotation Refinement (`BBC00008.jpg`)
1. **Decompose 4 Composite Boxes:** Split Box #22 (into 3 boxes), Box #33 (into 2 boxes), Box #43 (into 2 boxes), and Box #08 (into 2 boxes) so that each box bounds a single interveinal necrotic focus.
2. **Delete Non-Lesion Box #52:** Remove Box #52 ($[290, 160, 304, 175]$) due to sub-threshold contrast ($\Delta L = 9.7$).
3. **Resolve Vein-Crevice Tracks:** Inspect and snug Box #34, #41, and #48 to exclude pure vein shadow crevices.
4. **Maintain Truncation Flag on Box #07:** Retain Box #07 in training, but flag it as truncated for validation.

### Correction 3: Frame 20 Policy Formalization (`LR00001.jpg`)
1. **Dual-Tier Annotation:** Formally distinguish **Focal Anthocyanin Lesions** (the 12 annotated boxes) from **Diffuse Lamina Blush** (the 310k px pink region).
2. **Leaf Polygon Definition:** Explicitly codify that in macro close-up frames where the leaf blade exceeds the camera field-of-view, a 4-corner boundary is permitted but must carry an attribute `is_full_frame_macro=True`.

### Correction 4: Frame 00 Separation (`BBC00001.jpg`)
1. Adjust Box #12 and #19 by 1–2 pixels along their shared boundary to ensure zero bounding box overlap.

---

## 7. Updated Annotation Rules for Dataset Expansion

To prevent similar discrepancies when annotating the remaining 45 images of Task #6, the following rules are added to the CottonGuard AI Annotation Standard:

```
+----------------------------------------------------------------------------------------------------+
|                            UPDATED ANNOTATION PROTOCOL FOR PHASE 9.3.4                             |
+----------------------------------------------------------------------------------------------------+
| 1. SINGLE-LESION PURITY: A bounding box MUST NOT enclose multiple parenchymal lesions separated  |
|    by primary or secondary veins. Boxes with fill factor < 0.28 are strictly rejected.           |
|                                                                                                    |
| 2. VEIN-SHADOW REJECTION: Dark linear depressions along primary/secondary veins are natural leaf   |
|    contours, not lesions. Boxes with Aspect Ratio >= 2.0 must be verified for parenchymal necrosis.|
|                                                                                                    |
| 3. CONTRAST LOWER BOUND: Annotations with Delta L < 12.0 relative to adjacent lamina are rejected |
|    as non-pathological pigmentation variations (unless supported by unambiguous chlorotic halos). |
|                                                                                                    |
| 4. NO ARBITRARY COORDINATE SLICING: Slicing oversized SAM masks with planar horizontal/vertical    |
|    cutlines is forbidden. Large marginal symptom bands must be re-segmented into biological lobes. |
|                                                                                                    |
| 5. JASSID HOPPERBURN SPECIFICITY: Jassid polygons must follow the chlorotic/curled marginal ribbon  |
|    (5-15 px width) and must NOT engulf green central leaf lamina (Hue 36-85 deg).                  |
|                                                                                                    |
| 6. DUAL-TIER LEAF REDDING: Bounding boxes are reserved for distinct focal necrotic lesions. Diffuse|
|    interveinal anthocyanin blush is annotated as a semantic layer or lobe-level classification.     |
+----------------------------------------------------------------------------------------------------+
```

---

## 8. Final Recommendation and Governance Gate

### Decision Accounting

- **Frame 30 (`HL00002.jpg`):** **`APPROVED`** (Ready for expansion benchmark).
- **Frame 00 (`BBC00001.jpg`):** **`APPROVED WITH CORRECTIONS`** (Minor overlap tweak).
- **Frame 20 (`LR00001.jpg`):** **`APPROVED WITH CORRECTIONS`** (Adopt Dual-Tier policy).
- **Frame 35 (`BBC00008.jpg`):** **`APPROVED WITH CORRECTIONS`** (Curate 13 flagged boxes).
- **Frame 16 (`LHJ00059.jpg`):** **`REJECTED`** (Must be redrawn botanically).

### Overall Suite Recommendation
**`APPROVED WITH CORRECTIONS — HOLD EXPANSION PENDING USER APPROVAL`**

Under strict Phase 9.3.3 safety restrictions, **no annotations were altered, no CVAT shapes were modified, and no models were trained**.

The engineering team awaits explicit user authorization on whether to:
1. Apply the recommended corrections to Frames 16, 35, 20, and 00 in CVAT Task #6.
2. Or proceed directly to Phase 9.3.4 dataset expansion using the updated 6-rule protocol.

```
FINAL GATE STATUS: PHASE_9_3_3_AUDIT_COMPLETE_HOLD_PENDING_USER_APPROVAL
```