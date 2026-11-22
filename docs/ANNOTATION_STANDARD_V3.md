# CottonGuard AI — Master Precision Annotation Standard (v3.0)

**Project:** CottonGuard AI  
**Scope:** Consolidated 100-Image Benchmark Dataset (Task #4, Task #5, Task #6) across all 7 SAR-CLD-2024 Cotton Leaf Classes  
**Target Architectures:**
1. **EfficientNet-B0** — Whole-image 7-class disease classification
2. **Faster R-CNN ResNet-50 FPN V2** — High-precision focal lesion detection ($[x_1, y_1, x_2, y_2]$ bounding boxes)
3. **U-Net ResNet-34** — Precise dual-target pixel segmentation ($0=\text{Background}, 1=\text{Healthy Lamina}, 2=\text{Diseased Tissue}$) for leaf-restricted severity estimation

---

## 1. Core Principles & Biological Philosophy

1. **Native Visual Evidence Over Automation:**
   - The original native-resolution ($800 \times 800$) image is the primary biological truth.
   - Algorithmic predictions (SAM, HSV thresholds, color spaces) are aids, never final authority. No AI-generated polygon is accepted without manual verification.
   - Normal green lamina, shadows, dark primary/secondary veins, dirt flecks, specular highlights, and natural leaf edges must never be labeled as disease.

2. **Focal vs. Diffuse Symptom Differentiation (Dual-Tier Policy):**
   - **Focal Lesions (Tier 1):** Distinct, localized, necrotic/chlorotic spots with sharp boundaries (e.g., angular leaf spots of Bacterial Blight, necrotic puncture cores of Leaf Hopper Jassids, dark necrotic spots of Leaf Redding). These are valid targets for Faster R-CNN bounding boxes.
   - **Diffuse / Systemic Symptoms (Tier 2):** Broad regional physiological discolorations (e.g., anthocyanin blush in Leaf Redding, marginal curling/cupping in Curl Virus or Herbicide Damage, broad marginal chlorosis in Jassids). These must be segmented for U-Net pixel analysis but **excluded from Faster R-CNN object bounding boxes** to prevent false anchor suppression and anchor training collapse.

3. **True Negative Controls:**
   - Healthy Leaf samples serve as golden negative controls: exactly 1 valid leaf polygon, strictly 0 disease polygons, strictly 0 disease bounding boxes, and strictly 0% disease severity.

4. **Split Governance & Integrity:**
   - 100 consolidated images: exactly 70 Training and 30 Validation.
   - Sealed 321-image test set (`artifacts/manifests/splits/test.csv`) is strictly off-limits.
   - Zero duplicate-group leakage across train/validation/test partitions.

---

## 2. Seven-Class Disease & Visual Morphology Standards

### 2.1 Bacterial Blight (*Xanthomonas citri* pv. *malvacearum*)
- **Visual Presentation:** Angular, water-soaked, vein-delimited necrotic spots ranging from dark brown to black, often with subtle greasy margins.
- **Leaf Mask:** Full lamina contour; exclude petiole beyond leaf base.
- **Focal Detections (Faster R-CNN):** One tight bounding box per discrete lesion. Where lesions coalesce along veins, if separated by $\ge 10\text{ px}$ of green lamina, decompose into distinct boxes; if completely contiguous, bound the cohesive cluster. Healthy primary veins bisecting lesions must not be boxed in isolation.
- **Segmentation (U-Net):** Pixel-exact polygons enclosing dark necrotic cores.

### 2.2 Curl Virus (Cotton Leaf Curl Geminivirus - CLCuV)
- **Visual Presentation:** Severe upward/downward cup-shaped laminar curling, thickened secondary veins, enations on abaxial surface, mosaic puckering.
- **Leaf Mask:** Follow the deformed, wrinkled, convoluted leaf perimeter faithfully.
- **Focal Detections (Faster R-CNN):** None (or strictly restricted to secondary necrotic focal tears). Whole-leaf structural curling is a morphological syndrome, not a collection of small object boxes.
- **Segmentation (U-Net):** Affected crinkled and chlorotic laminar areas annotated as diseased tissue ($2$); remaining lamina as healthy ($1$).

### 2.3 Healthy Leaf (Control)
- **Visual Presentation:** Uniform green lamina with normal palmate venation, free of necrotic spots, chlorosis, or abnormal morphological twisting.
- **Leaf Mask:** Exact outer perimeter enclosing 100% of the visible blade.
- **Focal Detections (Faster R-CNN):** **STRICTLY ZERO.** Emits empty label file (`0` lines).
- **Segmentation (U-Net):** Mask strictly contains values $\{0, 1\}$. Value $2$ (disease) is strictly forbidden.

### 2.4 Herbicide Growth Damage (Phytotoxicity / Hormone Disruption)
- **Visual Presentation:** Epinastic twisting, strapping (parallel leaf veins, "shoestringing"), marginal serration deformities, stunting.
- **Leaf Mask:** Follows distorted narrow lobes.
- **Focal Detections (Faster R-CNN):** Diffuse structural deformities must not be chopped into artificial small boxes. Only localized chemical necrosis spots qualify for bounding boxes.
- **Segmentation (U-Net):** Deformed/chlorotic lamina sectors segmented as diseased tissue.

### 2.5 Leaf Hopper Jassids (*Amrasca biguttula biguttula*)
- **Visual Presentation:** "Hopperburn" — triangular marginal yellowing (chlorosis) progressing inward from leaf tips/margins, followed by browning, downward curling, and necrotic focal punctures.
- **Leaf Mask:** Outer perimeter including curled margins.
- **Focal Detections (Faster R-CNN):** Distinct necrotic puncture feeding spots are boxed. Wide marginal chlorotic sectors are annotated as discrete marginal ribbons, avoiding arbitrary rectilinear grid slicing. Normal green central lamina must be excluded.
- **Segmentation (U-Net):** Curled marginal chlorotic ribbons and necrotic tips segmented as diseased tissue ($2$). Normal green central lamina remains healthy ($1$).

### 2.6 Leaf Redding (Physiological Anthocyanosis / Nutrient Stress)
- **Visual Presentation:** Intense red, purple, or bronze laminar discoloration between veins due to anthocyanin synthesis triggered by low night temperatures, magnesium deficiency, or insect feeding.
- **Leaf Mask:** Macro leaf contour (or full-frame $[0,0, 800,800]$ for close-up macro shots).
- **Focal Detections (Faster R-CNN):** Strictly **Tier 1 focal dark necrotic spots**. The diffuse red blush (**Tier 2**) is excluded from detector boxes.
- **Segmentation (U-Net):** Both focal necrosis and diffuse anthocyanin blush are segmented as diseased tissue ($2$), with auxiliary tier tagging (`tier=focal` vs `tier=diffuse`).

### 2.7 Leaf Variegation (Genetic Chimera / Natural Patterning)
- **Visual Presentation:** Striking irregular white, cream, or pale-yellow patches interspersed with dark green lamina, bounded by cellular lineage boundaries.
- **Leaf Mask:** Complete outer lamina outline.
- **Focal Detections (Faster R-CNN):** Cohesive variegation patches annotated with discrete boxes where visual contrast $\Delta L \ge 15.0$.
- **Segmentation (U-Net):** Achlorophyllous variegated patches segmented as distinct regions.

---

## 3. Geometric & Quality Rules

### 3.1 Bounding Box Rules
1. **Coordinate Bounds:** All coordinates $[x_1, y_1, x_2, y_2]$ must satisfy $0 \le x_1 < x_2 \le 800$ and $0 \le y_1 < y_2 \le 800$.
2. **Minimum Dimensions:**
   - Width $w = x_2 - x_1 \ge 12\text{ px}$ and Height $h = y_2 - y_1 \ge 12\text{ px}$ for automatic acceptance.
   - $10 \le w, h < 12\text{ px}$: Flagged for manual optical review. Accepted only if a distinct necrotic core is visible at native resolution.
   - $w < 10\text{ px}$ or $h < 10\text{ px}$: **REJECTED.**
3. **Minimum Area:** $w \times h \ge 100\text{ px}^2$.
4. **Boundary Truncation:** Lesions bisected by the image border must be tagged with `is_boundary_truncated=True`.
5. **No Vein Shadow Artifacts:** Boxes must be tightly fitted to lesion boundaries, excluding adjacent dark vein shadows.

### 3.2 Leaf Mask Segmentation Rules
1. **Exclusions:** Background surfaces, fingers, gloves, soil, stems, weeds, and tags must receive pixel label $0$.
2. **Smooth Contour:** Use non-self-intersecting polygons with adequate vertex density ($16\text{--}250$ points) following lobes, sinuses, and margins.
3. **Macro Shots:** Full-frame $[0,0, 800,800]$ is permitted **only** when the entire frame consists of leaf lamina without visible background margins (`is_full_frame_macro=True`).

### 3.3 Disease Segmentation & Severity Calculation
1. **Pixel Class Mapping:**
   - $0$: Background (exterior to leaf).
   - $1$: Healthy cotton leaf lamina.
   - $2$: Diseased leaf tissue (necrotic, chlorotic, or anthocyanic).
2. **Clipping Invariant:** Disease pixels must be strictly contained within the leaf mask:
   $$\text{Disease Pixels} = \text{Disease Mask} \cap \text{Leaf Mask}$$
   No disease pixel may exist outside the leaf polygon.
3. **Severity Formula (Leaf-Restricted Mutually Exclusive Accounting):**
   $$\text{Severity } (\%) = \left( \frac{\sum_{(x,y)} \mathbb{I}(M(x,y) = 2 \land L(x,y) = 1)}{\sum_{(x,y)} \mathbb{I}(L(x,y) = 1)} \right) \times 100$$
   where $M(x,y)$ is the disease mask and $L(x,y)$ is the leaf mask.

---

## 4. Uncertain Region Taxonomy & Handling Policy

When visual evidence is ambiguous (low contrast, optical blur, dirt vs. lesion):

| Review Status | Definition | Action Taken |
|:---|:---|:---|
| `CONFIRMED_DISEASE` | Unambiguous necrotic/chlorotic symptom with clear border ($\Delta L \ge 15.0$) | Exported to training and validation sets |
| `CONFIRMED_HEALTHY` | Verified healthy lamina or natural vein/leaf variation | Retained as healthy background ($1$) |
| `POSSIBLE_DISEASE` | Suspicious discoloration but lacks sharp margins | Excluded from focal boxes; retained as segmentation ignore/review |
| `INSUFFICIENT_VISUAL_EVIDENCE` | Occluded by glare, motion blur, or out-of-focus optics | Excluded from training loss; logged in review queue |
| `BOUNDARY_TRUNCATED` | Genuine lesion clipped by image perimeter | Annotated with `is_boundary_truncated=True` |
| `REQUIRES_EXPERT_REVIEW` | Potential co-infection, physiological stress vs. pathogen | Staged in review queue with high-res crop |

---

## 5. Automated 20-Check Validation Pipeline

Every annotated frame must pass the complete 20-point validation suite:

1. **Check 01:** Source image exists on disk and is readable.
2. **Check 02:** Frame-to-filename mapping verified against manifest.
3. **Check 03:** Native image dimensions strictly $800 \times 800$.
4. **Check 04:** Label IDs strictly conform to schema ($5=\text{Leaf}, 6=\text{Diseased\_Region}$).
5. **Check 05:** All polygon/box coordinates within $[0, 800]$.
6. **Check 06:** Polygons are non-degenerate (vertex count $\ge 3$, area $\ge 50\text{ px}^2$).
7. **Check 07:** No self-intersecting polygon boundaries.
8. **Check 08:** Bounding box dimensions satisfy minimum width/height thresholds.
9. **Check 09:** Polygon-to-box consistency verified ($x_1 = \min(x)$, $y_1 = \min(y)$, etc.).
10. **Check 10:** Disease masks strictly restricted to leaf interior (outside pixels $= 0$).
11. **Check 11:** Healthy negative controls have strictly 0 disease shapes and 0% severity.
12. **Check 12:** Zero duplicate boxes ($IoU = 1.0$ within same class).
13. **Check 13:** No unexplained same-class overlaps ($IoU > 0.50$ without composite justification).
14. **Check 14:** Leaf mask covers $\ge 15\%$ of image unless boundary fragment.
15. **Check 15:** Split assignment matches manifest (train vs validation).
16. **Check 16:** Zero duplicate-group leakage across splits.
17. **Check 17:** Absolute zero overlap with sealed 321-image test set.
18. **Check 18:** Segmentation mask pixel values strictly $\in \{0, 1, 2\}$.
19. **Check 19:** Healthy control mask values strictly $\in \{0, 1\}$.
20. **Check 20:** Zero CVAT tracks present in task (shapes-only mode).
