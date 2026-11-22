# Cotton Leaf Semantic Segmentation Annotation Guide

## 1. Overview and Class Definitions

To support reliable semantic segmentation and affected-area estimation without fabricating ground-truth annotations, images must be annotated using a 3-class pixel-level schema:

| Class Value | Class Name | Color (Hex / RGB) | Description |
|:---:|---|:---:|---|
| **0** | **Background** | `#000000` `(0, 0, 0)` | Soil, hands, sky, background clutter, and non-target vegetation. |
| **1** | **Healthy Leaf** | `#22C55E` `(34, 197, 94)` | Green, asymptomatic, physiologically normal cotton leaf lamina. |
| **2** | **Diseased / Damaged Tissue** | `#EF4444` `(239, 68, 68)` | Lesions, necrotic spots, chlorosis, reddened zones, curl distortions, or jassid burn. |

> [!IMPORTANT]
> The source dataset `SAR-CLD-2024` provides image-level classification labels only. Real semantic segmentation models must strictly be trained on authentic pixel-level masks. Never fabricate or synthesize fake masks for model training.

---

## 2. Recommended Annotation Tools

### Option A: Computer Vision Annotation Tool (CVAT)
1. Install CVAT locally via Docker or use the hosted version at [app.cvat.ai](https://app.cvat.ai).
2. Create a new Task named `Cotton_Leaf_Segmentation`.
3. Define the two foreground labels:
   - `healthy_leaf` (Polygon or Mask)
   - `diseased_tissue` (Polygon or Mask)
4. Upload images from `data/annotations/images/`.
5. Annotate using the Polygon or AI-assisted Brush tool (SAM / Segment Anything plugin inside CVAT is highly recommended for leaf outlines).
6. Export as **Segmentation Mask 1.1** (PNG index masks where pixel values correspond to class indices `0, 1, 2`).

### Option B: Label Studio
1. Install Label Studio: `pip install label-studio`.
2. Start the server: `label-studio start`.
3. Create a project and select **Semantic Segmentation with Polygons/Masks**.
4. Configure classes `0: background`, `1: healthy_leaf`, `2: diseased_tissue`.
5. Export annotations as **PNG Masks**.

---

## 3. Storage and File Naming Conventions

- **Input Images:** Store in `data/annotations/images/<filename>.jpg` (or `.png`).
- **Paired Masks:** Store in `data/annotations/masks/<filename>.png` (must match the base filename of the image).
- **Mask Format:** Single-channel 8-bit grayscale PNG (`uint8`) where:
  - Pixel values are strictly $0$, $1$, or $2$.
  - Dimensions $(W, H)$ must match the source image exactly.

---

## 4. Validating Annotations

Run the validation script to verify mask-image correspondence and integrity:

```powershell
python scripts\validate_annotations.py
```

This verifies:
1. Every mask has a corresponding image of the exact same resolution.
2. Pixel values contain only classes $\{0, 1, 2\}$.
3. Total leaf area exceeds the minimum required threshold.
