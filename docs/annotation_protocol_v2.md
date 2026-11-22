# CottonGuard AI — Phase 6 Annotation Protocol & Quality Standard (v2.0)

**Scope:** All seven SAR-CLD-2024 cotton leaf classes  
**Target Pipelines:** Dual-export workflow supporting YOLO bounding-box detection (`0..5`) and U-Net 3-class pixel segmentation (`{0, 1, 2}`) for leaf-restricted severity estimation.  
**Split Governance:** Strictly preserves original dataset, augmented dataset, and leak-free group-aware splits (`artifacts/manifests/splits/{train,validation,test}.csv`). No augmented images may enter `validation` or `test` splits, and no duplicate/near-duplicate `group_id` may span train/val/test boundaries.

---

## 1. Seven-Class Disease & Morphology Annotation Criteria

| # | Classifier Class | Detection Label (`configs/detection_classes.json`) | Pixel Mask Rules (`0=Bg, 1=Leaf, 2=Disease`) | Visual Annotation Boundary Guidance |
|---|---|---|---|---|
| 0 | **Bacterial Blight** | `Diseased_Region` (`disease_type="Bacterial Blight"`, YOLO `class_id=0`) | `Leaf` polygon (`1`) + `Diseased_Region` polygons (`2`) | Enclose angular water-soaked spots, vein-delimited dark brown necrotic lesions, and coalesced bacterial blight patches tightly. Split separate lesion clusters into distinct boxes when separated by `>= 10 px` healthy lamina. |
| 1 | **Curl Virus** | `Diseased_Region` (`disease_type="Curl Virus"`, YOLO `class_id=1`) | `Leaf` polygon (`1`) + `Diseased_Region` polygons (`2`) | Trace outer deformed leaf outline as `Leaf`. Mark zones showing pronounced upward/downward cupping, vein swelling/enation, crinkling, and mosaic puckering as `Diseased_Region` boxes/polygons. |
| 2 | **Healthy Leaf** | **None** *(Negative sample; 0 target boxes -> empty `.txt`)* | `Leaf` polygon (`1`) across whole lamina; **0** `Diseased_Region` shapes | Trace full cotton leaf perimeter including lobes and petiole attachment if in frame. Exclude soil, fingers, paper, and sky (`0`). Do not draw disease polygons on minor natural dust/specular highlights. |
| 3 | **Herbicide Growth Damage** | `Diseased_Region` (`disease_type="Herbicide Growth Damage"`, YOLO `class_id=2`) | `Leaf` polygon (`1`) + `Diseased_Region` polygons (`2`) | Annotate cupped, strapped, narrow-lobed epinastic sectors and chlorotic/twisted phytotoxic margins inside the leaf perimeter. |
| 4 | **Leaf Hopper Jassids** | `Diseased_Region` (`disease_type="Leaf Hopper Jassids"`, YOLO `class_id=3`) | `Leaf` polygon (`1`) + `Diseased_Region` polygons (`2`) | Target marginal chlorosis ("hopper burn"), yellow-to-reddish-brown curled leaf edges, and inward-necrotic perimeter rims. |
| 5 | **Leaf Redding** | `Diseased_Region` (`disease_type="Leaf Redding"`, YOLO `class_id=4`) | `Leaf` polygon (`1`) + `Diseased_Region` polygons (`2`) | Segment anthocyanin crimson/purple/reddish laminar patches distinctly separated from remaining green chlorophyll tissue. Box each coherent reddened sector. |
| 6 | **Leaf Variegation** | `Diseased_Region` (`disease_type="Leaf Variegation"`, YOLO `class_id=5`) | `Leaf` polygon (`1`) + `Diseased_Region` polygons (`2`) | Enclose sharply contrasting pale-yellow/white/variegated chimera mosaic zones bounded by green lamina. |

### Handling Overlapping or Ambiguous Disease Regions
- **Overlapping adjacent lesions (same disease):** If two lesions merge continuously without a visible green lamina gap (`< 10 px`), draw one unified `Diseased_Region` polygon (U-Net) and one encompassing YOLO box. If clearly distinct with `>= 10 px` healthy tissue between them, draw separate polygons/boxes.
- **Ambiguous symptoms / secondary rot:** Assign `disease_type` matching the primary verified image diagnosis unless a distinct verified co-infection exists. **Never guess** uncertain blurred edges or specular sun-glare reflections — leave uncertain reflections/shadows as `Leaf` (`1`) rather than inflating diseased severity (`2`), and flag blurred/ambiguous frames in CVAT comments for agronomic QA review.
- **Background clutter & occluded leaf borders:** Keep `Leaf` (`1`) strictly on visible cotton leaf tissue; exclude overlapping stems, fingers, soil, or dry weeds (`0`). Remember `Diseased_Region` pixels are automatically clipped inside `Leaf` (`Diseased_Region ∩ Leaf`), so any stray outer stroke cannot corrupt area percentage.

---

## 2. Dual Annotation Workflows Inside CVAT Community Edition

### Live CVAT Recovery & Schema Fix Note
From the Phase 6 live audit of `D:\Main_Project\cvat_setup\cvat`:
- Containers `cvat_server`, `cvat_ui`, `cvat_db`, `cvat_redis_*`, and SAM ViT-H (`nuclio-nuclio-pth-facebookresearch-sam-vit-h`) are active; `traefik` and `nuclio` are currently exited (`Exited (127)`). Restart ingress before opening the web UI:
  ```powershell
  docker compose -f D:\Main_Project\cvat_setup\cvat\docker-compose.yml -f D:\Main_Project\cvat_setup\cvat\components\serverless\docker-compose.serverless.yml up -d traefik nuclio
  ```
- Existing project `Cotton_Leaf_Disease_Segmentation` (`id=2`) lacks the mutable attribute `disease_type` on `Diseased_Region` (`engine_attributespec = 0 rows`), and existing task `Cotton_Leaf_Segmentation_Pilot` (`id=2`) contains `2` holdout test images (`HGD00001.jpg`, `LHJ00029.jpg`). **Do not use Task #2 as training data**; create fresh split-clean tasks using `artifacts/manifests/cvat_task_label_schema.json` and `data/pilot_batch_01/`.

### Label Schema Configuration (`artifacts/manifests/cvat_task_label_schema.json`)
1. `Leaf` (`type: "polygon"`) — traces total visible cotton leaf blade.
2. `Diseased_Region` (`type: "polygon"`, or `"rectangle"` for box-only refinement) with **mutable select attribute**:
   - Name: `disease_type`
   - Options (`6`): `Bacterial Blight`, `Curl Virus`, `Herbicide Growth Damage`, `Leaf Hopper Jassids`, `Leaf Redding`, `Leaf Variegation`

### Workflow A — YOLO Bounding-Box Pipeline (`data/detection/`)
1. Draw tight `Diseased_Region` polygons (or rectangles) with exact `disease_type` attribute set.
2. Export task in **CVAT for images 1.1 (`.zip`)** format and run `scripts/prepare_cvat_annotations.py --input <export.zip>`.
3. Polygons automatically convert to normalized bounding boxes `[class_id cx cy w h]` in `data/detection/labels/<stem>.txt`. Healthy Leaf samples emit an intentional 0-line `.txt` file as hard-negative examples.
4. Validate strictly using `scripts/validate_phase6_annotations.py`.

### Workflow B — U-Net Pixel-Level Leaf & Disease Mask Pipeline (`data/annotations/`)
1. Use the deployed **SAM ViT-H interactor** inside CVAT (`Interact -> Segment Anything`) to trace clean outer `Leaf` boundaries (`label="Leaf"`), then trace interior lesion regions (`label="Diseased_Region"`).
2. Strict leaf-restricted rasterization rule:
   - Background (`0`) = outside `Leaf`
   - Healthy leaf lamina (`1`) = `Leaf \ Diseased_Region`
   - Diseased leaf tissue (`2`) = `Diseased_Region ∩ Leaf` (clipped strictly inside the leaf mask so diseased area never exceeds leaf area)
3. Export 8-bit single-channel index PNGs (`data/annotations/masks/<stem>.png`) matching source image dimensions `(W, H)` exactly.
4. Validate strictly using `scripts/validate_annotations.py` and `scripts/validate_phase6_annotations.py`.

---

## 3. Pilot Strategy & Reliable Training Thresholds

> **Important Guardrail:** Existing 10-pair smoke checks are **not** sufficient for reliable model training or scientific reporting.

| Stage | Sample Budget | Split Rules | Purpose |
|---|---:|---|---|
| **Stage 1: Representative Pilot** | **21 images** (`3` per class × `7` classes; `14` train + `7` val) | Generated deterministically into `data/pilot_batch_01/` (`0` test overlap, `0` augmented images, group-disjoint). | Human QA review of boundary precision, `disease_type` attribute correctness, mask rasterization, and YOLO box tightness before scaling. **Do NOT train production models on 21 images alone.** |
| **Stage 2: Minimum Training Gate** | **`>= 50` verified pairs** (recommended **`70+`**, i.e., `>= 10` per class across all 7 classes) | Strictly drawn from `train.csv` and `validation.csv` only. | Minimum threshold before `scripts/train_unet_resnet34.py` allows training on RTX 3050 6 GB (`512×512`, AMP, early stopping). |
| **Stage 3: Held-Out Evaluation Subset** | **`50–70` annotated `test.csv` images** (`7–10` per class) | Quarantined separately in `data/test_eval/`; **never** passed to `prepare_cvat_annotations.py` training folders. | Independent test-set reporting (`Dice`, `IoU`, `mAP50`, `mAP50-95`) and manual calibration of provisional severity thresholds (`category_validated`). |
