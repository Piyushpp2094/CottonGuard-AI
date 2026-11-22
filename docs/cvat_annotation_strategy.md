# CVAT Annotation Verification & Dataset Expansion Strategy (Phase 6)

**CVAT Server:** `http://localhost:8080` (Docker Compose deployment kept isolated from application DB)  
**Project Schema:** `Leaf` polygon + `Diseased_Region` polygon (mutable attribute `disease_type` with 6 disease options)

---

## 1. Label Schema & Inheritance Verification Checklist

In CVAT Community Edition, project-level labels do **not** automatically validate until verified at the task level (`Cotton_Leaf_Segmentation_Pilot`):

1. Start Docker Desktop & CVAT stack cleanly (`docker compose up -d` inside `D:\Main_Project\cvat_setup\cvat`).
2. Verify label inheritance programmatically:
   ```powershell
   wsl -d Ubuntu -- ~/.venvs/cotton_leaf_ai/bin/python /mnt/d/Main_Project/cotton_leaf_ai/scripts/inspect_cvat.py --username <cvat_user> --prompt-password
   ```
3. Confirm output reports `"is_schema_valid": true`:
   - Label `1`: `Leaf` (polygon)
   - Label `2`: `Diseased_Region` (polygon) with **mutable** attribute `disease_type` covering:
     `Bacterial Blight`, `Curl Virus`, `Herbicide Growth Damage`, `Leaf Hopper Jassids`, `Leaf Redding`, `Leaf Variegation`.
   - Healthy Leaf images carry a `Leaf` polygon and **zero** `Diseased_Region` polygons (except true incidental abiotic blemish verified by human review).

---

## 2. Export & Automated Ingestion Pipeline

Once annotations are submitted in CVAT:
1. Export task using format **CVAT for images 1.1** (`.zip` archive containing `annotations.xml` and/or images).
2. Run ingestion & quality verification:
   ```powershell
   wsl -d Ubuntu -- ~/.venvs/cotton_leaf_ai/bin/python /mnt/d/Main_Project/cotton_leaf_ai/scripts/prepare_cvat_annotations.py --input <path_to_export.zip>
   ```
3. This pipeline automatically enforces:
   - **Pixel derivation rule:** Background = `0`, Healthy = `Leaf - Diseased_Region` (`1`), Diseased = `Diseased_Region ∩ Leaf` (`2`) so lesion pixels never fall outside the leaf boundary.
   - **Leakage Guard:** Compares image SHA-256 hashes & filenames against `artifacts/manifests/splits/test.csv` so untouched classifier test images are never mixed into U-Net or YOLO training splits.
   - **Dual Dataset Export:**
     - Semantic segmentation PNG index masks -> `data/annotations/masks/` & `images/`
     - YOLO normalized bounding-box labels (`[class_id cx cy w h]`) -> `data/detection/labels/` & `images/`
   - **Visual QA Inspection:** Renders side-by-side color mask overlays + bounding boxes in `artifacts/annotations_qa/` and writes `artifacts/manifests/cvat_annotation_quality_report.json`.

---

## 3. Pilot Limitation & Scientifically Defensible Expansion Strategy

> [!WARNING]
> **Pilot Size Disclaimer:** The initial `Cotton_Leaf_Segmentation_Pilot` contains **14 images** (2 per diseased class + 2 Healthy Leaf). Fourteen images are strictly a schema and pipeline validation pilot — **insufficient** to train a reliable production U-Net segmentation or YOLO detector.

### Recommended Staged Expansion Plan (Respecting Group-Aware Splits)

| Stage | Target Sample Count | Split Selection Policy | Validation Milestone |
|---|---:|---|---|
| **Stage 1 (Pilot)** | 14 images (completed pilot) | Select **only** from `train.csv` and `validation.csv`; verify zero overlap with `test.csv`. | Validate polygon-to-mask conversion, `Leaf - Diseased` derivation, QA overlay rendering. |
| **Stage 2 (Baseline Training)** | 140–210 images (20–30 per class) | Stratified sampling per `group_id` from `train.csv` (110–160 images) & `validation.csv` (30–50 images). | First U-Net (`mIoU`/`Dice`) & YOLO (`mAP50`) baseline benchmarks on RTX 3050 GPU. |
| **Stage 3 (Human-in-the-Loop Scaling)** | 420–700 images (60–100 per class) | Run Stage 2 U-Net/YOLO pre-annotations on unlabelled `train.csv` leaves -> import pre-annotations into CVAT -> human expert corrects polygons instead of drawing from scratch (~4x faster). | Target Dice > 0.80, IoU > 0.70, mAP50 > 0.70; reserve 50–70 annotated test-group images separately **only** for locked segmentation/detection test evaluation. |
