# CottonGuard AI — Phase 1 Dataset & Annotation Pipeline Completion Report
**Document ID:** CGAI-REP-P1-001  
**Generated Date:** 2026-11-19  
**Status:** COMPLETED & VERIFIED  
**Governing Authority:** Lead AI/ML Engineer & Systems Architect  
**Workspace:** `D:\Main_Project\cotton_leaf_ai`  

---

## 1. Executive Summary
Phase 1 has successfully established the canonical dataset and derived annotation export pipeline for the CottonGuard AI system. Without mutating or deleting any data in live CVAT Task #6 (which remains intact at 1,276 shapes), a programmatic **Derived Annotation Export** was constructed. 

The pipeline ingested the authoritative Post-Stage-3 CVAT snapshot, excluded the 303 frozen deletion IDs, excluded all bounding rectangles (including the 37 DEC-01 boxes), and enforced the strict leaf-priority clipping rule. A verified 70-image development dataset (35 from Batches 1 & 2, 35 from Batch 3) has been fully rasterized into high-fidelity 3-class segmentation masks with verified pixel values `{0, 1, 2}` and accompanying QA overlays. CVAT holdout frames 35–49 (Global IDs 85–99) remain strictly sealed.

---

## 2. Classification Dataset & Split Integrity Verification

The primary cotton leaf dataset comprises **2,132 total images** across the 7 recognized disease and health categories.

### 2.1 Official Splits & Class Distribution

| Class Label | Train Split (`train.csv`) | Validation Split (`validation.csv`) | Test Split (`test.csv` - SEALED) | Total Dataset |
| :--- | :---: | :---: | :---: | :---: |
| **Leaf Redding** | 404 | 86 | 88 | 578 |
| **Curl Virus** | 302 | 65 | 64 | 431 |
| **Herbicide Growth Damage** | 196 | 42 | 42 | 280 |
| **Healthy Leaf** | 177 | 37 | 38 | 252 |
| **Bacterial Blight** | 175 | 37 | 38 | 250 |
| **Leaf Hopper Jassids** | 157 | 34 | 34 | 225 |
| **Leaf Variegation** | 81 | 18 | 17 | 116 |
| **Total Images** | **1,492** | **319** | **321** | **2,132** |

### 2.2 Test Split Protection Statement
- **Split File:** [test.csv](file:///D:/Main_Project/cotton_leaf_ai/artifacts/manifests/splits/test.csv) (321 images).
- **Integrity Rule:** The test set is strictly protected. It was not accessed, sampled, modified, or used for model tuning or architecture selection during Phase 1.
- All evaluation, threshold tuning, and model comparison in subsequent phases are conducted strictly on `validation.csv` (319 images).

---

## 3. CVAT Derived Annotation Export Architecture

### 3.1 Non-Destructive Ingestion Methodology
Live CVAT Task #6 / Job #6 remains locked and unmodified at **1,276 live shapes**. Because live Stage 4 deletion was not authorized, the clean segmentation ground truth was generated using a programmatic derived export pipeline:
1. **Source Snapshot:** [cvat_job6_stage3_post_addition_snapshot.json](file:///D:/Main_Project/cotton_leaf_ai/artifacts/backups/cvat_job6_stage3_post_addition_snapshot.json) (1,276 shapes).
2. **Exclusion Registry:** [stage4_frozen_deletion_registry.json](file:///D:/Main_Project/cotton_leaf_ai/artifacts/audit_phase11/stage4_frozen_deletion_registry.json) (303 frozen deletion IDs).
3. **Retained Inventory:** Exactly $1,276 - 303 = 973$ shapes.
   - 52 Leaf polygons (`label_id = 5`)
   - 576 Diseased_Region polygons (`label_id = 6`)
   - 345 Diseased_Region rectangles (including the 37 DEC-01 bounding boxes)
4. **DEC-01 & Rectangle Policy:** Rectangles represent bounding box detection targets rather than semantic segmentations. All 345 rectangles were strictly excluded from U-Net segmentation mask rasterization.
5. **Clipping Rule:** 
   $$\text{Final Mask}[(M_{\text{leaf}} = 1) \land (M_{\text{disease}} = 1)] = 2$$
   - Background = 0
   - Healthy Leaf Tissue = 1
   - Diseased Leaf Region = 2

---

## 4. Development Dataset Inventory (70 Verified Images)

The 70 development images comprise:
- **Global IDs 00–34 (35 images):** Pre-verified annotations from Pilot Batch 01 (21 images) and Pilot Batch 02 (14 images).
- **Global IDs 50–84 (35 images):** High-precision derived annotations from CVAT Task #6 frames 00–34 (Pilot Batch 03).
- **Holdout Frames 35–49 (Global IDs 85–99):** Strictly sealed and excluded from mask generation and inspection.

### 4.1 Class Distribution (70 Development Images)

| Disease Class | Image Count | % of Development Set |
| :--- | :---: | :---: |
| **Bacterial Blight** | 12 | 17.1% |
| **Curl Virus** | 12 | 17.1% |
| **Healthy Leaf** | 12 | 17.1% |
| **Herbicide Growth Damage** | 10 | 14.3% |
| **Leaf Hopper Jassids** | 8 | 11.4% |
| **Leaf Reddening** | 8 | 11.4% |
| **Leaf Variegation** | 8 | 11.4% |
| **Total Development Images** | **70** | **100.0%** |

### 4.2 Pixel Statistics & Severity Metrics

| Metric | Aggregated Value |
| :--- | :--- |
| **Total Development Images** | 70 images |
| **Total Leaf Pixels** | 25,829,207 px |
| **Total Diseased Pixels** | 4,879,470 px |
| **Mean Affected Severity %** | 17.35% |
| **Min Affected Severity %** | 0.00% (Healthy Leaf controls) |
| **Max Affected Severity %** | 99.24% (Severe blight / necrosis) |
| **Mask Pixel Values Verified** | Exactly $\{0, 1, 2\}$ for all 70 masks |

---

## 5. Artifact & Deliverable Registry

| Artifact Name | Relative File Path | Description |
| :--- | :--- | :--- |
| **Dataset Manifest (JSON)** | [derived_70_image_dataset_manifest.json](file:///D:/Main_Project/cotton_leaf_ai/artifacts/manifests/derived_70_image_dataset_manifest.json) | Complete metadata, shape counts, SHA-256 hashes for 70 pairs |
| **Dataset Manifest (CSV)** | [derived_70_image_dataset_manifest.csv](file:///D:/Main_Project/cotton_leaf_ai/artifacts/manifests/derived_70_image_dataset_manifest.csv) | Tabular export with fold assignments and pixel statistics |
| **Dataset Summary** | [derived_70_dataset_summary.json](file:///D:/Main_Project/cotton_leaf_ai/artifacts/manifests/derived_70_dataset_summary.json) | High-level distribution and pixel aggregation metrics |
| **Derived Masks Directory** | `data/cvat_derived_export/masks/` | 70 single-channel PNG ground-truth masks |
| **Derived Overlays Directory**| `data/cvat_derived_export/overlays/` | 70 side-by-side RGB visual QA overlays |
| **Review Sample Gallery** | `artifacts/derived_export_review/` | High-priority inspection samples (Frames 15, 21, 22, 24, 31, 50, etc.) |
| **Pipeline Script** | [build_derived_annotation_dataset.py](file:///D:/Main_Project/cotton_leaf_ai/scripts/build_derived_annotation_dataset.py) | Reproducible derived dataset generation script |

---

## 6. Sign-off and Readiness for Phase 2
Phase 1 is complete. The derived dataset manifest, 70 validated ground truth masks, and visual QA overlays are ready for model training, evaluation, and segmentation benchmarking.
