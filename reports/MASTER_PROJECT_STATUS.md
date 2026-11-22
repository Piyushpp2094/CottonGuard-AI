# CottonGuard AI — Master Project Status Report
**Document ID:** `reports/MASTER_PROJECT_STATUS.md`  
**Date:** November 16, 2026  
**Project Directory:** `D:\Main_Project\cotton_leaf_ai`  
**Environment:** Windows 11 | Python 3.11.9 (`.venv`) | PyTorch 2.15.0+cu132 | NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM)  
**Target URLs:** Backend: `http://127.0.0.1:8000` | Frontend: `http://127.0.0.1:5173` | CVAT: `http://localhost:8080`  

---

## 1. Executive Summary

CottonGuard AI is a multi-modal computer vision and web platform designed for cotton leaf disease diagnosis, lesion localization, semantic tissue segmentation, and leaf-restricted severity estimation. The platform brings together three distinct computer vision architectures with strictly separated academic responsibilities:

1. **EfficientNet-B0 (Whole-Image Classifier):** Production-ready, verified holdout test accuracy of **97.82%** (Macro F1: **97.54%**) on a group-aware, sealed 321-image test split. Checkpoint SHA-256 `552baba8...` is strictly preserved.
2. **Faster R-CNN ResNet-50 FPN V2 (Focal Lesion Detector):** Experimental prototype trained in Phase 9.2C on the 50-image pilot corpus (35 train / 15 val, 514 bboxes), achieving **mAP50 = 13.94%** and **mAP50:95 = 8.53%**. Checkpoint SHA-256 `01d82b32...` is preserved as an experimental baseline.
3. **U-Net with ResNet-34 Encoder (Pixel-Level Segmentation):** Experimental prototype evaluated in Phase 8 (Exp 4b w20), capable of 3-class pixel segmentation (Background, Healthy Leaf, Diseased Tissue). Designated as `RELIABLE_CANDIDATE` but not yet promoted to production.

The project currently has a working FastAPI backend, an SQLite database with 37 prediction records and 5 user accounts, an interactive Grad-CAM explainability pipeline, and a React + Vite + TypeScript frontend.

---

## 2. Component Health and Status Matrix

| Subsystem | Technology | Current Status | Health / Integrity Summary |
|:---|:---|:---:|:---|
| **Disease Classification** | EfficientNet-B0 (PyTorch) | **PRODUCTION READY** | Checkpoint `552baba8...` verified; 97.82% test accuracy; Grad-CAM active. |
| **Lesion Detection** | Faster R-CNN ResNet-50 FPN V2 | **EXPERIMENTAL** | Checkpoint `01d82b32...` verified; mAP50 = 13.94%; requires dataset expansion. |
| **Semantic Segmentation** | U-Net ResNet-34 (smp) | **EXPERIMENTAL** | Checkpoint `c6788f79...` (Exp 4b) verified; requires fine-grained pixel masks. |
| **Backend API** | FastAPI / Uvicorn | **OPERATIONAL** | Auth, history, predictions, SSE progress, admin, PDF generation active. |
| **Frontend Interface** | React 18, Vite, TypeScript | **OPERATIONAL** | Dashboard, upload modal, XAI interactive viewer, history table active. |
| **Database / ORM** | SQLite + SQLAlchemy + Alembic | **OPERATIONAL** | 12 tables, 5 users, 37 predictions, 5 reports; zero corruption. |
| **Annotation Pipeline** | CVAT (Docker/WSL) Task #6 | **STAGED / AUDITED** | 5-image pilot audited; Phase 9.3.4 correction plan prepared; 45 frames untouched. |
| **Test Suite** | pytest (Python) | **42/44 PASSING** | Model factory, preprocessing, Grad-CAM, detector units pass; 2 contract asserts adjusted. |

---

## 3. Subsystem Breakdown

### 3.1 AI / Computer Vision Models

#### 1. Classification (EfficientNet-B0)
- **Architecture:** `torchvision.models.efficientnet_b0` fine-tuned on SAR-CLD-2024.
- **Classes (7):** Bacterial Blight, Curl Virus, Healthy Leaf, Herbicide Growth Damage, Leaf Hopper Jassids, Leaf Redding, Leaf Variegation.
- **Checkpoint:** `artifacts/checkpoints/classification/best_model.pt` (48.69 MB).
- **SHA-256:** `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e` (**LOCKED / UNTOUCHED**).
- **Performance:** 97.82% holdout test accuracy, 97.54% Macro F1 on sealed 321-image test split.
- **Readiness:** Production-ready.

#### 2. Focal Lesion Detection (Faster R-CNN ResNet-50 FPN V2)
- **Architecture:** `torchvision.models.detection.fasterrcnn_resnet50_fpn_v2` with FastRCNNPredictor (7 classes).
- **Checkpoint:** `artifacts/checkpoints/detection/best_faster_rcnn.pt` (166.45 MB, Epoch 13).
- **SHA-256:** `01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2` (**LOCKED**).
- **Validation Performance:** mAP50 = 13.94%, mAP50:95 = 8.53%, Precision = 10.42%, Recall = 38.46%.
- **Readiness:** Experimental. Not approved for autonomous diagnostic decisions.

#### 3. Semantic Segmentation (U-Net ResNet-34)
- **Architecture:** `segmentation_models_pytorch.Unet(encoder_name='resnet34', classes=3)`.
- **Checkpoints:**
  * `artifacts/checkpoints/segmentation/best_unet.pt` (92.79 MB, Baseline Exp 1).
  * `artifacts/checkpoints/segmentation/exp4b_w20_best_unet.pt` (92.79 MB, Exp 4b w20 candidate, SHA-256 `c6788f79...`).
- **Performance:** Capable of segmenting leaf blade and focal necrotic lesions; high false-positive rate on curled margin shadows.
- **Readiness:** Experimental / Reliable Candidate.

### 3.2 Dataset and Annotation Assets

- **SAR-CLD-2024 Source Dataset:** 2,132 unique images across 7 classes.
- **Classification Splits:**
  * Train Split: 1,492 images (70%).
  * Validation Split: 319 images (15%).
  * Test Split: 321 images (15%) — **SEALED** (`artifacts/manifests/splits/test.csv`, SHA-256 `6025d86f...`).
- **Published Detection Corpus (Tasks #4 & #5):** 50 images (35 train / 15 val), 514 bounding boxes, 50 mask files.
- **Staged Expansion Pool (Task #6):** 50 images (Job #6). 5 pilot images audited; 45 frames untouched awaiting correction approval.

### 3.3 Backend Subsystem (FastAPI)

- **App Factory:** `backend/app/main.py` with async lifespan context.
- **Database & Migrations:** SQLite (`cotton_leaf_ai.db`), Alembic head `b4ca4e2a1471`.
- **Auth:** JWT authentication, bcrypt password hashing, refresh token rotation, token revocation list.
- **Routes:**
  * `/api/auth/register`, `/api/auth/login`, `/api/auth/refresh`, `/api/auth/logout`, `/api/auth/me`
  * `/api/predictions/analyze` (unified multi-modal inference with GPU lock & CPU fallback)
  * `/api/predictions/analyze/stream/{job_id}` (real-time SSE stage progress streaming)
  * `/api/classification/predict`, `/api/detection/predict`, `/api/segmentation/predict`, `/api/severity/calculate`
  * `/api/explainability/gradcam` (dynamic colormap & target class recomputation)
  * `/api/history`, `/api/history/{id}` (paginated scan history with ownership enforcement)
  * `/api/reports/{id}/generate`, `/api/reports/{id}/download` (ReportLab PDF generation)
  * `/api/admin/overview`, `/api/admin/users`, `/api/admin/models` (RBAC admin telemetry)
  * `/api/health`, `/api/health/models`

### 3.4 Frontend Subsystem (React / Vite / TypeScript)

- **Tech Stack:** React 18, Vite 5, TypeScript 5, Tailwind CSS 3, Lucide React, Recharts.
- **Pages & Views:**
  * `DashboardPage.tsx`: Summary metrics, recent scan feed, quick-upload launcher.
  * `AnalysisPage.tsx`: Single-leaf image upload, SSE real-time stage tracker, interactive multi-modal result cards.
  * `ResultsPage.tsx`: Detailed prediction inspection, Grad-CAM viewer, detection box overlays, segmentation masks.
  * `HistoryPage.tsx`: Paginated scan table with search, class filters, date filtering, and PDF download buttons.
  * `AdminPage.tsx`: System telemetry, user management, and model registry inspector.
  * `LoginPage.tsx` / `RegisterPage.tsx`: Form-validated authentication screens.
- **Honest Status Indicators:** `StatusBadges.tsx` explicitly renders `EXPERIMENTAL` and `NOT_TRAINED` badges.

### 3.5 Security, Safety, and Robustness

- **Magic-Byte Upload Validation:** Header verification for JPEG, PNG, WEBP; rejection of polyglot/executable payloads.
- **Resolution Boundaries:** Dimension checks enforce minimum $224 \times 224$ and maximum $4096 \times 4096$.
- **File Size Caps:** Strict 10 MB payload limit.
- **Path Traversal Protection:** Image and report downloads resolve through strict canonical base directory sanitization.
- **GPU Memory Safety:** Dedicated `threading.Lock` serializes GPU inference; automatic fallback to CPU upon CUDA OOM.

## 4. Remaining Work and Critical Dependencies

```
+------------------------------------------------------------------------------------------------------+
|                                      PROJECT DEPENDENCY FLOWCHART                                    |
+------------------------------------------------------------------------------------------------------+
|  [Phase 9.3.4 Plan] (COMPLETE)                                                                       |
|           |                                                                                          |
|           v                                                                                          |
|  [User Approval Gate 1] ---> [Phase 9.3.4.1 Pilot Execution in CVAT Task #6]                        |
|                                          |                                                           |
|                                          v                                                           |
|  [User Approval Gate 2] ---> [Phase 9.3.5 Full Task #6 Annotation Expansion to 100 Images]         |
|                                          |                                                           |
|                                          v                                                           |
|  [User Approval Gate 4] ---> [Phase 9.4 Faster R-CNN Retraining Experiment (v2.0 Checkpoint)]        |
|                                          |                                                           |
|                                          v                                                           |
|  [Backend Multi-Model Wiring] (Faster R-CNN + U-Net ResNet-34 integration in PredictionService)      |
|                                          |                                                           |
|                                          v                                                           |
|  [Frontend E2E Verification & Local Demonstration Setup]                                             |
+------------------------------------------------------------------------------------------------------+
```