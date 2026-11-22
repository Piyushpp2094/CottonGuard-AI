# Cotton Leaf AI — Comprehensive Project Audit, Architecture & Master Implementation Plan

**Audit Date:** 2026-11-10  
**Project Directory:** `D:\Main_Project\cotton_leaf_ai` (WSL path: `/mnt/d/Main_Project/cotton_leaf_ai`)  
**Dataset Directory (Read-Only):** `D:\Main_Project\dataset\SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection`  
**Audit Method:** Read-only inspection of source files, manifests, checkpoints, database, WSL Ubuntu 26.04, Docker/CVAT, and Windows tooling.

---

## 1. Environment Audit Summary

| Environment Component | Verified Status | Findings & Constraints |
|---|---|---|
| **Host OS & Hardware** | Windows 11 Home, HP Victus 15-fa2556TX | Intel Core 5 210H, NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM). |
| **Windows Security Policy** | Smart App Control ON | Windows Code Integrity events 3033 & 3077 reported blocking unsigned/insufficiently signed `torch\lib\c10_cuda.dll`. **Action:** Keep Smart App Control enabled and Windows `.venv` untouched; run AI backend & PyTorch in Ubuntu WSL 2. |
| **WSL 2 Distribution** | `Ubuntu` (Ubuntu 26.04.1 LTS, Resolute Raccoon) | Running on WSL 2 with 955 GB free disk space on `/`. `nvidia-smi` inside WSL sees RTX 3050 6144 MiB (Driver 620.70.03, CUDA UMD 13.5). Default system Python is `3.14.4` (no `pip`, `sudo` requires interactive password -> user-space setup required). No Linux venv or PyTorch installed yet. |
| **Node.js / Vite Tooling** | Missing on Windows & WSL | `frontend/node_modules` does not exist; `node`/`npm` not found on Windows PATH or WSL. `winget` is available on Windows. |
| **Docker & CVAT** | Installed, Daemon Currently Stopped | CVAT repository at `D:\Main_Project\cvat_setup\cvat`; Docker Desktop installed at `%LOCALAPPDATA%\Programs\DockerDesktop`. `docker-desktop` WSL distro is Stopped; `http://localhost:8080` timed out. |
| **Database** | SQLite `cotton_leaf_ai.db` (81,920 bytes) | Tables: `users` (2 rows), `predictions` (14 rows), `prediction_results` (14 rows). Lacks separate tables/columns for YOLO detection results, detailed pixel counts (`leaf_pixels`, `disease_pixels`, `healthy_pixels`), model versions, and persisted report metadata. |

---

## 2. Project Component Inventory & Gap Analysis

| Subsystem | Current Files | Status | Identified Gaps |
|---|---|---|---|
| **Dataset & Manifests** | `artifacts/audit/`, `artifacts/manifests/splits/{train,validation,test}.csv` | **COMPLETE** | None. 2,132 cleaned images across 7 classes; 6 duplicate groups isolated; 1,492 train / 319 val / 321 test; zero cross-split leakage verified. |
| **Model A: EfficientNet-B0 Classifier** | `src/models/classifier.py`, `model_factory.py`, `best_model.pt` | **COMPLETE** | Trained & provenance-verified (97.82% accuracy, 0.9754 macro F1 on 321 test images). Missing standalone `POST /api/classification/predict` endpoint and full class probability return on `/api/predictions/analyze`. |
| **Model B: YOLO Object Detection** | *None* | **MISSING** | No detector module, no license audit documentation, no CVAT polygon-to-bbox preparation script, no training/eval scripts, no `POST /api/detection/predict` route, no DB table. |
| **Model C: U-Net Segmentation** | `src/models/unet.py`, `segmentation_losses.py`, `scripts/train_segmentation.py`, `validate_annotations.py` | **PARTIAL** | 0 images/masks in `data/annotations/`; no trained `best_unet.pt`; bug in `prediction_service.py` line 161 (256x256 input gets re-shrunk to 224x224 by `self.transform`); missing CVAT export-to-mask converter, evaluation script, and standalone `POST /api/segmentation/predict` route. |
| **Model D: Grad-CAM** | `src/explainability/gradcam.py`, `visualization.py`, `backend/app/routes/explainability.py`, `scripts/test_gradcam.py` | **COMPLETE** | `GET /api/predictions/{id}` does not currently return saved `gradcam_*_url` fields for history detail inspection. |
| **Severity & Area Engine** | `src/metrics/area_calculation.py`, `severity.py`, `configs/severity_rules.yaml` | **PARTIAL** | Needs `healthy_pixels` count tracking, strict leaf-mask intersection check, segmentation model version tracking, `POST /api/severity/calculate` route, and explicit provisional-threshold disclaimer until calibrated on real segmentation outputs. |
| **Recommendation Engine & PDF** | `src/recommendations/recommendation_engine.py`, `configs/recommendations.json`, `backend/app/services/report_service.py` | **PARTIAL** | PDF report only embeds original image; should also embed Grad-CAM comparison, segmentation overlay & YOLO detection image when available, plus full pixel metrics and provenance disclaimer. |
| **FastAPI Backend** | `backend/app/{main,models,schemas,database,auth}.py`, `routes/*`, `services/*` | **PARTIAL** | Missing `/api/classification/predict`, `/api/detection/predict`, `/api/segmentation/predict`, `/api/severity/calculate`, `/api/history` (+ filters & pagination), `/api/history/{id}`, `/api/dashboard/stats`, and role-protected `/api/admin/*` endpoints. `tests/test_backend_api.py::test_01_root_and_health` expects JSON on `/` while `/` returns HTML. |
| **Frontend (React + Vite)** | `frontend/src/*` & CDN SPA `backend/app/static/index.html` | **INCOMPLETE** | `frontend/` dependencies never installed (`node_modules` absent); lacks YOLO bounding-box viewer, U-Net 3-panel + interactive opacity slider, Severity pixel breakdown, 7-class probability chart, History filters/pagination/detail drawer, real Admin API wiring (currently hardcoded values in `AdminPage.tsx`). |
| **Deployment Scripts** | *None* | **MISSING** | Need `start_project.ps1`, `stop_project.ps1`, `check_environment.ps1` orchestrating WSL 2 GPU backend (`:8000`) + local Vite frontend (`:5173`), isolated from CVAT (`:8080`). |

---

## 3. Target System Architecture

```text
+-------------------------------------------------------------------------+
|                    WINDOWS 11 HOST (HP Victus Laptop)                   |
|                                                                         |
|  [Browser] ---> http://localhost:5173 (React + Vite + Tailwind + Lucide)|
|      |                     | (Vite proxy /api & /artifacts)             |
|      |                     v                                            |
|      |          http://127.0.0.1:8000 (WSL2 localhost forwarding)       |
|      |                                                                  |
|  [Docker Desktop - Separate] ---> http://localhost:8080 (CVAT CE)       |
+-------------------------------------------------------------------------+
                             |
                             v
+-------------------------------------------------------------------------+
|               UBUNTU 26.04 WSL 2 (NVIDIA RTX 3050 6GB GPU)              |
|  Linux venv: ~/.venvs/cotton_leaf_ai (Python 3.11 + PyTorch CUDA)       |
|                                                                         |
|  FastAPI Backend (/mnt/d/Main_Project/cotton_leaf_ai/backend/app)       |
|   +-- Auth & RBAC (JWT + Bcrypt, Farmer & Admin roles)                  |
|   +-- PredictionOrchestrator (VRAM-safe lazy/sequential GPU inference)  |
|   |     +-- Model A: EfficientNet-B0 (97.82% acc, active)               |
|   |     +-- Model B: YOLO Detector (active if checkpoint, else HONEST   |
|   |     |            UNAVAILABLE)                                       |
|   |     +-- Model C: U-Net 3-Class (active if checkpoint, else HONEST   |
|   |     |            UNAVAILABLE)                                       |
|   |     +-- Model D: Grad-CAM (features[-1] saliency aid)               |
|   +-- SeverityEngine (Diseased / Total Leaf * 100, excludes background) |
|   +-- RecommendationEngine (ICAR-CICR offline JSON advisories)          |
|   +-- ReportService (ReportLab PDF + visual assets)                     |
|   +-- SQLAlchemy ORM -> SQLite (cotton_leaf_ai.db) / PostgreSQL         |
+-------------------------------------------------------------------------+
```

---

## 4. Phased Implementation Plan

- **Phase 1 (Complete):** Read-only project audit, inventory, architecture & gap analysis.
- **Phase 2:** WSL 2 Python 3.11 + PyTorch CUDA user-space environment setup (`~/.venvs/cotton_leaf_ai`) + GPU tensor & checkpoint verification + Windows Node.js LTS setup for Vite.
- **Phase 3 & 5:** Database schema migration (preserving existing users & 14 predictions), complete FastAPI REST routes (`/api/classification/predict`, `/api/detection/predict`, `/api/segmentation/predict`, `/api/severity/calculate`, `/api/history`, `/api/admin/*`, upgraded `/api/health`, `/api/predictions/analyze`, `/api/reports/{id}`), 256x256 U-Net transform fix, VRAM management & CPU fallback.
- **Phase 6:** Inspect CVAT status & `Cotton_Leaf_Segmentation_Pilot`; implement CVAT-to-UNet 3-class mask converter (`Leaf - Diseased_Region`), mask quality visualizer, group-aware split leakage check, CVAT-to-YOLO bbox exporter, and dataset annotation expansion strategy.
- **Phase 7 & 8:** Implement U-Net evaluation script + YOLO detector architecture, licensing audit report, training & evaluation scripts (strictly gated on genuine annotated data — zero fabricated data).
- **Phase 9 & 10:** Upgrade Severity Engine (`healthy_pixels`, leaf-restricted calculation, provisional threshold controls) & unified 4-model orchestrator.
- **Phase 4 & 11:** Complete React + Vite + TypeScript + Tailwind + Recharts + Lucide frontend (`frontend/`) with all 8 views (Landing, Auth, Farmer Dashboard, Upload/Analyze, Unified Results Workspace, Filtered Paginated History + Detail View, Reports, Live Admin Dashboard), connected strictly to live backend APIs.
- **Phase 12 & 13:** Launch/stop/check PowerShell scripts (`start_project.ps1`, `stop_project.ps1`, `check_environment.ps1`), fix & expand automated backend/ML/frontend test suites, run full end-to-end verification in WSL GPU + Vite UI, and finalize documentation.
