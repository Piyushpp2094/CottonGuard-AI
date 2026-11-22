# Cotton Leaf AI — Complete Project File Inventory (Read-Only Audit)

**Audit Date:** 2026-11-11  
**Project Root:** `D:\Main_Project\cotton_leaf_ai` (`/mnt/d/Main_Project/cotton_leaf_ai`)  
**Dataset Root (Read-Only):** `D:\Main_Project\dataset\SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection`  
**Audit Scope:** Complete read-only inspection of workspace files, Python virtual environments, CUDA runtime, checkpoints, SQLite database, FastAPI backend, PyTorch ML modules, React/Vite frontend, and sibling parent-folder artifacts prior to any code generation or modification.

---

## 1. Environment & Model Readiness Snapshot

| Component | Verified State | Observations & Conflicts |
|---|---|---|
| **GPU Hardware** | `NVIDIA GeForce RTX 3050 6GB Laptop GPU` (`6144 MiB`, Driver `621.14`) | Detected cleanly on host via `nvidia-smi`. |
| **Windows Python (`.venv`)** | `Python 3.11.9`, `PyTorch 2.15.0+cu132`, `CUDA: True` | Direct CUDA tensor allocation on `cuda:0` succeeded during audit. Currently contains FastAPI/SQLAlchemy/ReportLab stack; lacks `pytest`, `opencv-python`, `flask`, `ultralytics`, `segmentation-models-pytorch`. Native Windows PATH has **no `node` / `npm`**. |
| **WSL 2 Ubuntu Venv (`~/.venvs/cotton_leaf_ai`)** | `Python 3.11.18`, `PyTorch 2.15.0+cu132`, `CUDA: True` | Configured via `uv` (`no pip module`). Includes FastAPI `0.162.3`, `uvicorn 0.54.0`, `SQLAlchemy 2.1.8`, `opencv-python-headless 5.0.0.93`, `pytest 9.2.0`, `httpx 0.28.1`. WSL Node.js `v22.23.4` + npm `10.9.9` at `~/.local/node/bin`. Does **not** have `flask`, `ultralytics`, or `segmentation-models-pytorch`. |
| **EfficientNet-B0 Classifier (`best_model.pt`)** | **READY & VERIFIED** (`48,685,496` bytes) | Trained on clean group-aware `SAR-CLD-2024` split (7 classes, `97.82%` accuracy / `0.9754` Macro F1 on 321 test images). |
| **Grad-CAM Explainability** | **READY & VERIFIED** | Implemented in `src/explainability/{gradcam,visualization}.py` using `features[-1]` of `CottonLeafClassifier`. |
| **YOLO Lesion Detector** | **CODE READY (CUSTOM & YOLOV8N), 50 VERIFIED LABELS (514 BOXES)** | Pure-PyTorch anchor-free detector & YOLOv8n gated wrapper. 50 published image-label pairs (514 bounding boxes) in `data/detection/`. Model remains strictly `NOT_TRAINED` awaiting explicit user training approval. |
| **U-Net Segmentation** | **CODE READY & TRAINED (EXPERIMENTAL), 50 VERIFIED MASKS** | 3-class segmentation architecture (UNet-ResNet34). Trained experimental checkpoint `best_unet.pt` (97,921,803 bytes, SHA-256 `612fca0a...`) saved in `artifacts/checkpoints/segmentation/`. Macro Dice `0.6845` on 15 validation samples. Status: `EXPERIMENTAL` (not production validated). |
| **SQLite Database (`cotton_leaf_ai.db`)** | **LIVE & POPULATED** (`262,144` bytes) | 7 tables: `users` (5), `predictions` (35), `prediction_results` (35), `detection_results` (20), `segmentation_results` (20), `severity_results` (20), `reports` (5). |

---

## 2. Existing Files and Folders

### A. Root & Governance Files (`D:\Main_Project\cotton_leaf_ai\`)
- `.gitignore` — Ignores `.venv/`, `__pycache__/`, `.env`, `artifacts/**`, `*.pt`, `*.pth`, `*.onnx`.
- `pyproject.toml` — Requires Python `>=3.11,<3.12`; configures pytest paths.
- `requirements.txt` — Minimal CPU Phase 1 audit packages (`numpy`, `pandas`, `Pillow`, `ImageHash`, `PyYAML`, `tqdm`).
- `requirements_wsl.txt` — Full WSL 2 FastAPI + CV + PDF + Pytest dependencies (`torch`/`torchvision` noted via CUDA wheel index).
- `check_environment.ps1`, `start_project.ps1`, `stop_project.ps1` — PowerShell orchestration targeting WSL 2 CUDA backend (`:8000`) + WSL Node Vite frontend (`:5173`).
- `cotton_leaf_ai.db` — Active SQLite database with 7 tables and existing user/prediction records.
- `PROJECT_PROGRESS.md` — Master implementation tracker documenting FastAPI + TS React phases.
- `README.md` — Project overview and WSL 2 + FastAPI + Vite quick-start guide.
- `PROJECT_FILE_INVENTORY.md` *(created in audit)* — This inventory.
- `FILE_CHANGE_LOG.md` *(created in audit)* — Persistent change tracking log.

### B. Configuration & Documentation (`configs/`, `docs/`)
- `configs/classes.json`, `artifacts/manifests/class_names.json` — Verified 7 classification labels aligned to `best_model.pt`.
- `configs/detection_classes.json`, `configs/yolo_config.yaml` — 6 symptomatic disease classes (permissive custom detector + Ultralytics YOLOv8n RTX 3050 config).
- `configs/recommendations.json`, `configs/settings.yaml`, `configs/severity_rules.yaml` (6-tier provisional scale, `category_validated: false`).
- `artifacts/metrics/benchmark_report.json` — Measured GPU validation benchmark & untrained footprint report.
- `docs/` (`api_documentation.md`, `architecture.md`, `cvat_annotation_strategy.md`, `dataset_documentation.md`, `license_audit.md`, `model_documentation.md`, `project_audit_and_plan.md`, `segmentation_annotation_guide.md`, `user_manual.md`).

### C. Core ML, Explainability & Metrics (`src/`)
- `src/models/classifier.py` (`CottonLeafClassifier`, EfficientNet-B0 + Dropout `0.3` + 7-class Linear head), `src/models/model_factory.py` (`build_classifier`).
- `src/models/detector.py` (`CottonLeafYOLODetector`, pure-PyTorch BSD-3/Apache-2.0) & `src/models/yolov8_detector.py` (`YOLOv8DiseaseDetector`, gated `NOT_TRAINED` wrapper).
- `src/models/unet.py` (`CottonLeafUNet`, 3-class segmentation) & `src/models/unet_resnet34.py` (`UNetResNet34Segmentor`, SMP `ResNet-34`, `512x512`, Albumentations, weighted CE+Dice, gated `NOT_TRAINED`).
- `src/explainability/gradcam.py`, `src/explainability/visualization.py` — Heatmap, overlay, 3-panel comparison, and JSON metadata generator.
- `src/metrics/area_calculation.py`, `src/metrics/severity.py` — Leaf/diseased/healthy pixel counting & 6-tier `category_validated=false` severity evaluation.
- `src/recommendations/recommendation_engine.py` — Offline advisory matcher.
- `src/data/` (`dataset.py`, `preprocessing.py`, `dataloader.py`) — `224x224` ImageNet normalization transforms & WSL/Windows path resolver integration.
- `src/training/` (`trainer.py`, `losses.py`, `detection_losses.py`, `segmentation_losses.py`, `callbacks.py`).
- `src/utils/paths.py` — Cross-OS (`Windows` / `/mnt/<drive>/`) dataset path resolution (`resolve_platform_path`).

### D. CLI Scripts & Tests (`scripts/`, `tests/`)
- `scripts/`: `audit_dataset.py`, `benchmark_inference.py`, `create_admin.py`, `create_manifest.py`, `evaluate_classifier.py`, `evaluate_segmentation.py`, `evaluate_yolo.py`, `inspect_cvat.py`, `prepare_cvat_annotations.py`, `review_duplicates.py`, `run_backend_wsl.sh`, `run_frontend_wsl.sh`, `test_api_endpoints.py`, `test_gradcam.py`, `train_classifier.py`, `train_segmentation.py`, `train_unet_resnet34.py`, `train_yolo.py`, `train_yolov8.py`, `validate_annotations.py`, `verify_checkpoint_provenance.py`, `verify_dataset_leakage.py`.
- `tests/`: `conftest.py`, `test_backend_api.py`, `test_classifier.py`, `test_data_pipeline.py`, `test_segmentation_severity_recommendations.py`, `test_yolo_and_cvat.py`, `tests/backend/test_phase3_backend.py`, and `tests/models/test_phase4_ai_integration.py`.

### E. Backend Application & Migrations (`backend/app/`, `alembic/`) — FastAPI
- `.env` (git-ignored local config), `.env.example`, `alembic.ini`, `alembic/{env.py, script.py.mako, versions/0001_baseline_core_tables.py, versions/b4ca4e2a1471_add_phase3_refresh_revoked_jobs.py}`.
- `backend/app/main.py` — FastAPI app, CORS, static artifact mounts, structured error handlers, router registration, `/api/health`, `/api/health/models`, `/` fallback.
- `backend/app/config.py` — Typed `pydantic-settings` config (`Settings`, `get_settings`).
- `backend/app/database.py`, `models.py`, `schemas.py`, `auth.py` — SQLAlchemy ORM (11 tables incl. `AuditLog`, `RefreshToken`, `RevokedToken`, `PredictionJob`), Pydantic v2 schemas, HS256 JWT access/refresh rotation & revocation + bcrypt + `require_admin`.
- `backend/app/utils/image_validation.py` — Magic-byte MIME, size (`<= 10 MB`), corruption & dimension (`100..8000 px`) validation.
- `backend/app/routes/`: `authentication.py` (`/register`, `/login`, `/refresh`, `/logout`, `/me`), `predictions.py` (`/analyze`, `/analyze/async`, `/analyze/stream/{job_id}`, `/jobs/{job_id}`, `/{id}`), `ai_models.py`, `explainability.py`, `history.py`, `dashboard.py`, `reports.py`, `admin.py`.
- `backend/app/services/`: `audit_service.py`, `model_registry.py` (`artifacts/model_registry.json`), `job_service.py` (SSE pub-sub worker), `prediction_service.py` (GPU lock + CUDA OOM CPU fallback + 7-stage pipeline), `report_service.py` (ReportLab PDF builder).
- `backend/app/static/index.html` — Older single-file fallback SPA (`51,221` bytes, `2026-11-07`).
- `artifacts/backups/cotton_leaf_ai_pre_phase3.db` — Verified pre-Alembic SQLite backup (`294,912` bytes).
- `artifacts/model_registry.json` — Synchronized model status registry (`classifier=trained`, `detector=not_trained`, `segmentor=not_trained`).

### F. Frontend Application (`frontend/`) — Currently React 18 + Vite 5 + TypeScript
- `frontend/package.json`, `package-lock.json`, `vite.config.ts`, `tsconfig.json`, `tailwind.config.js`, `postcss.config.js`, `index.html`.
- `frontend/src/`: `main.tsx`, `App.tsx`, `index.css`, `vite-env.d.ts`, `context/AuthContext.tsx`, `services/api.ts`, `components/{Navbar.tsx, PredictionResultsView.tsx}`, `pages/{LandingPage, AuthPage, DashboardPage, AnalyzePage, HistoryPage, AdminPage}.tsx`.
- `frontend/node_modules/` (installed) & `frontend/dist/` (production build present).

### G. Data & Artifacts (`data/`, `artifacts/`)
- `data/annotations/{images,masks}` — Published combined dataset (`50` image-mask pairs, 3-class pixel masks); `data/detection/{images,labels}` — Published combined dataset (`50` images, `50` YOLO label files, `514` bounding boxes, `detection_manifest.csv`).
- `data/pilot_batch_01/images/` (`21` images) and `data/pilot_batch_02/images/` (`29` images).
- `data/cvat_exports/` (`task4_pilot_all21.zip`, `task5_pilot_batch02.zip`).
- `artifacts/manifests/` — Includes `pilot_batch_manifest.csv` (21), `batch_02_manifest.csv` (29), `combined_pilot_manifest.csv` (50), `class_distribution_report.json`, `group_overlap_report.json`, and `phase6_annotation_quality_report.json`.
- `artifacts/checkpoints/classification/{best_model.pt, last_model.pt}` — Trained EfficientNet-B0 weights (`SHA-256: 552baba8...` preserved).
- `artifacts/manifests/` & `artifacts/audit/` & `artifacts/metrics/` — Audited splits (`1492 train / 319 val / 321 test`), provenance & leakage verification reports, test metrics & plots.
- `artifacts/{explainability,predictions,reports,logs}/` — Generated Grad-CAM images, uploaded test images, ReportLab PDFs, and WSL logs.

### H. Parent Directory Sibling Items (`D:\Main_Project\`)
- `D:\Main_Project\dataset\SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection\` (`Augmented Dataset/`, `Original Dataset/`).
- `D:\Main_Project\cvat_setup\cvat\` — Local CVAT repository (Docker Desktop daemon stopped).
- `D:\Main_Project\.venv\` — Earlier Phase 1 CPU audit Python environment (`numpy`, `pandas`, `ImageHash`, `PyYAML`, etc.).
- `D:\Main_Project\Cotton_Leaf_Disease_Area_Severity_Full_Pipeline.ipynb` — Standalone notebook outside modular workspace.
- `D:\Main_Project\folder_structure.py` — Standalone dataset folder printer script.
- `D:\Main_Project\cotton_leaf_ai.zip` — `2.44 GB` archive created `2026-11-10`.

---

## 3. Classification vs Target Specification (Depends on Confirmed Architecture)

> **CRITICAL DECISION REQUIRED:** The new specification requests a **Flask + JSX React + `ultralytics` + `segmentation-models-pytorch` + Windows `.venv`** architecture, whereas `cotton_leaf_ai/` already contains a tested **FastAPI + TypeScript React + Custom BSD-3 YOLO/U-Net + WSL 2/Windows CUDA** platform. Both tracks are mapped below so nothing is overwritten blindly.

### Files to Reuse Without Modification (Preserve Under ALL Options)
- `artifacts/checkpoints/classification/best_model.pt`, `last_model.pt` — Preserved verbatim (trained classifier weights).
- `artifacts/manifests/**/*`, `artifacts/audit/**/*`, `artifacts/metrics/**/*` — Verified non-leaked splits & audit evidence.
- `D:\Main_Project\dataset\SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection\**\*` — Read-only dataset source.
- `configs/classes.json`, `configs/recommendations.json`, `configs/severity_rules.yaml`, `configs/detection_classes.json` — Domain mappings & clinical rules.
- `src/models/classifier.py`, `src/models/model_factory.py`, `src/data/{dataset,preprocessing,dataloader}.py`, `src/explainability/{gradcam,visualization}.py`, `src/metrics/{area_calculation,severity}.py`, `src/recommendations/recommendation_engine.py`, `src/utils/paths.py`.
- `cotton_leaf_ai.db` — Must be preserved / backed up before any additive schema migration (`users`, `predictions`, `reports`, etc.).

### Files That Need Modification
- **If Option A (Recommended — Preserve & Extend Existing FastAPI + TypeScript React Stack):**
  - `backend/app/models.py`, `schemas.py`, `database.py` — Add optional `training_jobs`, `audit_logs`, and dataset telemetry support if missing from requested admin workflows.
  - `backend/app/routes/admin.py`, `dashboard.py`, `reports.py` — Expose any missing admin model-retrain job controls / audit log endpoints / CSV exports while preserving existing API contracts.
  - `frontend/src/components/PredictionResultsView.tsx`, `pages/AdminPage.tsx`, `pages/DashboardPage.tsx`, `services/api.ts` — Add interactive side-by-side overlay controls / dataset analytics / training job monitors if requested, keeping existing `.tsx` architecture.
  - `check_environment.ps1`, `start_project.ps1`, `README.md`, `PROJECT_PROGRESS.md` — Sync launch behavior (WSL 2 or Windows native `.venv` now that CUDA works on Windows too).
- **If Option B (Full Migration to Pasted Flask + JSX Spec):**
  - Nearly all of `backend/app/**`, `tests/test_backend_api.py`, `frontend/src/**`, `requirements.txt`, and model wrappers (`src/models/{unet,detector}.py`) would have to be replaced or dual-hosted, creating high regression risk against the working FastAPI + TS platform.

### New Files Requested by Pasted Spec (Missing Currently — Pending Confirmation)
- **Dataset & Weight Targets:**
  - `data/detection/` or CVAT exports (`images/`, `labels/`) + `artifacts/checkpoints/detection/best_yolo.pt` (or `backend/weights/best_yolo.pt`)
  - `data/annotations/` populated pairs + `artifacts/checkpoints/segmentation/best_unet.pt` (or `backend/weights/best_unet.pt`)
- **If Option B (Flask + JSX structure) is explicitly chosen:**
  - `backend/run.py`, `backend/requirements.txt`, `backend/.env`, `backend/app/config.py`, `backend/app/extensions.py`
  - `backend/app/models/{user,prediction,training_job,audit_log}.py`
  - `backend/app/routes/{auth,farmer,admin,report,system}.py`
  - `frontend/src/{main.jsx,App.jsx,layouts/FarmerLayout.jsx,layouts/AdminLayout.jsx,...}`
  - `scripts/start.bat`
- **If Option A (Extend FastAPI + TS) is chosen:**
  - Only targeted additions (e.g., `backend/app/routes/training_jobs.py` / `audit_logs.py` or additive extensions in existing files + CVAT annotation dataset when supplied).

---

## 4. Potentially Unnecessary, Duplicate, or Outdated Files

1. `D:\Main_Project\cotton_leaf_ai\backend\app\static\index.html` (`51,221` bytes, modified `2026-11-07`) — Legacy single-file CDN HTML dashboard superseded by the compiled React + Vite app in `frontend/`.
2. `D:\Main_Project\.venv\` (created `2026-11-07 10:13:16`) — Parent-directory CPU audit virtualenv duplicating packages already present in `cotton_leaf_ai\.venv` and WSL `~/.venvs/cotton_leaf_ai`.
3. `D:\Main_Project\Cotton_Leaf_Disease_Area_Severity_Full_Pipeline.ipynb` (`132,399` bytes) — Early monolithic prototype notebook outside `cotton_leaf_ai/`.
4. `D:\Main_Project\folder_structure.py` (`762` bytes) — One-off helper script outside `cotton_leaf_ai/scripts/`.
5. `D:\Main_Project\cotton_leaf_ai.zip` (`2,440,974,526` bytes, created `2026-11-10`) — Snapshot archive backup consuming ~2.44 GB.
6. Test/smoke-run outputs inside `artifacts/{explainability,predictions,reports}/` (especially early `20261107_*` files and synthetic test JPEG/PDF records from prior API checks).

---

## 5. Files Proposed for Deletion (NOT DELETED — Awaiting Explicit User Confirmation)

Per strict cleanup policy, **zero files were deleted or moved during this audit**. Removal requires explicit confirmation item-by-item:

1. **`D:\Main_Project\cotton_leaf_ai\backend\app\static\index.html`**
   - **Purpose:** Original single-file CDN HTML SPA served at `GET /`.
   - **Why redundant/outdated:** Superseded by `frontend/` React + Vite + Tailwind app (`frontend/dist/`). Keeping two separate UIs creates UX drift and confusion on `:8000` vs `:5173`.
   - **Risk of keeping:** Low; only confuses users hitting `127.0.0.1:8000/` directly instead of `localhost:5173`.
   - **Safe to remove?** **Yes**, once confirmed that `frontend/` (`:5173` or serving `frontend/dist`) is the sole UI. (Alternatively `GET /` already falls back to platform metadata JSON when `index.html` is absent.)

2. **`D:\Main_Project\.venv\`** *(parent directory venv only — NOT `D:\Main_Project\cotton_leaf_ai\.venv`)*
   - **Purpose:** Early CPU audit virtualenv created before configuring project-scoped `.venv` and WSL venv.
   - **Why redundant/outdated:** Unused by all scripts (`cotton_leaf_ai\.venv` and `~/.venvs/cotton_leaf_ai` supersede it).
   - **Risk of keeping:** Confusion over which Windows `.venv` is active.
   - **Safe to remove?** **Yes**, provided no external user command relies on `D:\Main_Project\.venv`.

3. **`D:\Main_Project\cotton_leaf_ai\artifacts\{explainability,predictions,reports}\` stale `20261107_*` test outputs** *(optional workspace hygiene — keep `best_model.pt`, `manifests/`, `audit/`, `metrics/` untouched)*
   - **Purpose:** Generated test images (`1412`-byte dummy JPEGs) and smoke-test PDFs (`cotton_health_report_2..10.pdf`) created during early API debugging.
   - **Why redundant:** Clutter directory listings and include broken/early test files while referencing deleted or test predictions.
   - **Risk of keeping:** Low (small disk usage, though history rows in `cotton_leaf_ai.db` may reference some filenames!).
   - **Safe to remove?** **Conditional** — safest to keep DB-referenced artifacts or prune ONLY orphaned files after DB verification.

4. **`D:\Main_Project\cotton_leaf_ai.zip`** (`~2.44 GB` backup archive)
   - **Purpose:** Prior manual/session backup archive of the project.
   - **Why redundant:** Duplicate of on-disk `cotton_leaf_ai/`.
   - **Risk of keeping:** Consumes `2.44 GB` disk space.
   - **Safe to remove?** **Only if you no longer need this Nov 10 restore point** (recommend keeping until architectural decisions and new changes are verified).
