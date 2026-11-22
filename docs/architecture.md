# Cotton Leaf AI — Complete System Architecture

**Project Title:** Leaf Disease Analysis and Severity Estimation Using U-Net and YOLO-Based Detection

---

## 1. Hybrid Local Execution Architecture (Windows 11 + Ubuntu WSL 2)

Due to Windows Smart App Control Code Integrity policies (Events 3033 & 3077 blocking `c10_cuda.dll`), the local runtime separates UI presentation and GPU AI compute safely without weakening Windows security or deleting the legacy Windows `.venv`:

```text
+-------------------------------------------------------------------------+
|                    WINDOWS 11 HOST (HP Victus Laptop)                   |
|  Browser:                                                               |
|   - Web Application UI:  http://localhost:5173 (React + Vite + Tailwind)|
|   - Swagger API Docs:    http://127.0.0.1:8000/docs                     |
|   - CVAT Annotation UI:  http://localhost:8080 (Separate Docker Stack)  |
+-------------------------------------------------------------------------+
                             | (WSL 2 Localhost Forwarding)
                             v
+-------------------------------------------------------------------------+
|               UBUNTU 26.04 WSL 2 (NVIDIA RTX 3050 6GB GPU)              |
|  Linux Environment: ~/.venvs/cotton_leaf_ai (Python 3.11 + PyTorch cu132|
|                                                                         |
|  FastAPI Backend (/mnt/d/Main_Project/cotton_leaf_ai/backend/app)       |
|   +-- Security & Auth (HS256 JWT + Bcrypt + Farmer/Admin RBAC)          |
|   +-- PredictionService (VRAM-aware Inference Orchestrator)             |
|   |     +-- Model A: EfficientNet-B0 Classifier (Active - 97.82% acc)   |
|   |     +-- Model B: Anchor-Free YOLO Detector (BSD-3/Apache-2.0)       |
|   |     +-- Model C: 3-Class U-Net Segmenter (256x256 transform)        |
|   |     +-- Model D: Grad-CAM Saliency Hook Engine (features[-1])       |
|   +-- SeverityEstimator (Diseased / Leaf * 100; provisional gates)      |
|   +-- RecommendationEngine (Offline ICAR-CICR JSON Knowledge Base)      |
|   +-- ReportService (ReportLab PDF with imagery & disclaimer)           |
|   +-- SQLAlchemy ORM -> cotton_leaf_ai.db (SQLite / PostgreSQL)         |
+-------------------------------------------------------------------------+
```

---

## 2. AI Model Components & Gating Rules

1. **Classification (`src/models/classifier.py`, `best_model.pt`):**
   - Backbone: `torchvision.models.efficientnet_b0`, Dropout ($0.3$), 7-class linear head.
   - Verified independent test accuracy: **97.82%** (314/321 correct), Macro F1: **0.9754**.
2. **YOLO Object Detection (`src/models/detector.py`, `src/training/detection_losses.py`):**
   - Implemented in pure PyTorch (`BSD-3-Clause`) using CSP/C2f-style backbone, SPPF, FPN+PANet neck, decoupled anchor-free heads, CIoU loss, and `torchvision.ops.nms`. Avoids AGPL-3.0 `ultralytics`.
   - Remains strictly `UNAVAILABLE` until human-verified CVAT bounding-box annotations train `artifacts/checkpoints/detection/best_yolo.pt`.
3. **U-Net Semantic Segmentation (`src/models/unet.py`):**
   - Encoder-decoder with skip connections predicting 3 pixel classes: Background (`0`), Healthy Leaf (`1`), Diseased Tissue (`2`).
   - Uses dedicated 256×256 deterministic inference preprocessing and bilinear upsampling back to image resolution before pixel quantification.
   - Remains strictly `UNAVAILABLE` until human-verified CVAT pixel masks train `artifacts/checkpoints/segmentation/best_unet.pt`.
4. **Grad-CAM Explainability (`src/explainability/gradcam.py`, `visualization.py`):**
   - Registers temporary forward/backward hooks on `backbone.features[-1]` (`Conv2dNormActivation`, $1280 \times 7 \times 7$), computes globally averaged gradient weights, applies ReLU, upsamples cleanly, removes hooks, and clears `.grad`.
   - Explicitly labeled as a saliency aid and **never** substituted for pixel segmentation or physical lesion area.

---

## 3. Database Schema (`backend/app/models.py`)

- `users`: credentials, roles (`farmer`, `admin`), ownership root.
- `predictions`: core classification result, area %, severity level, timestamps.
- `prediction_results`: Grad-CAM URLs, 7-class probability JSON, latency ms, ICAR-CICR advisories JSON, legacy overlay fields.
- `detection_results`: YOLO detection boxes JSON, count, annotated image path, model version, status.
- `segmentation_results`: leaf/healthy/diseased pixel counts, mask/overlay paths, model version, status.
- `severity_results`: severity grade, calibration flag (`thresholds_verified`), basis notes.
- `reports`: logged PDF report paths, sizes, and generation timestamps.
