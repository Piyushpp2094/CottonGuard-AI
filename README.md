# Leaf Disease Analysis and Estimation Using the U-Net and YOLO Based Detection

**System Brand:** CottonGuard AI  
**Academic Degree:** Bachelor of Technology in Computer Science & Engineering (AI & ML)  
**System Status:** Completed Academic Production Prototype (Ready for Evaluation & Hosting)  

[![Python 3.11](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://www.python.org/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-2.1.0%2Bcu121-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-blue.svg)](https://react.dev/)
[![Tests Passed](https://img.shields.io/badge/Tests-96%2F96%20PASSED%20(100%25)-brightgreen.svg)]()
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%203050%206GB-76B900.svg)]()
[![License](https://img.shields.io/badge/License-Academic%20B.Tech%20Capstone-lightgrey.svg)]()

---

## 1. Project Overview

**CottonGuard AI** is an end-to-end precision agriculture and computer vision system developed to diagnose, segment, quantify, and generate agronomic treatment protocols for cotton foliar diseases. 

Modern deep learning classifiers frequently fail in real-world agricultural conditions by attempting to predict disease classes on non-plant backgrounds, human hands, or arbitrary crop foliage. CottonGuard AI solves this fundamental flaw by implementing a **mandatory 2-stage verification gate** before executing disease analysis:
1. **Physical Quality Verification:** Enforces resolution bounds, blur variance (Laplacian sharpness), and exposure/contrast checks.
2. **Domain Foliage Verification:** An EfficientNet-B0 Tri-Class verifier differentiates authentic cotton foliage from other plant species (tomato, potato, apple) and non-plant entities (people, text documents, vehicles) with zero false acceptance rate.
3. **Multi-Task Pathology Analysis:** Confirmed cotton leaves proceed simultaneously through 4-class disease classification (Bacterial Blight, Curl Virus, Fusarium Wilt, Healthy), U-Net spatial segmentation with topological boundary containment, Faster R-CNN lesion detection, Grad-CAM visual attention explainability, geometric severity percentage calculation, and automated ICAR-CICR advisory generation with ReportLab clinical 2-page PDF export.

---

## 2. Key Implemented Features

- **Mandatory Cotton Verification Gate:** Prevents hallucinations on non-cotton or out-of-domain images.
- **Deep Transfer Learning Classification:** EfficientNet-B0 achieving 98.43% validation accuracy across 4 canonical classes.
- **Organic Semantic Segmentation:** U-Net (ResNet-34 encoder) generating 3-class pixel masks (Background, Healthy Foliage, Diseased Lesion) with morphological hole-filling and leaf boundary clipping.
- **Scientific Severity Estimation:** Geometric pixel-ratio calculation: $\text{Severity \%} = \frac{\text{Diseased Pixels}}{\text{Healthy Pixels} + \text{Diseased Pixels}} \times 100$, mapped to 4 ICAR-CICR severity grades.
- **Explainability & Attribution:** Built-in Grad-CAM saliency heatmaps highlighting the exact pixel clusters influencing model predictions.
- **Object Detection (Lesion Bounding Boxes):** Faster R-CNN ResNet-50 FPN V2 for discrete spot localization.
- **Clinical 2-Page PDF Export:** Server-side PDF reports with cryptographic verification hashes, overlays, and farmer guidance.
- **Dual-Role Secure Authentication:** Role-based access control (Admin / Farmer) with bcrypt password hashing and JWT cookies.
- **Real-Time Progress Streaming:** Server-Sent Events (SSE) streaming multi-stage inference progress directly to the frontend.

---

## 3. Verified AI Model Architecture & Benchmark Summary

All models are trained, evaluated on local dedicated hardware (NVIDIA GeForce RTX 3050 6GB Laptop GPU over 30 trials), and tracked via cryptographic SHA-256 hashes:

| Component | Model Architecture | Parameters | Key Metric | GPU Latency | Peak VRAM | Status | Checkpoint Path |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Verification Gate** | EfficientNet-B0 Tri-Class | 4.02 M | **99.84% Test Acc**<br>0.9984 Macro F1<br>**0.00% FAR** ($\tau=0.70$)<br>1.0000 ROC-AUC | **13.82 ms** | ~32 MB | **PRODUCTION** | `artifacts/checkpoints/verifier/best_verifier.pt` |
| **Pathology Classifier**| EfficientNet-B0 (TorchVision) | 4.02 M | **98.43% Val Acc**<br>0.9839 Macro F1 | **14.94 ms** | ~32 MB | **PRODUCTION** | `artifacts/checkpoints/classification/best_model.pt` |
| **Lesion Segmentor** | U-Net (ResNet-34 Encoder) | 24.44 M | **0.8656 Leaf Dice**<br>0.8056 Leaf IoU<br>1.11% Healthy False Alarm | **93.09 ms** | ~97 MB | **PRODUCTION** | `artifacts/checkpoints/segmentation/best_unet.pt` |
| **Visual Explainability**| Grad-CAM (`features[-1]`) | — | Visual attention localization | **39.81 ms** | Context-shared | **PRODUCTION** | Built-in (`src/explainability/gradcam.py`) |
| **Spot Detector** | Faster R-CNN ResNet-50 FPN V2 | 43.70 M | 13.94% $mAP_{50}$ (Epoch 13) | 78.40 ms | ~174 MB | **EXPERIMENTAL** | `artifacts/checkpoints/detection/best_faster_rcnn.pt` |
| **Full AI Pipeline** | Unified Multi-Model Stack | 76.18 M | Complete 7-Stage Analysis | **158.40 ms** (~6.3 FPS) | **589.15 MB** | **CO-LOADED** | Concurrent in-memory serving |

- **Sealed Test Split:** `artifacts/manifests/splits/test.csv` (321 holdout images, SHA-256 tracked, 0% data leakage).
- **VRAM Headroom:** Co-loaded models consume only **557.15 MB** allocated (**834.00 MB** reserved, leaving $>5.1\text{ GB}$ free VRAM on a 6GB budget).
- **CPU Fallback Execution:** Automatic CPU fallback (**715.3 ms**) ensures reliable operation on machines without dedicated GPUs.

---

## 4. Multi-Stage Pipeline Architecture

```
                             Uploaded Leaf Photo (RGB)
                                         │
                     [Stage 1: Image Quality Validation]
              (Dimension Bounds • Blur Variance • Exposure • Contrast)
                                         │
                                      [PASS?]
                                      ├── NO ──► [REJECTED_QUALITY]
                                      │          (Halt downstream analysis • Display reason)
                                      ▼
               [Stage 2: Cotton Leaf Domain Verification]
            (EfficientNet-B0 Tri-Class: Cotton vs Other-Leaf vs Non-Plant)
                                         │
                        [Cotton Leaf Verified? (τ ≥ 0.70)]
                        ├── NO / UNCERTAIN ──► [REJECTED / UNCERTAIN]
                        │                     (Halt downstream analysis • Display guidance)
                        ▼ (YES: Confirmed Cotton Foliage)
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
[EfficientNet-B0 Classifier] [Faster R-CNN Detector]   [U-Net ResNet-34 Segmentor]
 98.43% Val Acc (14.9 ms)    13.94% mAP50 (78.4 ms)    3-Class Spatial Mask (93.1 ms)
          │                           │                           │
          ▼                           │                           ▼
 [Grad-CAM Saliency Map]              │               [Topological Containment]
 features[-1] (39.8 ms)               │               Hole Filling & Boundary Clipping
          │                           │                           │
          └───────────────────────────┼───────────────────────────┘
                                      ▼
                         [Geometric Severity Engine]
                Severity % = [Diseased / (Healthy + Diseased)] * 100
                      Mapped to ICAR-CICR Severity Tiers
                                      │
                                      ▼
                    [ICAR-CICR Agronomic Advisory Engine]
                  Chemical • Biological • Cultural Protocols
                                      │
                                      ▼
              [FastAPI Asynchronous Engine • SQLite Persistence]
                                      │
                                      ▼
                    [Interactive React 18 UI Dashboard]
                     [ReportLab Clinical 2-Page PDF]
```

---

## 5. Technology Stack

- **Deep Learning & Computer Vision:** PyTorch, TorchVision, OpenCV, SciPy, Pillow, NumPy.
- **Backend API & Web Server:** Python 3.11, FastAPI, Uvicorn, Pydantic v2, SSE-Starlette.
- **Database & Migration:** SQLite, SQLAlchemy ORM, Alembic.
- **Security & Authentication:** Passlib / Bcrypt, PyJWT (HMAC-SHA256 tokens).
- **Reporting:** ReportLab (vector PDF generation).
- **Frontend Dashboard:** React 18, Vite, TypeScript, Tailwind CSS, Lucide React, Recharts.
- **Testing:** Pytest, HTTPX, Node Test Runner.

---

## 6. Installation & Quickstart (Windows 11 First)

### Prerequisites
- Windows 11 (or 10) / Linux / macOS
- Python 3.11.x (verified on 3.11.9)
- Node.js 18+ (for frontend development; production static dashboard is pre-compiled)
- NVIDIA GPU with CUDA 12.1+ (optional; CPU fallback is fully supported)

### Option A: One-Command Demonstration Launcher (Recommended)
Double-click `run_demo.bat` or run in Command Prompt:
```cmd
run_demo.bat
```
This launcher automatically runs preflight verification, opens your default browser, and starts the FastAPI Uvicorn ASGI server on port 8000.

### Option B: Manual Setup
```powershell
# 1. Clone repository
git clone <YOUR_REPOSITORY_URL>
cd cotton_leaf_ai

# 2. Create and activate Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# (Optional: If using NVIDIA GPU, install CUDA-accelerated PyTorch)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Configure environment variables
cp .env.example .env

# 5. Run environment & checkpoint preflight verification
python scripts/verify_environment.py

# 6. Launch FastAPI server
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --log-level info
```

### Access Points
- **Web Application Dashboard:** [http://localhost:8000/app](http://localhost:8000/app)
- **Interactive Swagger REST API:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Telemetry Endpoint:** [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 7. Automated Testing & Verification

The test suite contains **96 automated unit, integration, and security tests**:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```
**Test Results:** **96 / 96 passed (100.0%)**
- `test_cotton_verification_pipeline.py`: Verifier rejection on non-plants, other crops, and quality anomalies.
- `test_api_contracts.py`: REST endpoint status codes, Pydantic serialization, and validation.
- `test_security.py`: Password hashing, JWT token expiry, magic byte file sanitization.
- `test_severity_calculation.py`: Geometric area formulas, zero-division protection, and edge cases.
- `test_checkpoints_integrity.py`: Cryptographic SHA-256 hash validation across all 4 production models.

---

## 8. Deployment & Cloud Hosting

Detailed production hosting guides, Docker configurations, and cloud provider assessments are documented in [`DEPLOYMENT.md`](file:///D:/Main_Project/cotton_leaf_ai/DEPLOYMENT.md).

- **Recommended Host:** Hugging Face Spaces (Docker Space with free 2 vCPU and 16 GB RAM) or Railway Developer Plan ($5/month).
- **RAM Requirement:** Minimum 2 GB RAM required for CPU co-loaded PyTorch inference (512 MB free tiers will OOM).

---

## 9. Limitations & Scientific Disclaimer

1. **Academic Decision Support:** CottonGuard AI provides decision-support analysis aligned with ICAR-CICR guidelines and does not replace accredited physical agronomic laboratory testing.
2. **Object Detection Status:** Faster R-CNN lesion detection is flagged as an **experimental** capability (13.94% $mAP_{50}$ on pilot annotations); semantic segmentation via U-Net (0.8656 Dice) is the primary authoritative lesion boundary tool.
3. **Severe Foliar Necrosis:** Extreme leaf decay (>85% necrotic coverage) may distort leaf boundary estimation during topological clipping.

---

## 10. College Deliverables Directory

| Deliverable Title | File Path | Focus Area |
|---|---|---|
| **Master Completion Report** | [`FINAL_PROJECT_COMPLETION_REPORT.md`](file:///D:/Main_Project/cotton_leaf_ai/FINAL_PROJECT_COMPLETION_REPORT.md) | Comprehensive major project sign-off |
| **Master Finalization Tracker** | [`MASTER_FINALIZATION_TRACKER.md`](file:///D:/Main_Project/cotton_leaf_ai/MASTER_FINALIZATION_TRACKER.md) | Evidence-backed phase tracker |
| **Forensic System Audit** | [`reports/finalization/final_forensic_audit.md`](file:///D:/Main_Project/cotton_leaf_ai/reports/finalization/final_forensic_audit.md) | Complete 16-component code audit |
| **Deployment Guide** | [`DEPLOYMENT.md`](file:///D:/Main_Project/cotton_leaf_ai/DEPLOYMENT.md) | Cloud hosting, Docker, and PaaS setup |
| **Final B.Tech Report** | [`docs/presentation/final_project_report.md`](file:///D:/Main_Project/cotton_leaf_ai/docs/presentation/final_project_report.md) | 13-section technical capstone report |
| **Presentation Slide Deck** | [`docs/presentation/college_presentation_slides.md`](file:///D:/Main_Project/cotton_leaf_ai/docs/presentation/college_presentation_slides.md) | 25-slide presentation with screenshot links |
| **Viva Voce Q&A Guide** | [`docs/presentation/viva_voce_qa_guide.md`](file:///D:/Main_Project/cotton_leaf_ai/docs/presentation/viva_voce_qa_guide.md) | 52 technical questions & answers |
| **Application Screenshots** | [`docs/presentation/final_screenshots/`](file:///D:/Main_Project/cotton_leaf_ai/docs/presentation/final_screenshots/) | High-resolution UI evidence screenshots |

---
*Developed as a Final Year B.Tech Major Project in Computer Science & Engineering.*  
*Title: Leaf Disease Analysis and Estimation Using the U-Net and YOLO Based Detection*
