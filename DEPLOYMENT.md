# CottonGuard AI — Production Deployment & Hosting Guide

**Project Title:** Leaf Disease Analysis and Estimation Using the U-Net and YOLO Based Detection  
**System Name:** CottonGuard AI  
**Release Version:** 2.1.0  

---

## 1. Executive Deployment Architecture

CottonGuard AI is a multi-modal computer vision and agricultural pathology analysis system. It integrates a 4-tier deep learning inference pipeline:
1. **Verification Gate:** EfficientNet-B0 Tri-Class verifier (cotton foliage vs. other plant foliage vs. non-plant background/objects).
2. **Disease Classifier:** EfficientNet-B0 4-class pathology model (Bacterial Blight, Curl Virus, Fusarium Wilt, Healthy).
3. **Semantic Segmentor:** U-Net with ResNet-34 backbone for spatial lesion segmentation and topological containment.
4. **Lesion Detector:** Faster R-CNN ResNet-50 FPN V2 for local bounding-box spot localization.

### Recommended Deployment Topologies

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ OPTION A: Unified Containerized Deployment (Recommended for Cloud/PaaS)     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Client Browser                                                             │
│       │                                                                     │
│       ▼ HTTPS (Port 443)                                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Docker Container (FastAPI ASGI • Uvicorn)                             │  │
│  │                                                                       │  │
│  │  ├── Static Frontend Serving (/app, /) ──► React 18 SPA Dashboard     │  │
│  │  ├── REST API (/api/v1/*)              ──► Pydantic Validation Gate   │  │
│  │  ├── Interactive Docs (/docs)          ──► OpenAPI / Swagger UI       │  │
│  │  ├── Health Telemetry (/api/health)    ──► Diagnostic Heartbeat       │  │
│  │  ├── PyTorch Inference Engine          ──► Co-loaded CPU Models       │  │
│  │  └── Database Layer                    ──► SQLite (or PostgreSQL)     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ OPTION B: Decoupled Multi-Service Deployment (High Concurrency)             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Frontend (Vercel / Netlify / Cloudflare Pages)                             │
│       │ (Static React 18 + Vite SPA Assets)                                 │
│       ▼ HTTPS API Calls with CORS                                           │
│  Backend API (Railway / Render / AWS ECS / GCP Cloud Run)                   │
│       │ (FastAPI ASGI + PyTorch CPU Pipeline)                               │
│       ▼ Relational Queries                                                  │
│  Database (Managed PostgreSQL: Neon / Supabase / AWS RDS)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Hardware, Memory & Free-Tier Reality Check

> [!WARNING]
> **Free-Tier Memory Constraints (OOM Protection):**  
> PyTorch with torchvision, OpenCV, and 4 concurrent deep learning models (co-loaded in memory) requires approximately **850 MB to 1.5 GB of RAM** during CPU inference.  
> Standard 512 MB free tiers (such as Render free web service or Fly.io free tier) **will fail with Out-Of-Memory (OOM / Exit Code 137)** during startup.

### Platform Resource Comparison

| Hosting Platform | Free Tier Specifications | CottonGuard AI Compatibility | Recommended Plan |
| :--- | :--- | :--- | :--- |
| **Hugging Face Spaces** | **2 vCPU, 16 GB RAM** (Docker/Gradio/FastAPI) | **100% Compatible (Optimal Free Choice)** | Free Community Hardware |
| **Railway** | $5 monthly credit (~512MB–8GB elastic RAM) | **Fully Compatible** | Developer Plan ($5/mo) |
| **Render** | 512 MB RAM (Free web service) | ⚠️ Incompatible on free tier (OOM on PyTorch) | Starter Tier ($7/mo, 2GB RAM) |
| **Google Cloud Run** | 2M requests free / 2GB–4GB configurable RAM | **Fully Compatible** | Pay-as-you-go |
| **AWS App Runner** | 1 vCPU, 2GB–4GB RAM | **Fully Compatible** | Pay-as-you-go |
| **Vercel / Netlify** | Frontend static hosting only | **Frontend Only** (Cannot run PyTorch backend) | Free Tier (Frontend only) |

---

## 3. Model Storage & Checkpoint Distribution

Because Git and GitHub enforce strict file size limits (50 MB warning, 100 MB hard reject), the deep learning checkpoints are stored securely and must be made available to the deployment environment:

### Authoritative Checkpoints & SHA-256 Manifest

| Pipeline Component | Checkpoint File | File Size | SHA-256 Cryptographic Checksum |
| :--- | :--- | :--- | :--- |
| **Verification Gate** | `artifacts/checkpoints/verifier/best_verifier.pt` | 48.6 MB | `1e16b9c68745c0f100679aa23e8993b9ee2ef6279851d4e4c820e36bb0a13828` |
| **Pathology Classifier**| `artifacts/checkpoints/classification/best_model.pt` | 48.7 MB | `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e` |
| **Lesion Segmentor** | `artifacts/checkpoints/segmentation/best_unet.pt` | 97.9 MB | `612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43` |
| **Spot Detector** | `artifacts/checkpoints/detection/best_faster_rcnn.pt` | 345.8 MB | `01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2` |

### Recommended Checkpoint Hosting Strategies:
1. **GitHub Releases:** Upload the checkpoint binaries as assets under a GitHub Release tag (e.g., `v2.1.0-checkpoints`). Write a lightweight startup download script that fetches and verifies them via `scripts/verify_environment.py`.
2. **Hugging Face Model Hub:** Host weights in a private/public repository (e.g., `hf.co/username/cottonguard-checkpoints`) and download via `huggingface_hub` during container build.
3. **Cloud Object Storage (S3 / Cloudflare R2 / GCS):** Download checkpoints via signed URLs during deployment bootstrap.

---

## 4. Production Environment Configuration

Create a `.env` file in the project root based on `.env.example`:

```env
# Application Identity & Mode
APP_NAME="CottonGuard AI"
APP_VERSION="2.1.0"
ENVIRONMENT="production"

# Security & Cryptography (Generate via: openssl rand -hex 32)
SECRET_KEY="REPLACE_WITH_A_64_CHAR_CRYPTOGRAPHICALLY_SECURE_RANDOM_SECRET"
JWT_ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Cookie Security (Must be true in production HTTPS)
COOKIE_SECURE=true
COOKIE_SAMESITE="strict"

# Database Connection (SQLite or PostgreSQL)
# Single container with persistent disk:
DATABASE_URL="sqlite:///./cotton_leaf_ai.db"
# Managed PostgreSQL (e.g. Supabase, Neon, AWS RDS):
# DATABASE_URL="postgresql://user:password@ep-host.region.aws.neon.tech/cotton_guard?sslmode=require"

# File Upload & Safety Constraints
MAX_UPLOAD_SIZE_MB=10
MIN_IMAGE_DIM=100
MAX_IMAGE_DIM=8000
UPLOAD_DIR="./artifacts/predictions"
ARTIFACTS_DIR="./artifacts"

# Real-time SSE Stream Timeout
SSE_TIMEOUT_SECONDS=180
```

---

## 5. Deployment Blueprint: Hugging Face Spaces (Optimal Free Hosting)

Hugging Face Spaces provides **2 vCPU and 16 GB RAM** for free, making it the ideal host for CPU PyTorch inference.

### Step 1: Create Space
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Select **Docker** as the Space SDK.
3. Choose **Blank** template and set license to **Apache 2.0** or **MIT**.

### Step 2: Use the Production Dockerfile
The project includes a ready-to-run container specification:
```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## 6. Deployment Blueprint: Railway / Render (PaaS)

### Railway Deployment (Docker / Procfile)
1. Link your GitHub repository in Railway.
2. Add a persistent volume mounted at `/app/artifacts` and `/app/data` to preserve database logs and predictions across redeploys.
3. Set Environment Variables (`SECRET_KEY`, `COOKIE_SECURE=true`, `ENVIRONMENT=production`).
4. Set Start Command:
   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT --workers 2
   ```

### Render Deployment
1. Create a **Web Service** pointing to the GitHub repository.
2. Select **Starter** plan (minimum 2 GB RAM).
3. Set Build Command:
   ```bash
   pip install -r requirements.txt
   ```
4. Set Start Command:
   ```bash
   uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
   ```

---

## 7. Health Checks & Telemetry

Once deployed, verify the system health via:
- **Telemetry Check:** `GET https://your-domain.com/api/health`
  ```json
  {
    "status": "ok",
    "timestamp": "2026-11-22T10:30:00Z",
    "version": "2.1.0",
    "environment": "production",
    "database": "connected",
    "models_loaded": {
      "verifier": true,
      "classifier": true,
      "segmentor": true,
      "detector": true
    }
  }
  ```
- **API Documentation:** `GET https://your-domain.com/docs`
- **Application Dashboard:** `GET https://your-domain.com/app`
