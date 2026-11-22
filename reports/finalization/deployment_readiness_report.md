# CottonGuard AI — Phase 13: Setup, Run Scripts & Deployment Readiness Report

**Document Version:** 1.0.0  
**Execution Date:** 2026-11-19  
**Workspace:** `D:\Main_Project\cotton_leaf_ai`  
**Target Environment:** Local Windows 11 Workstation (NVIDIA GeForce RTX 3050 6GB Laptop GPU, Python 3.11.9, FastAPI)  
**Verification Script:** `scripts/verify_environment.py`  
**Demonstration Script:** `run_demo.bat`  

---

## 1. Executive Summary

Phase 13 establishes the operational deployment packaging, automated preflight tooling, and execution scripts for **CottonGuard AI — Intelligent Cotton Leaf Disease Detection, Segmentation, and Severity Estimation System**.

The platform is configured for one-click startup and local demonstration without external cloud dependencies. All static web assets, PyTorch checkpoints, database tables, and API routes are locally self-contained.

### Core Deliverables Built & Verified in Phase 13
1. **One-Click Startup Script (`run_demo.bat`):** Automates Python venv validation, runs the environment preflight verification, launches the default browser directly to the web app (`http://localhost:8000/app`), and initiates the FastAPI Uvicorn server on port 8000.
2. **Environment Preflight Tool (`scripts/verify_environment.py`):** Standalone diagnostic utility performing cryptographic hash matching, CUDA discovery, database connectivity, and asset presence checks.
3. **Master Project Documentation (`README.md`):** Comprehensively updated technical documentation containing architecture diagrams, multi-architecture benchmark results, quickstart guide, and links to college presentation deliverables.

---

## 2. Environment Verification Results

The automated preflight utility (`scripts/verify_environment.py`) was executed on the live environment:

```text
================================================================================
COTTONGUARD AI - ENVIRONMENT & DEPLOYMENT READINESS VERIFICATION
================================================================================
[PASS] Python Runtime: 3.11.9 (Requirement: >= 3.11.0)
[PASS] PyTorch / CUDA: 2.15.0+cu132 | GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU (6143.5 MB VRAM)

Verifying Core Deep Learning Checkpoints & SHA-256 Hashes:
  [PASS] Classifier (EfficientNet-B0) (48,685,496 bytes)
         Expected: 552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e
         Actual:   552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e
  [PASS] Segmentor (U-Net ResNet-34) (97,921,803 bytes)
         Expected: 612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43
         Actual:   612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43
  [PASS] Detector (Faster R-CNN V2) (345,797,103 bytes)
         Expected: 01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2
         Actual:   01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2

[PASS] Sealed Test Split: test.csv verified (321 samples preserved)
[PASS] SQLite Database: cotton_leaf_ai.db (622,592 bytes, 12 tables, 63 logged predictions)

[PASS] Frontend Static Dashboard: index.html (76,167 bytes)
[PASS] Preloaded Reference Gallery: samples.json

================================================================================
READINESS VERDICT: [100% PASS] — SYSTEM READY FOR DEMONSTRATION & SUBMISSION
================================================================================
```

---

## 3. Operational Access Points

When running `run_demo.bat`, the system binds to `0.0.0.0:8000` and exposes the following network interfaces:

| Interface | URL | Purpose |
| :--- | :--- | :--- |
| **Interactive Web Application** | `http://localhost:8000/app` | Primary farmer and demonstration interface (Upload, Preloaded Samples, Diagnostic Overlays, Model Hub, Presentation Tab). |
| **Platform Root** | `http://localhost:8000/` | Web application redirect and platform metadata JSON. |
| **Interactive Swagger Docs** | `http://localhost:8000/docs` | OpenAPI documentation for executing and inspecting all REST endpoints. |
| **Alternative ReDoc** | `http://localhost:8000/redoc` | Clean technical API reference. |
| **Live Telemetry Endpoint** | `http://localhost:8000/api/health` | Real-time JSON health check providing GPU device name, allocated VRAM, and model readiness status. |
| **Model Registry Telemetry** | `http://localhost:8000/api/health/models`| Detailed parameter counts, model statuses, and architectural specifications. |

---

## 4. Evaluation Day Demonstration Procedure

For the final academic project defense and viva voce presentation:
1. Ensure the workstation is plugged into AC power (ensuring the NVIDIA RTX 3050 operates at its full 50W TGP power profile).
2. Double-click `run_demo.bat` in the root folder `D:\Main_Project\cotton_leaf_ai`.
3. The command console will display the green preflight check and automatically open Chrome or Edge to `http://localhost:8000/app`.
4. In the web interface:
   - Click the **"AI Models"** tab to demonstrate the multi-architecture benchmark table and model parameter breakdown.
   - Click the **"Project & Team"** tab to introduce the capstone project scope, engineering highlights, and team credentials.
   - Navigate to the **"Analyze"** tab and demonstrate real-time inference using the **Preloaded Reference Leaf Gallery** (instant one-click testing of Bacterial Blight, Curl Virus, Healthy Leaf, etc.).
   - Demonstrate the **Multi-Layer Diagnostic Viewer** by toggling between Original, Grad-CAM Heatmap, Leaf Mask, Lesion Mask, and Composite layers.
   - Click **"Download Official Diagnostic Report (PDF)"** to showcase the instant 2-page clinical certificate compiled dynamically by ReportLab.

---

## 5. Phase 13 Sign-Off

Phase 13 (Setup, Run Scripts & Deployment Readiness) is fully executed and verified. The system is 100% operational, robust, and presentation-ready.

- **Status:** **COMPLETED**
- **Next Phase:** Phase 14 — Final Project Completion Report (Master Project Synthesis & Sign-Off).
