# CottonGuard AI — Local Application Recovery and Startup Report

**Date:** 2026-11-17  
**Project:** CottonGuard AI (`cotton_leaf_ai`)  
**Phase:** Local Application Recovery & Service Startup  
**Author:** Autonomous Senior AI/ML & DevOps Engineer  
**Status:** **ALL SERVICES RECOVERED & OPERATIONAL**  
**Approval Gate 4 Status:** Strictly **`PENDING_APPROVAL`** (Awaiting explicit user authorization before model training).

---

## 1. Executive Summary

Following a computer reboot, the local CottonGuard AI development servers were offline because they are on-demand processes rather than persistent system daemons. CVAT remained functional on port 8080 due to Docker Desktop's container restart policy.

Both the FastAPI backend and React/Vite frontend have been safely recovered and verified:
- **FastAPI AI Backend (Port 8000):** **HEALTHY & OPERATIONAL** (CUDA GPU acceleration active, database connected).
- **React + Vite Frontend (Port 5173):** **HEALTHY & OPERATIONAL** (Serving UI with dynamic backend API proxying).
- **CVAT Annotation Platform (Port 8080):** **HEALTHY & UNTOUCHED** (Zero Docker restarts or task mutations).
- **Frontend-to-Backend Communication:** **VERIFIED (PASS)** via Vite proxy route `http://127.0.0.1:5173/api/health`.

---

## 2. Root Cause Analysis of Unavailable Services

| Service | Port | Pre-Recovery Status | Root Cause |
| :--- | :---: | :---: | :--- |
| **FastAPI Backend** | 8000 | `ERR_CONNECTION_REFUSED` | The FastAPI/Uvicorn server was terminated upon computer reboot. It is an on-demand process that was not restarted after system restart. |
| **React + Vite UI** | 5173 | `ERR_CONNECTION_REFUSED` | The Vite frontend development server was terminated upon computer reboot. Node.js is hosted in the WSL Ubuntu environment and required launching via the project startup script. |
| **CVAT Platform** | 8080 | `HTTP 200 OK` (Operational) | Docker Desktop automatically restarted the CVAT container stack upon Windows login due to container restart policies. |

---

## 3. Node.js & Environment Inspection

### 3.1 Host vs. WSL Environment Mapping
- **Native Windows Host:** Node.js and npm are **NOT installed** in the Windows host PATH.
- **WSL 2 Ubuntu Environment:** Node.js and npm **ARE installed** at `$HOME/.local/node/bin`:
  - `Node.js`: v20+ / npm 10+
  - Location: `/home/ubuntu/.local/node/bin/node`
- **Frontend Dependencies:** Pre-existing in `frontend/node_modules/` (Vite v5.4.21, React 18.3.1, TailwindCSS 3.4.4). **Zero dependencies were reinstalled or modified.**

---

## 4. Exact Startup Commands & Architectural Topology

To preserve separation of concerns and allow independent debugging, the services are maintained as separate processes:

### 4.1 Backend Startup Command (Native Windows Python Virtual Environment)
Executed in `D:\Main_Project\cotton_leaf_ai`:
```powershell
& "D:\Main_Project\cotton_leaf_ai\.venv\Scripts\python.exe" -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
- **Process ID:** PID 11524 (Background Daemon)
- **Log Location:** `artifacts/logs/backend.log` (and task log `task-9210.log`)
- **Listening Address:** `http://0.0.0.0:8000`

### 4.2 Frontend Startup Command (WSL Ubuntu Project Script)
Executed via WSL Ubuntu targeting `cotton_leaf_ai/frontend`:
```powershell
wsl -d Ubuntu -- bash /mnt/d/Main_Project/cotton_leaf_ai/scripts/run_frontend_wsl.sh
```
- **Proxy Configuration:** `VITE_PROXY_TARGET` dynamically resolves to the Windows host gateway IP (`172.28.16.1:8000`), routing `/api` and `/artifacts` requests directly to the native Windows backend.
- **Process ID:** Background Daemon (`task-9236`)
- **Log Location:** `artifacts/logs/frontend.log`
- **Listening Address:** `http://0.0.0.0:5173`

---

## 5. End-to-End Health & Verification Results

All endpoints were probed and verified:

```text
============================================================
COTTONGUARD AI — LOCAL APPLICATION HEALTH & CONNECTIVITY PROBE
============================================================

1. Backend Direct Probe (http://127.0.0.1:8000/api/health):
   HTTP Status: 200 OK
   Platform Status: healthy
   Database Connected: True
   CUDA Available: True
   Device Name: NVIDIA GeForce RTX 3050 6GB Laptop GPU
   Classifier Checkpoint: True (best_model.pt)
   Detection Checkpoint: True (best_faster_rcnn.pt)
   Segmentation Checkpoint: True (best_unet.pt)

2. Frontend Direct Probe (http://127.0.0.1:5173):
   HTTP Status: 200 OK
   HTML Title: CottonGuard AI — Leaf Disease Analysis & Severity Estimation
   UI Assets: Vite SPA bundle loading correctly

3. Frontend-to-Backend Proxy Probe (http://127.0.0.1:5173/api/health):
   HTTP Status: 200 OK
   Server Header: uvicorn (proxied through Vite)
   Response: 100% identical to backend telemetry

4. CVAT Direct Probe (http://localhost:8080):
   HTTP Status: 200 OK
   UI Status: Computer Vision Annotation Tool UI active
============================================================
```

---

## 6. Strict Governance & Preservation Audit

- **Production Database:** Verified completely unmodified (`cotton_leaf_ai.db`, 348,160 bytes, SHA-256 `90aef2b9...`, last modified `2026-11-12 13:05:28`).
- **Model Checkpoints:** Verified bit-for-bit identical (`best_model.pt`, `best_faster_rcnn.pt`, `best_unet.pt`).
- **Dataset Manifests & Splits:** Completely untouched (`consolidated_100_annotation_manifest.csv`, `test.csv`).
- **CVAT Containers & Data:** 100% untouched; Docker Desktop remained undisturbed.
- **Model Training:** **Zero** training epochs or optimizer updates executed.

---

## 7. Operational URLs for Browser Access

You can now immediately access and use the platform in your browser:

- **Frontend User Interface:** [http://localhost:5173](http://localhost:5173)
- **Backend API Documentation (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Backend Health Telemetry:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- **CVAT Annotation Platform:** [http://localhost:8080](http://localhost:8080)

**Remaining Errors:** **NONE.** All services are online and communicating properly.  
**Approval Gate 4 Status:** Strictly **`PENDING_APPROVAL`**.
