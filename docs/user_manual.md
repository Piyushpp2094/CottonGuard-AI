# Cotton Leaf AI — User & Developer Manual

**Platform:** Leaf Disease Analysis and Severity Estimation Using U-Net and YOLO-Based Detection

---

## 1. Environment Architecture (Smart App Control + WSL 2 CUDA)

On this Windows 11 system, Windows Smart App Control prevents loading `torch\lib\c10_cuda.dll` in the legacy Windows `.venv`. To keep Windows security intact and still unlock full **NVIDIA GeForce RTX 3050 (6 GB VRAM)** acceleration:

- **Windows `.venv`** is preserved untouched.
- **Ubuntu WSL 2 Linux venv (`~/.venvs/cotton_leaf_ai`)** executes the FastAPI AI backend, PyTorch `2.15.0+cu132`, and all ML training/evaluation scripts with verified GPU acceleration (`cuda:0`).
- **Node.js v22 LTS (`~/.local/node`)** builds and serves the local React + Vite frontend (`http://localhost:5173`).
- **CVAT Community Edition (`http://localhost:8080`)** remains completely separate inside Docker Desktop with its own database.

---

## 2. Launching & Stopping Locally

Open Windows PowerShell inside `D:\Main_Project\cotton_leaf_ai`:

### Check Environment Readiness
```powershell
.\check_environment.ps1
```

### Start Backend & Frontend
```powershell
.\start_project.ps1
```
- **React + Vite Web App:** [http://localhost:5173](http://localhost:5173)
- **FastAPI Backend API:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Swagger OpenAPI Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **CVAT (Separate Docker Service):** [http://localhost:8080](http://localhost:8080)

### Stop Application Servers
```powershell
.\stop_project.ps1
```

---

## 3. Using the Web Platform

1. **Authentication:** Register a new farmer account directly via the web portal modal, or sign in using your pre-configured local credentials. To create or promote a local Administrator account:
   ```powershell
   .\.venv\Scripts\python.exe scripts/create_admin.py --username admin --email admin@cotton.local
   ```
2. **Analyze Leaf Page:** Drag and drop any cotton leaf photo (`.jpg`, `.png`, `.webp`, `.bmp`, max 15 MB). View:
   - **Classification & Probabilities:** Top disease prediction + 7-class softmax bar chart.
   - **Grad-CAM Explainability:** Interactive 3-panel view, overlay, and heatmap with explicit interpretability notice.
   - **YOLO Detection & U-Net Segmentation:** Active automatically when genuine trained checkpoints exist (`best_yolo.pt`, `best_unet.pt`); otherwise displays honest unavailable notices without fabricated outputs.
   - **Severity & ICAR-CICR Advisories:** 2D affected pixel ratio (when segmented), scouting actions, safety warnings, and **Download PDF Report**.
3. **History & Dashboard:** Filter past scans by disease, severity, search text, or date range; inspect full stored breakdowns; view live condition/trend charts.
4. **Admin Dashboard:** Sign in with an `admin` account to view global usage, model/GPU readiness, dataset split counts, and toggle user roles.

---

## 4. Running Automated Tests Inside WSL 2

```powershell
# Full unit & API pytest suite
wsl -d Ubuntu -- bash -c "cd /mnt/d/Main_Project/cotton_leaf_ai && ~/.venvs/cotton_leaf_ai/bin/python -m pytest -v"

# Grad-CAM explainability validation suite
wsl -d Ubuntu -- bash -c "cd /mnt/d/Main_Project/cotton_leaf_ai && ~/.venvs/cotton_leaf_ai/bin/python scripts/test_gradcam.py --samples 5"

# Frontend TypeScript check & Vite build
wsl -d Ubuntu -- env PATH=/home/hakers_pc/.local/node/bin:/usr/local/bin:/usr/bin:/bin npm --prefix /mnt/d/Main_Project/cotton_leaf_ai/frontend run build
```

---

## 5. CVAT Annotation & Model Training Workflow

1. Start Docker Desktop & CVAT (`http://localhost:8080`) independently.
2. Verify pilot schema inheritance:
   ```powershell
   wsl -d Ubuntu -- ~/.venvs/cotton_leaf_ai/bin/python /mnt/d/Main_Project/cotton_leaf_ai/scripts/inspect_cvat.py --username <cvat_user> --prompt-password
   ```
3. Export task in **CVAT for images 1.1** format and ingest:
   ```powershell
   wsl -d Ubuntu -- ~/.venvs/cotton_leaf_ai/bin/python /mnt/d/Main_Project/cotton_leaf_ai/scripts/prepare_cvat_annotations.py --input <cvat_export.zip>
   ```
4. Train U-Net & YOLO once sufficient (`>= 10`) verified non-test annotations are validated:
   ```powershell
   wsl -d Ubuntu -- ~/.venvs/cotton_leaf_ai/bin/python /mnt/d/Main_Project/cotton_leaf_ai/scripts/train_segmentation.py
   wsl -d Ubuntu -- ~/.venvs/cotton_leaf_ai/bin/python /mnt/d/Main_Project/cotton_leaf_ai/scripts/train_yolo.py
   ```
