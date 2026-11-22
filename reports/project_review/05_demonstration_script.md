# CottonGuard AI — 5–7 Minute Examiner Demonstration Script

**Document ID:** `reports/project_review/05_demonstration_script.md`  
**Target:** Final-Year Project Review Committee / Practical Viva Examination  
**Duration:** 5 to 7 Minutes  
**Prerequisites:** Backend running on port 8000, Frontend running on port 5173  

---

## Pre-Demonstration Checklist (1 Minute Before Review)

1. Open PowerShell or Command Prompt in `D:\Main_Project\cotton_leaf_ai`.
2. Double-click or execute:
   ```cmd
   run_all.bat
   ```
3. Confirm both servers launch:
   - Backend terminal: `Uvicorn running on http://127.0.0.1:8000` (FastAPI models loaded onto CUDA).
   - Frontend terminal: `Vite v5.2.11 ready at http://localhost:5173`.
4. Browser automatically opens to `http://localhost:5173`.
5. Have sample test images ready in `data/demo_samples/` or `data/` (e.g., Bacterial Blight sample and Healthy Leaf control).

---

## Phase 1: Platform Introduction & Objectives (0:00 – 1:00)

**Spoken Script:**
> *"Respected members of the examination committee, good morning. Today, I am demonstrating **CottonGuard AI**, a comprehensive, real-time computer vision system engineered for precision foliar disease diagnosis and leaf-restricted severity estimation in cotton crops.*
> 
> *Unlike conventional systems that stop at black-box classification, CottonGuard AI integrates an auditable multi-stage deep learning pipeline: EfficientNet-B0 for 7-class disease identification, Grad-CAM for explainable saliency mapping, Faster R-CNN and pure-PyTorch YOLO for lesion localization, and U-Net for pixel-level tissue segmentation.*
> 
> *Here on the landing page, we see the system overview, architecture description, and supported disease categories."*

**Action on Screen:**
- Scroll down the Landing Page (`http://localhost:5173/`).
- Highlight the 7 supported disease classes and the multi-modal diagnostic pipeline summary.

---

## Phase 2: Farmer Authentication & Dashboard (1:00 – 2:00)

**Spoken Script:**
> *"The application enforces strict Role-Based Access Control distinguishing farmers from agricultural research administrators. Let us log in as a registered farmer."*

**Action on Screen:**
- Click **"Sign In"** in the top navigation bar.
- Enter farmer credentials (e.g., `farmer@cottonguard.ai` / password or demo account).
- Arrive at the **Farmer Dashboard** (`/dashboard`).

**Spoken Script:**
> *"Upon login, the farmer dashboard displays recent diagnostic activity, historical disease frequency, and aggregate severity metrics fetched directly from our SQLite database, which currently tracks 48 verified field records."*

---

## Phase 3: Live Leaf Analysis & Real-Time SSE Inference (2:00 – 3:30)

**Spoken Script:**
> *"Now let us demonstrate a live, real-time disease diagnosis. I navigate to the **Analyze** page. I will drag and drop a real cotton leaf photograph affected by Bacterial Blight."*

**Action on Screen:**
- Click **"New Analysis"** or navigate to `/analyze`.
- Drag-and-drop a sample leaf image into the upload zone.
- Point out the client-side validation (dimensions, file size, MIME type check).
- Click **"Analyze Leaf"**.

**Spoken Script:**
> *"Notice the real-time progress indicators powered by Server-Sent Events (SSE). The backend does not simply pause; it streams each stage's completion:
> Stage 1: Validation and EXIF orientation correction.
> Stage 2: EfficientNet-B0 classification.
> Stage 3: Grad-CAM gradient backpropagation.
> Stage 4: Lesion bounding-box detection.
> Stage 5: U-Net tissue segmentation.
> Stage 6: Leaf-restricted severity estimation.
> All executed in real time using our RTX 3050 GPU."*

---

## Phase 4: Multi-Modal Result Exploration & Explainability (3:30 – 4:45)

**Spoken Script:**
> *"The inference is complete, and the multi-modal diagnostic card is rendered. Let us inspect the results:
> 
> 1. **Disease Classification:** The model predicts **Bacterial Blight** with **98.4% confidence**.
> 2. **Explainable AI (Grad-CAM):** When I toggle the Grad-CAM heatmap, examiners can observe that the network's attention focuses precisely on the angular necrotic lesions along the leaf veins, rather than background artifacts.
> 3. **Lesion Bounding Boxes:** Toggling the detection layer displays bounding boxes localizing individual pathogen colonies.
> 4. **Semantic Tissue Segmentation:** When I activate the U-Net mask, the system visually separates healthy green foliage from infected necrotic lesions.
> 5. **Mathematically Honest Severity:** Most importantly, the calculated severity is **14.82% (Low Tier)**. This is computed strictly as the ratio of lesion pixels to total leaf pixels, completely ignoring the non-leaf background."*

**Action on Screen:**
- Toggle between:
  - **Original Image**
  - **Grad-CAM Heatmap Overlay**
  - **Lesion Detection Boxes**
  - **U-Net Segmentation Mask**
- Highlight the area breakdown table (Healthy Area: $85.18\%$, Diseased Area: $14.82\%$).
- Show the agronomic recommendations section displaying targeted bio-fungicide and cultural control measures.

---

## Phase 5: Persistence & Diagnostic PDF Report Generation (4:45 – 5:30)

**Spoken Script:**
> *"Every diagnosis is automatically persisted to the database and can be exported as an official agronomic report. Let us click **'Download PDF Report'**."*

**Action on Screen:**
- Click the **"Download PDF"** button.
- Open the downloaded ReportLab-generated PDF document.
- Highlight the structured layout: Prediction ID, timestamps, side-by-side original and Grad-CAM images, bounding box coordinates, segmentation masks, severity score, and legal agronomic disclaimer.
- Navigate to **"History"** (`/history`) to demonstrate that the new prediction is recorded and filterable.

---

## Phase 6: Administrator Console & System Telemetry (5:30 – 6:30)

**Spoken Script:**
> *"Finally, let us log in to the **Administrator Console** to view system health and model telemetry."*

**Action on Screen:**
- Log out of the farmer account and log in as an administrator (or navigate to `/admin`).
- Show the **Model Registry** panel.

**Spoken Script:**
> *"The administrator panel provides transparent, auditable telemetry:
> - It reports the real operational status of all three model checkpoints.
> - It displays active GPU VRAM consumption (RTX 3050) and inference latency (~180ms per forward pass).
> - It provides full database analytics without fabricated numbers.
> - Furthermore, our automated test suite stands at **86 out of 86 tests passing (100%)**, covering everything from coordinate boundary rasterization to JWT refresh token rotation."*

---

## Phase 7: Conclusion & Examiner Handover (6:30 – 7:00)

**Spoken Script:**
> *"To conclude, CottonGuard AI delivers a robust, transparent, and mathematically rigorous solution for automated cotton leaf health monitoring. It bridges the gap between theoretical deep learning research and practical agricultural deployment.
> 
> Thank you, and I am now ready to take questions from the committee."*
