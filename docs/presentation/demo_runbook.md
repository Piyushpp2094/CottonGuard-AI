# CottonGuard AI — Demonstration Runbook & Presentation Guide

**Document Reference:** `docs/presentation/demo_runbook.md`  
**Application Target:** CottonGuard AI Demonstration & Serving Stack  
**Primary Launch Script:** `D:\Main_Project\cotton_leaf_ai\run_demo.bat`  
**Evaluation Standard:** Final B.Tech Major Project College Demonstration  
**Date:** November 19, 2026  

---

## 1. Pre-Demonstration Checklist

Complete the following 5 checks before the viva examination begins:

- [ ] **Physical Hardware:** Windows 11 laptop connected to power adapter (to ensure GPU runs at full clock speed).
- [ ] **Virtual Environment:** Verify `.venv\Scripts\python.exe` is present.
- [ ] **Model Checkpoint Integrity:** Run `scripts/verify_environment.py` to confirm all 3 model hashes match:
  - Classifier: `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e`
  - Segmentor: `612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43`
  - Detector: `01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2`
- [ ] **Sample Images Present:** Verify `backend\app\static\samples\` contains the 7 reference images (`BBC00226.jpg`, `CV00248.jpg`, `HGD00202.jpg`, `HL00031.jpg`, `LHJ00185.jpg`, `LR00098.jpg`, `LV00044.jpg`).
- [ ] **Display & Resolution:** Open browser at 100% zoom (1920x1080 recommended).

---

## 2. One-Command Startup Procedure

Open Windows Command Prompt or PowerShell in `D:\Main_Project\cotton_leaf_ai` and execute:

```cmd
run_demo.bat
```

### What `run_demo.bat` Executes Automatically
1. Runs `scripts/verify_environment.py` preflight check in $<2\text{ seconds}$.
2. Opens default web browser directly to: `http://localhost:8000/app`
3. Launches FastAPI Uvicorn server on `0.0.0.0:8000` with CUDA acceleration.

---

## 3. Login & Authentication Procedure

1. In the top navigation bar, click the **Login** or **User Profile** icon.
2. Select your local demonstration account or log in as an administrator:
   - To create a fresh local admin account prior to the demo, run:
     ```cmd
     ".venv\Scripts\python.exe" scripts\create_admin.py
     ```
   - Enter your chosen local username and password.
3. Upon authentication, a green badge confirms user role and active JWT bearer token session.

---

## 4. Recommended Demonstration Image Sequence

Navigate to the **Analyze Studio** tab (`http://localhost:8000/app/analyze`). The sample gallery displays 7 preloaded images. Demonstrate them in the following pedagogical order:

### Case 1: Pristine Healthy Foliage
- **Click Card:** `HL00031.jpg` (**Healthy Leaf**)
- **Click:** **Analyze Image**
- **Expected Output:**
  - **Predicted Class:** `Healthy Leaf`
  - **Confidence:** $>99.0\%$
  - **Severity:** `0.0% to 1.5%` (Severity Grade: `Healthy / None`)
  - **Agronomic Advisory:** Confirms crop is thriving; zero chemical pesticide application prescribed.
  - **Demonstration Talking Point:** *"Notice how clean the segmentation is. The model does not falsely alarm on green foliage (only 1.11% false positive rate)."*

### Case 2: Bacterial Blight (Infectious Pathogen)
- **Click Card:** `BBC00226.jpg` (**Bacterial Blight**)
- **Click:** **Analyze Image**
- **Expected Output:**
  - **Predicted Class:** `Bacterial Blight`
  - **Confidence:** $>98.0\%$
  - **Severity:** `10% to 15%` (Severity Grade: `Moderate`)
  - **Explainability:** Grad-CAM accurately pinpoints angular water-soaked lesions.
  - **Advisory:** Prescribes Copper Oxychloride 50 WP ($2.5\text{ g/L}$) as per ICAR-CICR guidelines.
  - **Demonstration Talking Point:** *"Here the system distinguishes bacterial spots and immediately calculates the affected leaf area."*

### Case 3: Leaf Redding (Physiological Disorder vs Disease)
- **Click Card:** `LR00098.jpg` (**Leaf Redding**)
- **Click:** **Analyze Image**
- **Expected Output:**
  - **Predicted Class:** `Leaf Redding`
  - **Confidence:** $>97.0\%$
  - **Severity:** `30% to 40%` (Severity Grade: `Severe`)
  - **Advisory:** Identifies anthocyanin pigment accumulation from cold shock or magnesium deficiency; prescribes Magnesium Sulfate ($10\text{ g/L}$) foliar spray instead of toxic chemical fungicides.
  - **Demonstration Talking Point:** *"This highlights the agricultural impact: preventing farmers from spraying toxic fungicides on a nutritional/physiological condition."*

### Case 4: Leaf Variegation (High Severity Showcase)
- **Click Card:** `LV00044.jpg` (**Leaf Variegation**)
- **Click:** **Analyze Image**
- **Expected Output:**
  - **Predicted Class:** `Leaf Variegation`
  - **Confidence:** $>99.5\%$
  - **Severity:** $>50.0\%$ (Severity Grade: `Critical`)
  - **Demonstration Talking Point:** *"The multi-modal tab switcher allows the committee to inspect the Leaf Mask, Lesion Mask, Grad-CAM attention, and Composite Overlay side-by-side."*

---

## 5. Multi-Modal Visualizer Demonstration

On the **Results** screen, toggle between the 5 visualization modes:
1. **Original Image:** Shows the raw input photograph.
2. **Leaf Mask:** Delineates the cotton leaf boundary (U-Net Class 1).
3. **Lesion Mask:** Highlights isolated symptomatic regions (U-Net Class 2 in red).
4. **Grad-CAM:** Shows convolutional attention heatmap from EfficientNet `features[-1]`.
5. **Composite Overlay:** Blends leaf mask and lesion mask on top of the original photograph.

---

## 6. PDF Report Generation Demonstration

1. In the top-right corner of the Results screen, click **Download PDF Report**.
2. The browser automatically downloads `report_prediction_{id}.pdf`.
3. Open the downloaded PDF in your browser or Adobe Reader:
   - Point out the Report Title, Unique ID, and Timestamp.
   - Point out the 4-panel visual evidence grid (Photo, Leaf Mask, Disease Mask, Grad-CAM).
   - Point out the exact pixel counts ($N_{\text{leaf}}$, $N_{\text{disease}}$, $N_{\text{healthy}}$) and Affected Area percentage.
   - Point out the ICAR-CICR chemical prescriptions and dosages.
   - Point out the mandatory Scientific Advisory Notice.

---

## 7. History & Audit Trail Demonstration

1. In the top navigation bar, click **History** (`http://localhost:8000/app/history`).
2. Show the committee all 64 persisted historical predictions.
3. Demonstrate real-time searching: type "Blight" or "Redding" in the search box.
4. Filter by severity level (e.g., select "Moderate" or "Severe").
5. Click **View Details** on any historical row to show instant modal retrieval from the database.

---

## 8. Graceful Shutdown Procedure

1. Return to the Command Prompt / PowerShell console running `run_demo.bat`.
2. Press `Ctrl + C`.
3. When prompted `Terminate batch job (Y/N)?`, type `Y` and press `Enter`.
4. The server terminates cleanly without leaving orphaned Python processes.

---

## 9. Troubleshooting & FAQ for Demonstration

| Symptom / Error | Root Cause | Instant Fix |
|---|---|---|
| **Port 8000 already in use** | A previous Uvicorn process is still running | In PowerShell: `Get-Process python \| Stop-Process -Force` |
| **CUDA out of memory error** | Large background application open | Close Blender/Premiere/heavy games; system will auto-retry on CPU fallback |
| **Browser shows white screen** | Browser cache issue | Hard refresh using `Ctrl + F5` |
| **Image upload fails** | File exceeds 10MB or is not JPG/PNG | Select a sample image from the preloaded gallery |
