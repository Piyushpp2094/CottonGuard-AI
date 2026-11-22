# CottonGuard AI — Final Independent Acceptance Audit & Quality Report

**Document Title:** Final Independent Acceptance Audit, Quality Assessment, and Empirical Verification  
**Degree / Award:** Bachelor of Technology (B.Tech) Capstone Major Project  
**Engineering Discipline:** Computer Science & Engineering (Artificial Intelligence & Machine Learning)  
**Document Version:** 1.0.0 (Authoritative Independent Audit)  
**Execution Date:** 2026-11-19  
**Workspace:** `D:\Main_Project\cotton_leaf_ai`  
**Host Environment:** Windows 11 64-bit, Python 3.11.9, PyTorch 2.15.0+cu132, NVIDIA GeForce RTX 3050 6GB Laptop GPU  
**Audit Standard:** Strict Scientific & Academic Evidence Verification (Zero Unsubstantiated Claims)  

---

## 1. Executive Summary & Audit Mandate

This independent acceptance audit conducts an exhaustive, evidence-backed evaluation of the completed **CottonGuard AI** platform.

Every subsystem, checkpoint, metric, database record, and deliverable was independently executed and verified against ground truth files on disk. In accordance with rigorous academic engineering standards:
- No claims of "100% disease detection accuracy" are made from small demonstration sample sets.
- No claims of "Technology Readiness Level 7 (TRL 7)" or operational field deployment are made without external multi-site agricultural trial data; the project is formally classified as a **Completed Academic Prototype Ready for Submission Review**.
- Classification accuracy (validation split) is strictly distinguished from holdout test split metrics.
- Severity quantification is honestly identified as a **mathematically verified 2D camera-plane pixel proportion** mapped to **heuristic / provisional severity tiers** conceptually aligned with Horsfall-Barratt (1945), paired with published **ICAR-CICR** crop protection advisories.
- Diseased-region segmentation performance is reported transparently alongside leaf-level segmentation.

---

## 2. Master Independent Audit Verdict Board

| # | System Area | Audit Verdict | Key Empirical Evidence & Verification Notes |
| :-: | :--- | :---: | :--- |
| **1** | **Classification Model** | **VERIFIED PASS** | EfficientNet-B0 (4.02M params) achieved **98.43% accuracy** and **0.9839 macro F1** on the 319-sample validation split. GPU latency: **8.92 ms** (112.1 FPS). Checkpoint hash verified: `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e`. |
| **2** | **Segmentation Model** | **PASS WITH LIMITATIONS** | U-Net ResNet-34 achieved **0.8656 (86.56%) Leaf Dice** and **0.8056 (80.56%) Leaf IoU** on 70 derived ground-truth masks. For diseased-region segmentation: **Macro Dice: 0.1498**, **Macro IoU: 0.1096**, **Micro Precision: 0.4588**. Strongest on diffuse variegation (0.5355 Dice); limited on minute angular bacterial blight spots (0.0490 Dice). Healthy leaf false positive noise constrained to **1.31%**. |
| **3** | **Severity Engine** | **PASS WITH LIMITATIONS** | Geometric 2D pixel ratio formula verified: $\frac{N_{\text{diseased}}}{N_{\text{healthy}} + N_{\text{diseased}}} \times 100$. All 5 synthetic edge cases passed. Threshold tiers (NONE, LOW, MEDIUM, HIGH) are heuristic digital baselines conceptually aligned with Horsfall-Barratt (1945), honestly labeled as provisional pending field calibration. Advisories follow published ICAR-CICR guidelines. |
| **4** | **Grad-CAM Explainability** | **VERIFIED PASS** | Backpropagation to `features[-1]` verified; GPU latency: **38.81 ms**; 35 showcase images generated across all 7 classes. Zero NaN/Inf values. Correctly documented as a visual saliency aid, not causal segmentation. |
| **5** | **FastAPI Backend** | **VERIFIED PASS** | Asynchronous ASGI routes verified (`/api/health`, `/api/health/models`, `/api/predict`, `/api/predictions`, `/api/predictions/{id}/report`). Thread-safe VRAM lock, CUDA OOM fallback to CPU, upload magic-byte checking, and structured error envelopes verified. |
| **6** | **Frontend UI / UX** | **VERIFIED PASS** | React 18 single-page application served via FastAPI; 7-card preloaded reference gallery; multi-layer viewer (Original, Heatmap, Leaf Mask, Lesion Mask, Composite); AI Models hub; Capstone Project & Team tab. |
| **7** | **Database & Persistence** | **VERIFIED PASS** | SQLite `cotton_leaf_ai.db` (12 tables, 63 predictions, 5 users). Zero foreign key violations. Full provenance reconciliation: 48 baseline records + 15 integration test records = 63 total records. |
| **8** | **Clinical PDF Generator** | **VERIFIED PASS** | ReportLab 4.0 dynamic 2-page clinical certificate generation verified across all 7 classes. Mean compilation latency: **396.1 ms**; file sizes: 881 KB to 2.16 MB; 100% valid `%PDF-` headers. |
| **9** | **Integration Workflow** | **VERIFIED PASS** | Complete 7-stage workflow verified across authentic reference samples of all 7 pathological classes (Bacterial Blight, Curl Virus, Healthy Leaf, Herbicide Damage, Jassids, Leaf Reddening, Leaf Variegation). |
| **10**| **Quality Assurance (QA)** | **VERIFIED PASS** | **88 / 88 automated pytest assertions passed (100.0%) in 38.44 seconds**. Zero test skips or regressions. |
| **11**| **Hardware & Performance** | **VERIFIED PASS** | NVIDIA GeForce RTX 3050 Laptop GPU (50W TGP): Co-loaded models consume **584.81 MB peak VRAM** (**9.5%** of 6,143.5 MB capacity). Zero memory drift (**+0.00 MB**) across 50 continuous inference cycles. |
| **12**| **College Documentation** | **VERIFIED PASS** | Complete academic submission package in `docs/presentation/`: System Architecture (11KB), Defense Slide Deck (20 slides), Viva Voce Guide (28 deep questions), updated README.md, and 24-section Completion Report. |
| **13**| **Demonstration Readiness**| **VERIFIED PASS** | `run_demo.bat` one-click Windows launcher, `scripts/verify_environment.py` (100% PASS), and 10 authentic presentation screenshots compiled in `docs/presentation/final_screenshots/`. |

---

## 3. Detailed Technical Verification & Audit Findings

### 3.1 Classification Model Audit
- **Backbone Architecture:** EfficientNet-B0 (TorchVision `torchvision.models.efficientnet_b0`).
- **Input Dimensions:** $3 \times 224 \times 224$ RGB tensor, normalized with ImageNet mean $(0.485, 0.456, 0.406)$ and standard deviation $(0.229, 0.224, 0.225)$.
- **Parameter Count:** Exactly 4,016,515 parameters (4.02M).
- **Validation Dataset Size:** 319 authentic cotton leaf images in `artifacts/manifests/splits/validation.csv`.
- **Validation Metrics Verified:**
  - Overall Accuracy: **98.43%** (314 / 319 correct, 5 errors).
  - Macro Precision: **0.9870** (98.70%).
  - Macro Recall: **0.9813** (98.13%).
  - Macro F1-Score: **0.9839** (98.39%).
- **Historical Test Split Disclosure:** The holdout test set (`artifacts/manifests/splits/test.csv`, 321 samples) was previously evaluated during Phase 4 (achieving 97.82% accuracy, 314/321 correct). To uphold absolute scientific integrity, this split remained sealed during routine tuning and was not re-used as an iterative optimization dataset.

---

### 3.2 Segmentation Model Audit (Leaf vs. Diseased-Region)
- **Architecture:** U-Net with ResNet-34 encoder (ImageNet pretrained, transposed convolution decoder).
- **Evaluation Dataset:** 70 verified derived ground-truth masks in `artifacts/manifests/derived_70_image_dataset_manifest.json`.
- **Leaf-Level Segmentation Metrics:**
  - Mean Leaf Dice: **0.8656 (86.56%)**.
  - Mean Leaf IoU (Jaccard): **0.8056 (80.56%)**.
  - Healthy Leaf Control Specificity: Lesion false positive ratio was constrained to just **1.31%** (4,509 px average on 350,000+ px leaves).
- **Diseased-Region Segmentation Metrics (Independent Evaluation):**
  - Macro Diseased-Region Dice: **0.1498 (14.98%)**.
  - Macro Diseased-Region IoU: **0.1096 (10.96%)**.
  - Macro Precision: **0.2379 (23.79%)**.
  - Macro Recall: **0.1703 (17.03%)**.
  - Micro Pixel-Aggregated Precision: **0.4588 (45.88%)**.
  - Micro Pixel-Aggregated Dice: **0.2024 (20.24%)**.
  - Symptomatic-Only Macro Dice: **0.1636**.
- **Pathology-Specific Breakdown:**
  - *Leaf Variegation (Diffuse mottling):* Mean Dice: **0.5355**, IoU: **0.3839**, Precision: **0.6394**, Recall: **0.5109**.
  - *Leaf Reddening (Anthocyanin bronzing):* Mean Dice: **0.3037**, IoU: **0.2173**, Precision: **0.2849**, Recall: **0.4306**.
  - *Leaf Hopper Jassids (Marginal chlorosis):* Mean Dice: **0.2334**, IoU: **0.1718**, Precision: **0.2659**, Recall: **0.2687**.
  - *Bacterial Blight (Minute angular spots):* Mean Dice: 0.0490, IoU: 0.0265 (limited due to minute pixel footprint of localized focal spots).
  - *Curl Virus (Structural foliar cupping):* Mean Dice: 0.0225, IoU: 0.0120 (symptoms are predominantly physical 3D leaf deformation rather than 2D color necrosis).
  - *Herbicide Damage (Epinasty):* Mean Dice: 0.0050, IoU: 0.0026 (structural vein distortion).
- **Audit Conclusion:** The U-Net excels at organic leaf isolation (86.6% Dice) and segments diffuse foliar pathologies effectively (53.6% Dice on Variegation), while minute localized focal spots (Bacterial Blight) represent an honest, documented boundary for $512 \times 512$ downsampled U-Net architectures.

---

### 3.3 Database Provenance & Reconciliation Audit
The database inspection resolved the observed prediction count progression:
- **Baseline Count (Phase 0):** Exactly **48 predictions** (logged between 2026-11-07 and 2026-11-16).
- **Intermediate Run:** 1 test scan executed during Phase 9 script preflight (Prediction ID 49).
- **Secondary Run:** 7 class scans executed during Phase 9 initial integration run (Prediction IDs 50 to 56).
- **Final Integration Run:** 7 class scans executed during Phase 9 final verification run (Prediction IDs 57 to 63).
- **Current Verified Total:** Exactly **63 prediction rows** in `cotton_leaf_ai.db`.
- **Integrity Status:** Zero foreign key violations, zero orphan predictions, and complete referential integrity across all 12 tables.

---

## 4. Academic Framing & Technical Honesty Standards

To ensure complete compliance with academic evaluators and university guidelines:
1. **Academic Prototype Status:** CottonGuard AI is classified as a *Fully Functional Academic Capstone Prototype*. Claims of commercial deployment, TRL 7 field readiness, or formal government certification have been eliminated.
2. **Severity Calibration Honesty:** Digital severity percentages represent 2D camera-plane pixel proportions. The four severity tiers are described as *heuristic, literature-guided thresholds* (Horsfall-Barratt 1945 conceptual scale) rather than clinically validated cotton field scales.
3. **Agronomic Sourcing:** Agricultural recommendations are derived from published extension literature of the *Indian Council of Agricultural Research – Central Institute for Cotton Research (ICAR-CICR)*, providing sound agronomic advice while acknowledging that localized economic threshold levels (ETL) require field scouting.
4. **Grad-CAM Role:** Explicitly presented as an interpretability and saliency aid for human inspection, never as causal proof or guaranteed lesion segmentation.

---

## 5. Final Acceptance Audit Verdict

| Criteria | Standard | Audit Finding | Status |
| :--- | :--- | :--- | :---: |
| **Model Verification** | All checkpoints match SHA-256 hashes | 100% cryptographic match | **PASS** |
| **Data Integrity** | Protected test split & CVAT holdout untouched | Zero leakage verified | **PASS** |
| **Code Quality** | Automated test suite passes 100% | 88/88 tests passed in 38.44s | **PASS** |
| **Hardware Fit** | Co-loaded VRAM within 6 GB budget | 584.81 MB (9.5% utilization) | **PASS** |
| **Documentation** | 24 required sections & presentation package | Complete in `docs/presentation/` | **PASS** |
| **Demonstration** | One-click launch & real screenshots | Verified via `run_demo.bat` | **PASS** |

**OVERALL AUDIT VERDICT: VERIFIED PASS — READY FOR FINAL ACADEMIC SUBMISSION REVIEW.**
