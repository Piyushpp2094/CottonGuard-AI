# CottonGuard AI — Final Year B.Tech Project Presentation Slide Deck (25 Slides)

**Project Title:** CottonGuard AI — Intelligent Cotton Leaf Disease Detection, Segmentation, Severity Estimation, and Explainable AI System  
**Degree:** Bachelor of Technology (B.Tech) in Computer Science & Engineering (AI & ML)  
**Academic Year:** 2025–2026  
**Format:** Slide Deck Specification for Final Project Viva Voce & Defense Review  
**Screenshot Evidence Directory:** `docs/presentation/final_screenshots/`  

---

<!-- SLIDE 1 -->
## Slide 1: Title Slide
### CottonGuard AI: Intelligent Cotton Leaf Disease Detection, Segmentation, Severity Estimation & Explainable AI Platform

- **Candidate / Presenter:** B.Tech Final Year Student
- **Department:** Department of Computer Science & Engineering
- **Specialization:** Artificial Intelligence & Machine Learning
- **Capstone Project Type:** Major Capstone Engineering Project
- **Host System:** Windows 11 x64, Python 3.11, PyTorch 2.15, NVIDIA GeForce RTX 3050 GPU
- **Project Repository:** `D:\Main_Project\cotton_leaf_ai`

---

<!-- SLIDE 2 -->
## Slide 2: Problem Statement
### The Critical Dilemma in Agricultural AI

- **The Field Challenge:** Fungal, bacterial, viral, and nutritional disorders cause **30% to 50% annual yield losses** in Indian cotton crops (*Gossypium hirsutum*).
- **The Core Problem with Conventional AI:**
  1. **Black-Box Classification:** Standard CNN models output a discrete label (e.g., "Bacterial Blight") without showing *where* the disease resides on the leaf blade.
  2. **Confidence $\ne$ Severity:** Conventional systems confuse model certainty (e.g., 99% confidence) with biological disease severity (which might be only 2% damaged leaf area).
  3. **Background False Positives:** Models misclassify agricultural ground clutter (soil, mulch, stones, shadows) as symptomatic tissue.
  4. **Lack of Actionable Advisory:** Identification without treatment dosages fails to help farmers in the field.

---

<!-- SLIDE 3 -->
## Slide 3: Motivation
### The Economic & Human Imperative

- **National Agricultural Lifeline:** Cotton is cultivated across **12.5 million hectares** in India, supporting over **60 million farmers and textile workers**.
- **The Diagnostic Gap:** The ratio of trained phytopathologists to rural acreage is critically low ($<1$ expert per 5,000 hectares).
- **Misdiagnosis Consequences:** Farmers frequently mistake physiological *Leaf Redding* (cold shock / magnesium deficiency) for fungal blight, leading to:
  - Indiscriminate fungicide and pesticide application.
  - Environmental contamination and groundwater poisoning.
  - Significant financial debt with zero crop recovery.
- **The Solution:** A fast, objective, multi-modal AI system that quantifies actual foliage damage and delivers standardized ICAR-CICR treatment protocols.

---

<!-- SLIDE 4 -->
## Slide 4: Project Objectives
### Comprehensive Engineering Blueprint

1. **High-Accuracy 7-Class Disease Classification:** Attain $>98\%$ validation accuracy across 7 target leaf conditions without test-set leakage.
2. **Organic 3-Class Semantic Segmentation:** Delineate background, healthy leaf lamina, and diseased lesions using U-Net ResNet-34.
3. **Topological Damage Containment:** Constrain lesion masks strictly to the leaf boundary, eliminating background noise.
4. **Authentic Severity Quantification:** Calculate the true 2D percentage of affected leaf area and map to standardized ICAR-CICR severity grades.
5. **Visual Model Explainability:** Implement Grad-CAM convolutional saliency maps to verify attention alignment.
6. **Full-Stack Demonstrable System:** Deliver an asynchronous FastAPI backend, React 18 single-page dashboard, SQLite relational database, and instant 2-page clinical PDF report generator.

---

<!-- SLIDE 5 -->
## Slide 5: Existing Systems vs. CottonGuard AI
### Comparative Technological Advantage

| Capability / Feature | Traditional PlantVillage AI | Commercial Mobile Apps | CottonGuard AI (This Project) |
|---|:---:|:---:|:---:|
| **Disease Classification** | Single-label CNN | Single-label CNN | **EfficientNet-B0 (98.43% Val Acc)** |
| **Leaf Lamina Segmentation** | None (Lab backgrounds) | Bounding box only | **U-Net ResNet-34 (0.8656 Leaf Dice)** |
| **False-Positive Suppression**| High false alarms on soil | Medium | **1.11% Healthy False Positive Rate** |
| **Severity Quantification** | None (Conflated with conf) | Heuristic / None | **Exact 2D Pixel Ratio ($N_{\text{dis}} / N_{\text{leaf}}$)** |
| **Visual Explainability** | Black Box | None | **Grad-CAM Saliency Heatmaps** |
| **Advisory Integration** | None | Generic text | **Standardized ICAR-CICR Protocols** |
| **Exportable Clinical Report** | None | Paid / Proprietary | **Dynamic ReportLab 2-Page PDF** |

---

<!-- SLIDE 6 -->
## Slide 6: Proposed System Overview
### Multi-Stage Diagnostic Pipeline Architecture

```
[Field Photograph]
       │
       ▼
[Image Ingestion & Validation] ──► (Magic bytes, <10MB, PIL decode)
       │
       ├───► [Stage 1: EfficientNet-B0] ──► 7-Class Probability Vector & Predicted Disease
       │
       ├───► [Stage 2: U-Net ResNet-34] ──► Raw 3-Class Pixel Logits (Bg, Leaf, Lesion)
       │              │
       │              ▼
       │     [Morphological Filter]    ──► Hole filling, $A_{\min}=50\text{ px}$, Leaf Containment
       │              │
       │              ▼
       │     [Stage 3: Severity Engine] ──► Affected Area % = $N_{\text{dis}} / N_{\text{leaf}} \times 100$
       │
       └───► [Stage 4: Grad-CAM]        ──► Saliency Attention Overlay (features[-1])
                      │
                      ▼
        [ICAR-CICR Advisory Engine]     ──► Actionable Cultural, Biological & Chemical Prescriptions
                      │
                      ▼
         [FastAPI / SQLite / React]     ──► Instant UI Visualization & PDF Diagnostic Report
```

---

<!-- SLIDE 7 -->
## Slide 7: End-to-End System Architecture
### Full-Stack Enterprise Architecture

- **Presentation Layer (Frontend):** React 18, Tailwind CSS, Lucide Icons, Recharts, Vite SPA mounted at `/app`.
- **Application & API Layer (Backend):** FastAPI asynchronous framework, Uvicorn ASGI server, Pydantic V2 schema validation.
- **Inference & Compute Layer:** PyTorch 2.15, CUDA 12.4 acceleration, thread-safe GPU locking (`threading.RLock()`), automatic CPU fallback on VRAM exhaustion.
- **Data Persistence Layer:** SQLite 3 relational database, SQLAlchemy ORM, Alembic schema migration engine (`b4ca4e2a1471`).
- **Reporting Engine:** ReportLab PDF document synthesizer generating standardized agronomic advisory sheets.

---

<!-- SLIDE 8 -->
## Slide 8: Dataset Composition & Zero-Leakage Splitting
### Academic Integrity & Dataset Governance

- **Dataset Scale:** 2,240 standardized field photographs representing 7 target classes.
- **7 Target Conditions:**
  1. *Bacterial Blight* (*Xanthomonas citri* pv. *malvacearum*)
  2. *Curl Virus* (Cotton Leaf Curl Virus — CLCuV)
  3. *Healthy Leaf* (Pristine vegetative foliage)
  4. *Herbicide Growth Damage* (2,4-D / glyphosate drift deformity)
  5. *Leaf Hopper Jassids* (*Amrasca biguttula* — hopperburn)
  6. *Leaf Redding* (Physiological cold/magnesium disorder)
  7. *Leaf Variegation* (Genetic/nutritional chlorotic mosaic)
- **Zero-Leakage Split Partitioning:**
  - **Training Set (`train.csv`):** 1,600 samples (71.4%) — strictly for training.
  - **Validation Set (`validation.csv`):** 319 samples (14.3%) — strictly for model validation and selection.
  - **Sealed Test Set (`test.csv`):** 321 samples (14.3%) — cryptographically sealed and untouched.
  - **CVAT Holdout:** Frames 35–49 held out from all training iterations.

---

<!-- SLIDE 9 -->
## Slide 9: Data Preprocessing & Augmentation
### Robust Real-World Field Invariance

- **Classification Transforms ($224 \times 224\text{ px}$):**
  - Random Horizontal and Vertical Flips ($p=0.5$).
  - Random Affine Rotation ($\pm 15^\circ$) and scaling ($0.9\text{--}1.1$).
  - Color Jitter (Brightness $\pm 0.2$, Contrast $\pm 0.2$, Saturation $\pm 0.2$).
  - ImageNet Normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Segmentation Transforms ($256 \times 256\text{ px}$):**
  - Joint Spatial Transforms applying identical geometric flips to both RGB photograph and discrete multi-class target mask.
  - Nearest-neighbor interpolation on segmentation masks to prevent fractional label corruption.

---

<!-- SLIDE 10 -->
## Slide 10: Classification Model (EfficientNet-B0)
### Deep Transfer Learning with Compound Scaling

- **Why EfficientNet-B0?**
  - Achieves higher representation accuracy than ResNet-50 with **only 4.0 million parameters** (vs. 25.6M for ResNet-50).
  - Employs Mobile Inverted Bottleneck Convolutions (MBConv) with Squeeze-and-Excitation (SE) attention blocks.
  - Optimal compound scaling equation: $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$.
- **Training Hyperparameters:**
  - Optimizer: AdamW (Weight Decay $= 10^{-4}$).
  - Learning Rate: Cosine Annealing with Warmup ($\eta_{\max} = 10^{-3}$, $\eta_{\min} = 10^{-6}$).
  - Loss Function: Cross-Entropy with Label Smoothing ($0.1$).
  - Batch Size: 32 on NVIDIA RTX 3050.

---

<!-- SLIDE 11 -->
## Slide 11: Semantic Segmentation Model (U-Net ResNet-34)
### Spatial Delineation of Leaf and Lesion Boundaries

- **Architecture:** U-Net with ResNet-34 Encoder (SMP framework).
- **Encoder-Decoder Connectivity:**
  - High-level semantic features captured in deep encoder layers.
  - Spatial skip connections inject high-resolution boundary coordinates directly into decoder upsampling blocks.
- **Output Layer:** 3-channel logits corresponding to:
  - Channel 0: Background
  - Channel 1: Healthy Leaf Blade
  - Channel 2: Diseased / Symptomatic Lesion
- **Post-Processing Guardrails:**
  - Hole filling on leaf canopy (`ndimage.binary_fill_holes`).
  - Small connected component noise suppression ($A_{\min} = 50\text{ px}$).
  - Strict topological containment: $\text{Disease} = \text{Disease} \land \text{Leaf}$.

---

<!-- SLIDE 12 -->
## Slide 12: Severity Estimation Engine
### Mathematical Affected-Area Quantification

- **The Mathematical Formula:**
  $$\text{Affected Area Percentage} = \frac{N_{\text{disease}}}{N_{\text{healthy}} + N_{\text{disease}}} \times 100 = \frac{N_{\text{disease}}}{N_{\text{leaf\_total}}} \times 100$$
- **Edge-Case Protections:**
  - Division-by-Zero: If $N_{\text{leaf}} = 0$, returns `0.0%` with status `NO_LEAF_DETECTED`.
  - Micro-Leaves: If $N_{\text{leaf}} < 500\text{ px}$, returns `0.0%` with status `LEAF_TOO_SMALL`.
  - Bounded Range: Guaranteed strictly within $[0.0\%, 100.0\%]$.
- **ICAR-CICR Severity Categorization Tiers:**
  - **Healthy:** $0.00\%$
  - **Low / Mild:** $> 0.00\% \text{ to } \le 10.00\%$
  - **Moderate:** $> 10.00\% \text{ to } \le 25.00\%$
  - **High / Severe:** $> 25.00\% \text{ to } \le 50.00\%$
  - **Critical:** $> 50.00\% \text{ to } 100.00\%$

---

<!-- SLIDE 13 -->
## Slide 13: Model Explainability (Grad-CAM)
### Visual Saliency & Convolutional Attention

- **Mathematical Principle:**
  Grad-CAM computes gradients of the target class score $y^c$ with respect to feature activation map $A^k$ of the final convolutional layer:
  $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
- **Target Layer:** EfficientNet-B0 `features[-1]` (1280 channels, $7 \times 7$ spatial grid).
- **Scientific Advisory Disclaimer:** Formally identifies Grad-CAM as an attention heatmap reflecting model visual focus, not proof of causal biological pathology.

---

<!-- SLIDE 14 -->
## Slide 14: Enterprise FastAPI Backend
### High-Concurrency Asynchronous Architecture

- **10+ Modular API Routers:** Health, Authentication, Predictions, History, Explainability, Dashboard, AI Models, Reports, Admin.
- **Thread-Safe GPU Locking:** Encapsulates CUDA forward passes in a reentrant lock (`threading.RLock()`), preventing multi-tenant race conditions.
- **Zero-Crash CUDA OOM Fallback:** Catches `torch.cuda.OutOfMemoryError`, clears CUDA cache, moves tensors to CPU fallback, completes the analysis, and returns model to GPU.
- **Strict Image Ingestion Security:** Magic bytes validation (`FF D8 FF` for JPG, `89 50 4E 47` for PNG), $10\text{ MB}$ ceiling, and PIL dimension boundaries.

---

<!-- SLIDE 15 -->
## Slide 15: Modern React 18 Frontend
### Agronomy-Focused User Interface

- **Dashboard View:** Live analysis KPI counts, disease distribution charts, healthy-vs-diseased ratios, and recent analysis feeds.
  *(See: `docs/presentation/final_screenshots/01_dashboard.png`)*
- **Analyze Studio:** Drag-and-drop file uploader, file validation alerts, and 7-image one-click sample gallery.
  *(See: `docs/presentation/final_screenshots/02_analyze_upload.png`)*
- **Multi-Modal Results Viewer:** Tabbed switching between Original, Leaf Mask, Lesion Mask, Grad-CAM, and Composite Overlay.
  *(See: `docs/presentation/final_screenshots/03_prediction_result.png` & `08_severity_quantification.png`)*
- **History Portal:** Full audit trail with pagination, search, disease/severity filters, and PDF download triggers.
  *(See: `docs/presentation/final_screenshots/09_prediction_history.png`)*

---

<!-- SLIDE 16 -->
## Slide 16: Relational Database Architecture
### SQLite Relational Storage & Alembic Migrations

- **Database Engine:** SQLite 3 with WAL (Write-Ahead Logging) and `PRAGMA foreign_keys = ON`.
- **12 Normalized Tables:** `users`, `predictions`, `prediction_results`, `segmentation_results`, `severity_results`, `detection_results`, `recommendation_records`, `batch_jobs`, `crop_batches`, `feedback_records`, `audit_logs`, `alembic_version`.
- **Relational Integrity Audit:** Verified **0 foreign key violations** across all 64 persisted historical predictions.
- **Role-Based Access Control:** Preconfigured roles (`admin`, `field_officer`, `farmer`) with bcrypt-hashed credentials.

---

<!-- SLIDE 17 -->
## Slide 17: Classification Experimental Results
### State-of-the-Art Validation Accuracy (98.43%)

- **Validation Set Evaluation (319 Images):**
  - **Top-1 Accuracy:** **98.43%** (314 / 319 correct)
  - **Macro Precision:** **0.9870**
  - **Macro Recall:** **0.9813**
  - **Macro F1-Score:** **0.9839**
  - **Weighted F1-Score:** **0.9842**
  - **Mean Confidence:** **92.65%**
- **Flawless Classes ($1.000$ F1):** Herbicide Growth Damage, Leaf Variegation.
- **High-Fidelity Classes ($>0.98$ F1):** Bacterial Blight (0.9863), Curl Virus (0.9922), Healthy Leaf (0.9867), Leaf Redding (0.9829).

---

<!-- SLIDE 18 -->
## Slide 18: Confusion Matrix Analysis
### Diagonal Concentration & Error Audit

```
Predicted Class ->      BB    CV    HL   HGD   LHJ    LR    LV
True Class
Bacterial Blight (37)   36     0     0     0     1     0     0
Curl Virus (65)          0    64     1     0     0     0     0
Healthy Leaf (37)        0     0    37     0     0     0     0
Herbicide Damage (42)    0     0     0    42     0     0     0
Leaf Hopper Jassids (34) 0     0     0     0    31     3     0
Leaf Redding (86)        0     0     0     0     0    86     0
Leaf Variegation (18)    0     0     0     0     0     0    18
```

- **Analysis of 5 Errors:**
  - 3 Jassid samples confused with Reddening due to advanced toxic hopperburn mimicking anthocyanin reddening.
  - 1 Bacterial Blight confused with Jassid due to vein-margin chlorosis.
  - 1 Curl Virus confused with Healthy Leaf due to mild early-stage upward curling without leaf deformation.

---

<!-- SLIDE 19 -->
## Slide 19: Segmentation Results & Model Selection
### Empirical Findings & Decision Rationale

- **Production Baseline U-Net (`best_unet.pt`):**
  - **Leaf Lamina Dice:** **0.8656** (IoU: **0.8056**)
  - **Healthy Leaf False Positive Rate:** **1.11%** (Only 4,820 px average noise)
  - **Lesion Macro Dice:** **0.1498** (Micro Dice: **0.2024**)
- **Experimental 5-Fold Cross-Validation Audit:**
  - Out-of-fold candidate models increased healthy leaf false positives to **19.79%** (80,143 px false noise).
  - Leaf Dice dropped from $0.8656 \rightarrow 0.7983$.
- **Model Selection Verdict:** Production baseline `best_unet.pt` **RETAINED**. A model with low false-alarm rate (1.11%) is agronomically essential to avoid unnecessary chemical pesticide sprays.

---

<!-- SLIDE 20 -->
## Slide 20: Comprehensive Model Comparison
### Multi-Model Tradeoff Analysis

| Architecture | Task | Parameters | Primary Metric | Inference Latency | Deployment Role |
|---|---|:---:|:---:|:---:|---|
| **EfficientNet-B0** | Classification | **4.0 M** | **98.43% Val Acc** | **14.94 ms** | **Production Core** |
| **ResNet-50** | Classification | 25.6 M | 95.80% Val Acc | 32.10 ms | Superseded |
| **U-Net ResNet-34** | Segmentation | **24.4 M** | **0.8656 Leaf Dice**| **93.09 ms** | **Production Core** |
| **5-Fold Candidate**| Segmentation | 24.4 M | 0.7983 Leaf Dice | 93.10 ms | Rejected (High FP) |
| **Faster R-CNN V2** | Lesion BBoxes | 43.7 M | 13.94% mAP50 | 78.40 ms | Experimental |
| **Grad-CAM** | Explainability | — | Saliency Map | **39.81 ms** | **Production Core** |

---

<!-- SLIDE 21 -->
## Slide 21: Hardware Performance & VRAM Profiling
### Benchmarked on NVIDIA GeForce RTX 3050 6GB Laptop GPU

- **Empirical Monotonic Latency (30 Warm Trials):**
  - EfficientNet-B0 Classifier: **14.94 ms** (P95: 20.72 ms)
  - Grad-CAM Heatmap Synthesis: **39.81 ms** (P95: 53.57 ms)
  - U-Net Segmentation + Morphology: **93.09 ms** (P95: 100.70 ms)
  - **Full Multimodal Diagnostic Pipeline:** **146.93 ms** (~6.8 analyses/second)
- **VRAM Utilization:**
  - Allocated Memory: **557.15 MB**
  - Peak Reserved Memory: **834.00 MB** (out of 6,144 MB)
  - Headroom: $>5.3\text{ GB}$ free VRAM (86.4% safety margin)
- **CPU Fallback Latency:** **715.3 ms** (~1.4 analyses/second on Intel 14-core host).

---

<!-- SLIDE 22 -->
## Slide 22: Rigorous Testing & Quality Assurance
### 100% Automated Test Suite Verification

- **Test Framework:** Pytest 8.3+, FastAPI TestClient, SQLAlchemy Test Engine.
- **Canonical Test Execution Result:** **87 passed in 40.44 seconds (100% PASS)**.
- **Discrepancy Reconciliation:** An earlier document cited "88 tests" due to a manual arithmetic typo in a Markdown table. Exactly 87 test functions exist; 0 were skipped or removed.
- **Test Coverage Areas:**
  - REST API contracts & HTTP status codes (`test_predict.py`, `test_health.py`).
  - Image boundary security & magic bytes (`test_image_validation.py`).
  - Severity calculation edge cases & division by zero (`test_severity.py`).
  - Database foreign keys & migration rollbacks (`test_database.py`).
  - Model checkpoint cryptographic integrity (`test_model_registry.py`).

---

<!-- SLIDE 23 -->
## Slide 23: Limitations & Future Work
### Honest Academic Transparency & Roadmap

- **Documented Limitations:**
  1. *Small Lesion Overlap:* Bacterial blight punctate spots ($<30\text{ px}$) exhibit low intersection-over-union.
  2. *Annotation Guideline Variance:* Disparities between local spot vs. whole-leaf disease annotations in training batches.
  3. *Visual Scope:* Severity represents visible 2D leaf damage, serving as decision support rather than laboratory microbiology.
- **Future Engineering Roadmap:**
  1. *Edge Quantization:* Export models to INT8 ONNX / TensorRT for offline Android smartphone apps.
  2. *Multi-Spectral Drone Integration:* Scale pipeline to drone orthomosaics for automated field acreage damage maps.
  3. *Active Learning:* Exploit the SQLite feedback loop to continuously refine difficult disease boundaries.

---

<!-- SLIDE 24 -->
## Slide 24: Conclusion & Academic Summary
### Engineering Highlights & Deliverables

- **Delivered a Complete Agricultural AI System:** Took CottonGuard AI from research prototypes to a fully verified, professional major project.
- **Rigorous Technical Triumphs:**
  - **98.43% Validation Accuracy** on 7 cotton disease classes.
  - **0.8656 Leaf Dice** with **1.11% healthy false positive rate**.
  - **146.93 ms** full multimodal inference latency on RTX 3050 GPU.
  - **87 / 87 automated tests passing**.
  - Dynamic 2-page PDF diagnostic report generation.
- **Scientific Integrity:** Maintained zero data leakage, protected all checkpoints, and documented model limitations with complete academic honesty.

---

<!-- SLIDE 25 -->
## Slide 25: Live Demonstration & Viva Defense
### Interactive System Walkthrough

- **One-Command Demonstration Launch:**
  ```cmd
  run_demo.bat
  ```
- **Live System Walkthrough:**
  1. **Dashboard KPI Telemetry:** Review live diagnostic statistics and GPU readiness.
  2. **Analyze Studio:** One-click analysis of reference samples from all 7 classes.
  3. **Multi-Modal Visualizer:** Toggle between Leaf Mask, Lesion Mask, Grad-CAM, and Composite Overlays.
  4. **Severity Quantification:** Inspect pixel counts and ICAR-CICR advisory.
  5. **Audit History & PDF Export:** Download dynamically generated diagnostic report.

**Thank you to the Evaluation Committee. We welcome your questions!**
