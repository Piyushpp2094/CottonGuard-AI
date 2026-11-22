# CottonGuard AI — Final-Year Project Defense Presentation

**Document ID:** `reports/project_review/04_presentation_slides.md`  
**Target Audience:** Final-Year Engineering Project Review Committee / Faculty Examiners  
**Format:** 12-Slide Defense Slide Deck Draft  
**Status:** Complete & Ready for Presentation  

---

## Slide 1: Title & Project Identity

### **CottonGuard AI: Automated Cotton Leaf Disease Classification, Lesion Detection, Tissue Segmentation & Severity Estimation Platform**

- **Domain:** Artificial Intelligence, Computer Vision, Precision Agriculture
- **Candidate:** Final-Year Engineering Student / Project Team
- **Core Technology Stack:** PyTorch, CUDA, FastAPI, React 18, SQLite, Tailwind CSS
- **Academic Year:** 2026–2027

---

## Slide 2: Problem Statement & Motivation

### The Agricultural Crisis in Cotton Production
- Cotton (*Gossypium hirsutum*) is a vital cash crop supporting millions of smallholder farmers and the textile industry.
- Foliar diseases (Bacterial Blight, Cotton Leaf Curl Virus, Leaf Hopper damage) cause devastating annual yield losses of **25% to 60%**.
- **Challenges in Current Practices:**
  - Manual scouting is labor-intensive, subjective, and prone to inter-observer variability.
  - Standard agricultural apps only provide whole-image classification, giving no spatial localization or disease extent.
  - Over-application of chemical pesticides due to inaccurate severity estimation creates severe environmental toxicity and economic strain.

---

## Slide 3: Project Objectives & Scope

### Concrete Engineering Goals
1. **Accurate Disease Classification:** Multi-class classification across 7 foliar states with $>95\%$ precision.
2. **Transparent Explainability:** Generate Grad-CAM saliency heatmaps to visually ground model decisions on genuine lesions.
3. **Fine-Grained Lesion Detection:** Localize infected sites using bounding-box object detection (Faster R-CNN / pure PyTorch YOLO).
4. **Pixel-Level Tissue Segmentation:** Segment cotton leaves into Background, Healthy Foliage, and Diseased Necrotic Tissue using U-Net.
5. **Mathematically Honest Severity:** Calculate affected leaf area percentage strictly restricted to the leaf boundary ($100 \times \frac{\text{diseased}}{\text{leaf}}$).
6. **Full-Stack Production System:** Deliver a real, locally deployed web application with role-based access control (Farmer & Admin), real-time SSE streaming, and automated diagnostic PDF generation.

---

## Slide 4: Shortcomings of Prior Art vs. Proposed Innovations

| Limitation in Prior Academic Studies | Innovation in CottonGuard AI |
|:---|:---|
| Whole-image classification only without spatial context | **Multi-stage cascade:** Classification $\rightarrow$ Grad-CAM $\rightarrow$ Detection $\rightarrow$ Segmentation. |
| Naive severity formulas dividing lesion pixels by the rectangular camera frame ($H \times W$) | **Leaf-Restricted Severity:** Computes ratio strictly against segmented leaf foliage, eliminating background framing bias. |
| Black-box predictions with zero auditable evidence | **Grad-CAM Saliency Maps:** Visual verification of symptomatic leaf veins and margins. |
| Reliance on synthetic or toy datasets | Real-world field corpus with **strict coordinate boundary preservation** ($[0, 800]$ continuous space to $[0, 799]$ discrete matrix). |
| Static notebooks or mock UIs | **Complete enterprise web application** with live FastAPI backend, React UI, SQLite persistence, and ReportLab PDF reporting. |

---

## Slide 5: System Architecture & Data Flow

```mermaid
flowchart LR
    Img[Field Image] --> Ingest[Sanitization & EXIF]
    Ingest --> Clf[EfficientNet-B0<br/>7 Classes]
    Clf --> XAI[Grad-CAM<br/>features[-1]]
    Ingest --> Det[Faster R-CNN / YOLO<br/>Lesion Bounding Boxes]
    Ingest --> Seg[U-Net ResNet-34<br/>3-Class Pixel Mask]
    Seg --> Sev[Leaf-Restricted<br/>Severity Ratio]
    Clf & XAI & Det & Seg & Sev --> UI[React Frontend & PDF Report]
```

- **Client:** React 18 + Vite + Tailwind CSS with canvas overlay toggles.
- **Backend:** FastAPI with Server-Sent Events (SSE) streaming real-time stage progress (1/7 through 7/7).
- **Inference Engine:** PyTorch on RTX 3050 6GB GPU with automatic CPU fallback.

---

## Slide 6: Dataset Curation & Annotation Governance

- **Classification Corpus:** 2,132 field images across 7 classes:
  1. *Bacterial Blight* | 2. *Curl Virus* | 3. *Healthy Leaf* | 4. *Herbicide Growth Damage* | 5. *Leaf Hopper Jassids* | 6. *Leaf Reddening* | 7. *Leaf Variegation*.
- **Untouched Test Benchmark:** 321 images (`artifacts/manifests/splits/test.csv`) isolated with cryptographic SHA-256 integrity.
- **100-Image Annotation Corpus:**
  - 70-image Stratified 5-Fold Development Set (14 images/fold, singleton duplicate groups).
  - 15-image Excluded Historical Validation Set.
  - 15-image Reserved Sealed Holdout (unseen evaluation).
- **Coordinate Boundary Policy:** Preserves outer boundary vertices ($800.0$) using nearest-even rounding without arbitrary clamping to 799.

---

## Slide 7: Deep Learning Model Suite

### 1. EfficientNet-B0 (Classification)
- Compound scaling architecture balancing depth, width, and resolution.
- Pretrained on ImageNet; fine-tuned with dropout ($0.2$) on custom 7-class head.

### 2. Grad-CAM Engine (Explainability)
- Extracts gradients from final convolutional stage `features[-1]`.
- Generates transparent jet heatmap blended with original leaf photograph.

### 3. Faster R-CNN ResNet-50 FPN V2 & CottonLeafYOLO (Detection)
- Multi-scale feature pyramid network with Tier-2 ignore regions for ambiguous lesion zones.
- Pure-PyTorch YOLO alternative eliminating heavyweight AGPL dependencies.

### 4. U-Net ResNet-34 (Segmentation)
- Encoder-decoder network with skip connections for high-resolution edge recovery.
- Combined Soft Dice + Cross-Entropy loss for small necrotic spots.

---

## Slide 8: Mathematical Severity Estimation

$$\text{Severity Percentage} = \frac{\text{Diseased Tissue Pixels}}{\text{Healthy Leaf Pixels} + \text{Diseased Tissue Pixels}} \times 100\%$$

```
   ┌────────────────────────────────────────────────────────┐
   │ CAMERA FRAME (Excluded Background: Class 0)            │
   │                                                        │
   │          ┌───────────────────────────────────┐         │
   │          │  HEALTHY LEAF (Class 1)           │         │
   │          │                                   │         │
   │          │      ████ LESIONS (Class 2)       │         │
   │          │      ████                         │         │
   │          └───────────────────────────────────┘         │
   │                                                        │
   └────────────────────────────────────────────────────────┘
```

- **6-Tier Severity Schema:**
  - `Healthy`: $0.0\%$
  - `Very Low`: $0.0\% - 5.0\%$
  - `Low`: $5.0\% - 15.0\%$
  - `Moderate`: $15.0\% - 30.0\%$
  - `High`: $30.0\% - 50.0\%$
  - `Severe`: $> 50.0\%$

---

## Slide 9: Experimental Results & Benchmarks

### Verified Model Performance Metrics

| Evaluation Metric | Measured Benchmark | Validation Split / Dataset |
|:---|:---:|:---|
| **Untouched Test Accuracy** | **97.82%** | 314 / 321 correct on test split |
| **Validation Accuracy** | **98.43%** | EfficientNet-B0 validation fold |
| **Test Macro F1-Score** | **97.54%** | Harmonic mean across 7 classes |
| **Bacterial Blight F1** | **97.80%** | 46 test samples |
| **Healthy Leaf Accuracy** | **100.0%** | Zero false-positive disease attribution |
| **System Test Suite** | **86 / 86 Passed (100%)** | End-to-end integration & unit tests |

---

## Slide 10: Full-Stack Web Application Walkthrough

### Production-Ready Features
1. **Farmer Experience:**
   - Registration, login, and secure JWT session management.
   - Drag-and-drop leaf upload with real-time SSE progress indicators.
   - Interactive Multi-Modal Results: toggle original, Grad-CAM, lesion bounding boxes, and segmentation masks.
   - Agronomic treatment advice (organic bio-fungicide, synthetic options, cultural sanitation).
   - One-click ReportLab PDF diagnostic report download.
2. **Administrator Console:**
   - Real-time hardware telemetry (RTX 3050 VRAM usage, inference latency).
   - Database analytics (48 stored predictions, user distribution).
   - Live system health and model registry status.

---

## Slide 11: Academic Rigor, Limitations & Honest Disclosure

- **Hardware Feasibility:** Tuned for 6 GB laptop GPU (RTX 3050) with automated CPU fallback during VRAM exhaustion.
- **Honest Limitations:**
  - Micro-lesion detection ($<5 \times 5$ px) remains challenging in variable field illumination.
  - The 6-tier severity scale represents an engineering stratification, requiring local agronomist calibration for commercial chemical prescription.
  - CVAT Task #6 annotation remediation is maintained under strict governance (`APPROVAL_GATE_4 = PENDING_APPROVAL`) with automated rollback.

---

## Slide 12: Conclusion & Future Work

### Summary of Contributions
- Developed a complete, functional end-to-end AI platform for cotton foliar health.
- Proven classification accuracy of **97.82%** with auditable Grad-CAM transparency.
- Established mathematically robust leaf-restricted severity estimation.
- Delivered a fully tested application with **86/86 passing tests**, verified database persistence, and one-click launch scripts.

### Future Enhancements
- Edge device deployment on Raspberry Pi / NVIDIA Jetson for on-combine harvesting telemetry.
- Multi-spectral drone imagery integration for field-wide disease mapping.
- Integration of regional weather forecasting for predictive disease outbreak warnings.
