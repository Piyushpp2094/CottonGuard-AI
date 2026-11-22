# CottonGuard AI — Phase 12: College Project Deliverables Manifest

**Document Title:** College Presentation Package & Academic Deliverables Manifest  
**Degree / Award:** Bachelor of Technology (B.Tech) in Computer Science & Engineering (AI & ML)  
**Document Version:** 1.0.0  
**Date:** November 2026  
**Workspace:** `D:\Main_Project\cotton_leaf_ai`  

---

## 1. Executive Summary

Phase 12 compiles the authoritative final-year academic presentation and technical defense deliverables for **CottonGuard AI — Intelligent Cotton Leaf Disease Detection, Segmentation, and Severity Estimation System**.

All three core deliverables have been authored, peer-reviewed against academic rubrics, cross-referenced with empirical project artifacts, and stored under `docs/presentation/`.

---

## 2. Master Deliverables Registry

| # | Deliverable Title | File Path | Format / Size | Cryptographic SHA-256 Hash | Target Audience / Usage |
| :-: | :--- | :--- | :---: | :--- | :--- |
| **1** | **Comprehensive System Architecture & Design Document** | `docs/presentation/system_architecture_design.md` | Markdown / 11,198 bytes | `CFBB58FD97E7E983C862BF36E2F6775E780441154534C29FBC2B34C82D33AAD3` | Project Guide, Technical Evaluators, System Reference |
| **2** | **College Presentation Slides (20-Slide Defense Deck)** | `docs/presentation/college_presentation_slides.md` | Markdown Slides / 14,433 bytes | `A88198732CD9AA13FAF7EAFF93C0B20ECA3C99F34A8FC499BD0B51CF30483C42` | Final Viva Defense, Capstone Presentation Committee |
| **3** | **Viva Voce Examiner Q&A Guide (28 Technical Questions)** | `docs/presentation/viva_voce_qa_guide.md` | Markdown / 22,320 bytes | `6D174FBB0BEE68B0109A968C0A083C174CE0181CF620CAE962007AE0C8A64689` | Oral Defense Preparation, External Examiner Q&A |

---

## 3. Deliverable Content Breakdown

### 3.1 Deliverable 1: System Architecture & Design Document
- **Location:** [system_architecture_design.md](file:///D:/Main_Project/cotton_leaf_ai/docs/presentation/system_architecture_design.md)
- **Key Sections:**
  - Executive Problem Statement & Agricultural Motivation.
  - Global System Architecture (Full Mermaid Component Diagram).
  - Multi-Architecture Selection Strategy (EfficientNet-B0 vs ResNet-50 vs EfficientNet-B2).
  - 3-Class U-Net ResNet-34 Segmentation & Topological Boundary Containment.
  - Geometric Severity Quantification Engine & Clinical Tier Mapping.
  - Grad-CAM Activation Layer Derivations (`features[-1]`).
  - FastAPI Asynchronous Gateway, SSE Streaming, and SQLite Persistence.
  - ReportLab 4.0 Dynamic 2-Page Clinical PDF Generation Engine.

### 3.2 Deliverable 2: College Presentation Slide Deck
- **Location:** [college_presentation_slides.md](file:///D:/Main_Project/cotton_leaf_ai/docs/presentation/college_presentation_slides.md)
- **Structure:** 20 structured slides formatted for presentation software (Marp, Deckset, or PowerPoint export).
- **Core Topics Covered:**
  - Motivation, Economic Toll, Problem Statement & Limitations.
  - Architecture Diagram, Dataset Governance & Stratified Splits.
  - CVAT Annotation Protocol, Derived Export & Priority Clipping.
  - Classification Benchmark Table & Pareto Selection.
  - U-Net Segmentation Metrics (0.8656 Leaf Dice, 0.8056 Leaf IoU).
  - Decoupled Severity Formulation & ICAR-CICR Translation.
  - Grad-CAM 5-Layer Stack, Frontend UI Showcase, Backend Architecture.
  - Edge GPU Profiling (584 MB VRAM, 8.9 ms latency, zero memory leak).
  - 88/88 Test Suite Pass, 7-Class Empirical Case Studies, Future Scope, Conclusion.

### 3.3 Deliverable 3: Viva Voce Examiner Q&A Guide
- **Location:** [viva_voce_qa_guide.md](file:///D:/Main_Project/cotton_leaf_ai/docs/presentation/viva_voce_qa_guide.md)
- **Breadth:** 28 deep, rigorous questions with complete theoretical, mathematical, and practical answers.
- **Seven Core Categories:**
  1. AI / ML Architectures & Design Choices (4 questions).
  2. Data Engineering, CVAT Annotations & Zero-Leakage Governance (4 questions).
  3. Semantic Segmentation & Severity Mathematics (3 questions).
  4. Model Explainability & Grad-CAM Backprop (2 questions).
  5. Backend Engineering, Security & Performance (5 questions).
  6. Agricultural Science & ICAR-CICR Advisories (2 questions).
  7. Software Quality, Testing & Production Readiness (8 questions).

---

## 4. Phase 12 Sign-Off

Phase 12 (College Project Deliverables) is formally signed off. The presentation package is complete, professional, and ready for deployment and evaluation.

- **Status:** **COMPLETED**
- **Next Phase:** Phase 13 — Setup, Run Scripts & Deployment Readiness (`run_demo.bat`, `scripts/verify_environment.py`, and updated root `README.md`).
