# CottonGuard AI — System Architecture & Inference Pipeline

**Document ID:** `reports/project_review/01_system_architecture_and_pipeline.md`  
**Project:** CottonGuard AI (`cotton_leaf_ai`)  
**Scope:** Final-Year Engineering Project Technical Architecture Specification  
**Status:** Complete & Verified  

---

## 1. High-Level System Architecture

CottonGuard AI is structured as an enterprise-grade, asynchronous multi-tier AI platform designed for automated cotton leaf health monitoring, disease diagnosis, lesion localization, tissue segmentation, and leaf-restricted severity estimation.

```mermaid
flowchart TD
    subgraph Client["Client Layer (React 18 + Vite + Tailwind CSS)"]
        UI_Home["Landing & Info Pages"]
        UI_Auth["Farmer & Admin Auth (JWT / Refresh)"]
        UI_Dash["Farmer Dashboard & Real-Time Stats"]
        UI_Upload["Image Upload & Client-Side Validation"]
        UI_View["Multi-Modal Result Viewer (Overlays & Toggles)"]
        UI_Hist["Prediction History & Search Filter"]
        UI_Admin["Admin Telemetry & Health Panel"]
    end

    subgraph Gateway["API & Transport Layer (FastAPI / Uvicorn)"]
        API_Auth["/api/auth (OAuth2, JWT Rotation)"]
        API_Predict["/api/predictions (SSE Stream & Sync)"]
        API_Models["/api/models (Telemetry & Metrics)"]
        API_Admin["/api/admin (System Audit & Analytics)"]
        API_Reports["/api/reports (ReportLab PDF Generation)"]
        API_Health["/api/health (Liveness & CUDA Probe)"]
    end

    subgraph Service["Backend Service Layer"]
        S_Pred["PredictionService (Singleton Engine)"]
        S_Val["ImageValidation (Magic Bytes & Dimensions)"]
        S_Sev["SeverityCalculator (Leaf-Restricted Area)"]
        S_PDF["ReportService (Diagnostic PDF Engine)"]
        S_Reg["ModelRegistry (Honest Telemetry)"]
    end

    subgraph Models["Deep Learning Inference Engines (CUDA 12.1 / RTX 3050 6GB)"]
        M_Clf["Stage 1: EfficientNet-B0 (7 Classes, Softmax)"]
        M_XAI["Stage 2: Grad-CAM Engine (features[-1] Jet Heatmap)"]
        M_Det["Stage 3: Faster R-CNN / YOLO (Lesion Box Coordinates)"]
        M_Seg["Stage 4: U-Net ResNet-34 (3-Class Pixel Segmentation)"]
    end

    subgraph Storage["Persistence & Artifact Storage"]
        DB[(SQLite: cotton_leaf_ai.db)]
        FS["File Storage (artifacts/predictions/, checkpoints/)"]
    end

    Client <-->|REST & SSE Events| Gateway
    Gateway --> Service
    Service --> Models
    Service <--> Storage
```

---

## 2. Multi-Stage AI Inference Pipeline

When a user submits a cotton leaf photograph, the system executes an orchestration pipeline comprising 7 discrete, fault-isolated stages.

```mermaid
sequenceDiagram
    autonumber
    actor Farmer as Farmer / Client UI
    participant Backend as FastAPI Server
    participant Val as Validation Engine
    participant Engine as PredictionService
    participant DB as SQLite Storage

    Farmer->>Backend: POST /api/predictions/stream (Multipart Form)
    Backend->>Val: Check Magic Bytes, MIME, Resolution & Size
    Val-->>Backend: Image Validated (Sanitized & EXIF Corrected)
    
    Backend-->>Farmer: SSE event: progress (stage="validation", step 1/7)

    Backend->>Engine: Run Stage 1: EfficientNet-B0
    Engine-->>Backend: Class: Bacterial Blight (Prob: 0.984)
    Backend-->>Farmer: SSE event: progress (stage="classification", step 2/7)

    Backend->>Engine: Run Stage 2: Grad-CAM Explainability
    Engine-->>Backend: Heatmap Tensor & Jet Overlay PNG
    Backend-->>Farmer: SSE event: progress (stage="gradcam", step 3/7)

    Backend->>Engine: Run Stage 3: Faster R-CNN Lesion Detection
    Engine-->>Backend: Bounding Boxes [xmin, ymin, xmax, ymax], Scores
    Backend-->>Farmer: SSE event: progress (stage="detection", step 4/7)

    Backend->>Engine: Run Stage 4: U-Net Semantic Segmentation
    Engine-->>Backend: 3-Class Discrete Mask (Background, Leaf, Lesion)
    Backend-->>Farmer: SSE event: progress (stage="segmentation", step 5/7)

    Backend->>Engine: Run Stage 5: Leaf-Restricted Severity Calculation
    Engine-->>Backend: Severity: 14.82% (Tier: Low), Area Stats
    Backend-->>Farmer: SSE event: progress (stage="severity", step 6/7)

    Backend->>DB: Persist PredictionRecord, Metrics & File Paths
    DB-->>Backend: Record ID #49 Committed
    Backend-->>Farmer: SSE event: complete (Payload: Full Multi-Modal Result)
```

---

## 3. Mathematical Severity Formulation

Standard naive segmentation models compute disease ratio relative to the full rectangular camera frame, introducing extreme bias when the leaf occupies only a fraction of the viewfinder. 

CottonGuard AI implements **Leaf-Restricted Severity Calculation**:

$$\text{Severity Percentage} = \frac{\text{Diseased Tissue Pixels}}{\text{Healthy Leaf Pixels} + \text{Diseased Tissue Pixels}} \times 100\%$$

Formulated in continuous set-theoretic terms:

$$\text{Severity} = 100 \times \frac{|\mathcal{M}_{\text{diseased}} \cap \mathcal{M}_{\text{leaf}}|}{|\mathcal{M}_{\text{leaf}}|}$$

Where:
- $\mathcal{M}_{\text{background}} = \text{Class 0}$ (Strictly excluded from denominator)
- $\mathcal{M}_{\text{healthy}} = \text{Class 1}$
- $\mathcal{M}_{\text{diseased}} = \text{Class 2}$
- $\mathcal{M}_{\text{leaf}} = \mathcal{M}_{\text{healthy}} \cup \mathcal{M}_{\text{diseased}}$

### Six-Tier Severity Schema
1. **Healthy:** $0.0\%$ affected
2. **Very Low:** $0.0\% < \text{Severity} \le 5.0\%$
3. **Low:** $5.0\% < \text{Severity} \le 15.0\%$
4. **Moderate:** $15.0\% < \text{Severity} \le 30.0\%$
5. **High:** $30.0\% < \text{Severity} \le 50.0\%$
6. **Severe:** $\text{Severity} > 50.0\%$

---

## 4. Hardware Optimization & Memory Management

- **Primary Target:** NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM).
- **Execution Mode:** Mixed-precision FP16 evaluation on CUDA `cuda:0`.
- **Dynamic Failover:**
  - If CUDA Out-Of-Memory (OOM) occurs, the `PredictionService` catches `torch.cuda.OutOfMemoryError`.
  - Clears CUDA cache (`torch.cuda.empty_cache()`).
  - Automatically switches the execution device to CPU (`torch.device('cpu')`), ensuring zero server crashes during high concurrency.
- **Model Lifecycle:** Singleton pattern loads models once during module initialization, preventing costly per-request re-initialization (saving ~35 seconds per request).
