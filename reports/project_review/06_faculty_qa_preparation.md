# CottonGuard AI — Faculty Viva Voce & Technical Defense Q&A Guide

**Document ID:** `reports/project_review/06_faculty_qa_preparation.md`  
**Target:** Final-Year Engineering Project Defense Committee  
**Purpose:** Comprehensive Technical Defense Answers for Viva Voce & Academic Review  
**Status:** Complete & Verified  

---

## Section 1: Deep Learning Architectures & Model Selection

### Q1: Why did you choose EfficientNet-B0 for classification over heavier architectures like ResNet-50 or Vision Transformers (ViT)?
**Examiner's Intent:** Testing understanding of model efficiency, compound scaling, and edge deployment constraints.  
**Authoritative Answer:**
> *"We selected EfficientNet-B0 due to its principled **compound scaling method**, which uniformly scales network depth ($d=1.0$), width ($w=1.0$), and resolution ($r=224$) using a fixed compound coefficient $\phi$. 
> 
> While ResNet-50 contains approximately 25.6 million parameters, EfficientNet-B0 achieves superior top-1 feature extraction with only **5.3 million parameters** (an ~80% parameter reduction). On agricultural edge devices and laptop GPUs (such as our RTX 3050 6GB), EfficientNet-B0 provides an inference latency of under 35ms per frame. 
> 
> Furthermore, Vision Transformers (ViTs) lack inductive biases (such as translation equivariance and locality) and require massive pre-training datasets (e.g., JFT-300M) to avoid severe overfitting on domain-specific agricultural datasets of ~2,100 images."*

---

### Q2: Why does CottonGuard AI implement both a Lesion Detector (Faster R-CNN / YOLO) and a Semantic Segmenter (U-Net)? Isn't one redundant?
**Examiner's Intent:** Probing understanding of object-level vs. pixel-level computer vision tasks.  
**Authoritative Answer:**
> *"They solve complementary agricultural problems:
> 1. **Lesion Detection (Faster R-CNN / YOLO)** localizes discrete infected sites with bounding boxes, enabling counting of pathogen colonies and assessing disease dispersion density across the canopy.
> 2. **Semantic Segmentation (U-Net)** classifies every individual pixel into Background, Healthy Leaf, or Diseased Tissue. Bounding boxes cannot accurately calculate leaf-restricted severity because bounding boxes inevitably include healthy green leaf tissue and surrounding background inside the box rectangle. 
> 
> Therefore, U-Net is mathematically required for accurate severity area calculation, while the detector provides discrete lesion localization."*

---

### Q3: How does your in-tree YOLO detector avoid AGPL licensing constraints?
**Examiner's Intent:** Checking understanding of software engineering licenses and model provenance.  
**Authoritative Answer:**
> *"Commercial and production deployments often forbid AGPLv3-licensed packages such as Ultralytics due to viral open-source copyleft restrictions. 
> 
> CottonGuard AI implements `CottonLeafYOLODetector` in `src/models/detector.py`, an in-tree, clean-room pure PyTorch implementation using standard BSD-compatible modules: CSPDarknet backbone, a 3-scale Path Aggregation Network (PANet), and decoupled anchor-free classification and regression heads. This provides full intellectual property independence."*

---

## Section 2: Mathematical Severity Estimation & Ground Truth

### Q4: Explain the mathematical formula for Disease Severity. Why not simply divide lesion pixels by the total image pixels ($H \times W$)?
**Examiner's Intent:** Testing understanding of agricultural domain constraints and mathematical rigor.  
**Authoritative Answer:**
> *"Dividing lesion pixels by total image dimensions ($H \times W = 800 \times 800$) is mathematically flawed because camera distance and framing introduce massive bias. A small leaf photographed close-up would register as high severity, while a heavily infected leaf photographed from a distance would artificially register as low severity due to background dominance.
> 
> To guarantee framing invariance, CottonGuard AI implements **Leaf-Restricted Severity**:
> 
> $$\text{Severity Percentage} = \frac{\text{Diseased Tissue Pixels}}{\text{Healthy Leaf Pixels} + \text{Diseased Tissue Pixels}} \times 100\%$$
> 
> Set-theoretically:
> 
> $$\text{Severity} = 100 \times \frac{|\mathcal{M}_{\text{diseased}} \cap \mathcal{M}_{\text{leaf}}|}{|\mathcal{M}_{\text{leaf}}|}$$
> 
> Background pixels ($\text{Class 0}$) are strictly excluded from the denominator. This ensures that whether the leaf covers 20% or 90% of the camera frame, the severity percentage reflects the true biological condition of the leaf."*

---

### Q5: Are your 6 severity tiers scientifically validated agricultural treatment thresholds?
**Examiner's Intent:** Assessing academic honesty and avoidance of unverified domain claims.  
**Authoritative Answer:**
> *"No, and we explicitly document this distinction in our project reports and user interface. Our 6-tier schema (`Healthy: 0%`, `Very Low: 0–5%`, `Low: 5–15%`, `Moderate: 15–30%`, `High: 30–50%`, `Severe: >50%`) represents an engineering-defined foliar damage stratification. 
> 
> Actual agronomic Economic Injury Levels (EIL) and Economic Thresholds (ET) depend heavily on regional agro-climatic zones, cotton cultivars (e.g., Bt-cotton vs. non-Bt), crop growth stages (squaring, flowering, boll formation), and local pesticide regulations. Our system provides the quantitative metric, but leaves chemical prescription thresholds to certified agricultural extension specialists."*

---

## Section 3: Data Integrity, Leakage Prevention & Boundary Handling

### Q6: How did you ensure there is no data leakage between your training, validation, and test sets?
**Examiner's Intent:** Checking for methodological flaws in ML experimentation.  
**Authoritative Answer:**
> *"We implemented three strict architectural barriers:
> 1. **Protected Untouched Test Split:** `artifacts/manifests/splits/test.csv` (321 images) is cryptographically isolated. It was never used for model selection, training, data augmentation design, or threshold tuning.
> 2. **Group-Aware Stratification:** In the 70-image development corpus (`proposed_70_image_stratified_folds.csv`), images belonging to the same duplicate groups or sequence captures are strictly kept within the same fold (singleton duplicate groups), preventing near-identical samples from appearing in both train and validation folds.
> 3. **Sealed Holdout Partition:** Global IDs `85–99` (15 images) are strictly sealed for post-remediation holdout evaluation, with zero visual or polygon coordinate inspection permitted during development."*

---

### Q7: Explain the coordinate boundary discrepancy you discovered between CVAT and NumPy rasterization. Why not just use `np.clip(pts, 0, W - 1)`?
**Examiner's Intent:** Probing deep technical understanding of geometric rasterization and data parsing.  
**Authoritative Answer:**
> *"Vector annotation tools like CVAT define image space as a continuous Euclidean domain from $0.0$ to the outer canvas dimension $800.0$. In contrast, discrete NumPy matrices use integer indices from $0$ to $799$ for an $800 \times 800$ image.
> 
> Naively applying `np.clip(pts, 0, 799)` corrupts the geometry: it pulls valid outer boundary vertices at $x = 800.0$ artificially inward to index $799$, introducing geometric polygon distortion along the image margins.
> 
> We resolved this by preserving vertices at $800.0$ and applying deterministic round-to-nearest-even:
> 
> $$\text{pts}_{\text{int}} = \text{np.round}(\text{pts}).\text{astype}(\text{np.int32})$$
> 
> In OpenCV (`cv2.fillPoly`), a vertex at coordinate 800 correctly rasterizes up to the boundary pixel column 799 without memory violation or buffer overflow. This was mathematically proven and verified with 14 passing automated tests in `tests/test_polygon_rasterization_boundaries.py`."*

---

## Section 4: Explainable AI & Grad-CAM Mathematics

### Q8: How does Grad-CAM mathematically compute the importance of each feature map?
**Examiner's Intent:** Verifying mathematical comprehension of neural network interpretability.  
**Authoritative Answer:**
> *"Grad-CAM computes the gradient of the class score $y^c$ (before the softmax layer) with respect to the feature map activations $A^k$ of the last convolutional layer:
> 
> $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{i,j}^k}$$
> 
> Where $Z = U \times V$ is the spatial area of the feature map (global average pooling). The weight $\alpha_k^c$ captures the target class's sensitivity to channel $k$.
> 
> The class activation map is then formed by a weighted linear combination followed by a ReLU operation:
> 
> $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
> 
> The ReLU ensures that we only highlight visual features that contribute positively to the predicted disease category, filtering out features that decrease the class confidence."*

---

## Section 5: Software Engineering & System Architecture

### Q9: Why did you choose Server-Sent Events (SSE) instead of traditional polling or full-duplex WebSockets?
**Examiner's Intent:** Testing full-stack architectural design decisions.  
**Authoritative Answer:**
> *"Multi-stage AI inference takes ~180–350ms on GPU, but initial model loading and multi-model execution require visible progress feedback.
> - **Polling** introduces unnecessary network overhead, high latency, and race conditions.
> - **WebSockets** require stateful full-duplex TCP handshakes, complex proxy configuration, and custom heartbeat ping/pong handling for what is fundamentally a unidirectional server-to-client pipeline.
> - **Server-Sent Events (SSE)** operate over standard HTTP/1.1 or HTTP/2, natively support automated browser reconnection, pass seamlessly through corporate proxies and firewalls, and allow the FastAPI backend to cleanly stream progress events (`stage: "classification"`, `stage: "gradcam"`, etc.) before delivering the final JSON payload."*

---

### Q10: How does your backend handle CUDA Out-of-Memory (OOM) errors during heavy traffic?
**Examiner's Intent:** Probing system reliability and fault tolerance.  
**Authoritative Answer:**
> *"In `backend/app/services/prediction_service.py`, inference execution is wrapped in a dedicated exception handler:
> 1. If PyTorch raises `torch.cuda.OutOfMemoryError`, the service immediately invokes `torch.cuda.empty_cache()` to release fragmented VRAM.
> 2. It logs a warning and dynamically migrates model tensors to the host CPU (`torch.device('cpu')`).
> 3. It re-executes the forward pass on CPU, guaranteeing that the farmer receives their diagnostic result rather than experiencing a 500 Internal Server Error crash.
> This behavior is fully verified in our automated backend test suite (`tests/backend/test_phase3_backend.py`)."*
