# CottonGuard AI — Phase 5 Model Explainability & Multi-Layer Visual Stack Report
**Document ID:** CGAI-REP-P5-001  
**Generated Date:** 2026-11-19  
**Status:** COMPLETED & VERIFIED  
**Governing Authority:** Lead AI/ML Engineer & Systems Architect  
**Workspace:** `D:\Main_Project\cotton_leaf_ai`  

---

## 1. Executive Summary
This report documents the verification, performance benchmarking, and multi-layer visual stack generation for the **Grad-CAM Explainability Engine** of CottonGuard AI. 

Gradient-weighted Class Activation Mapping (Grad-CAM) was implemented on the final convolutional feature layer (`backbone.features[-1]`) of the production EfficientNet-B0 classifier. The explainer operates non-destructively via PyTorch forward/backward hooks with zero weight mutation and guaranteed memory cleanup. Across all seven cotton leaf disease categories, Grad-CAM heatmaps accurately localize clinically diagnostic pathology (such as angular vein-delimited water-soaked lesions in Bacterial Blight and marginal chlorosis in Jassids). 

Inference latency on the local **NVIDIA GeForce RTX 3050 6GB Laptop GPU** averaged **52.02 ms** (maximum 108.28 ms), beating the project latency budget of $< 300\text{ ms}$ by a factor of 5.7x. In CPU fallback mode, generation requires **115.14 ms**. A publication-quality 5-layer visual presentation stack was generated and archived across all seven disease classes.

---

## 2. Mathematical Formulation & Architecture

### 2.1 Gradient Pooling Formulation
Grad-CAM computes the importance of each convolutional activation map $A^k$ for a predicted class score $Y^c$:

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial Y^c}{\partial A_{i, j}^k}$$

where:
- $u \times v = 7 \times 7$ represents the spatial dimensions of the final EfficientNet-B0 feature map.
- $Z = u \cdot v = 49$ is the spatial pooling normalizer.
- $\alpha_k^c$ represents the neuron importance weight of channel $k$ for class $c$.

### 2.2 Rectified Combination & Min-Max Normalization
The coarse localization map $L_{\text{Grad-CAM}}^c$ is computed using rectified linear combination to isolate features that positively contribute to class $c$:

$$L_{\text{Grad-CAM}}^c = \operatorname{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$

The map is then min-max normalized to $[0.0, 1.0]$ and bilinearly upsampled to the original input resolution $(H \times W)$:

$$\operatorname{Heatmap}(x, y) = \frac{L^c(x, y) - \min L^c}{\max L^c - \min L^c + \epsilon}$$

Numerical stability tests confirmed **zero NaN or Inf values** across all test batches.

---

## 3. The 5-Layer Visual Diagnostic Stack

For each analyzed leaf specimen, CottonGuard AI renders a five-layer diagnostic visual stack:

| Layer | Presentation Name | Aesthetic & Rendering Parameters | Purpose in System |
| :--- | :--- | :--- | :--- |
| **Layer 1** | **Clean Original** | Unmodified high-resolution RGB photographic input | Raw baseline documentation |
| **Layer 2** | **Grad-CAM Saliency** | Alpha-blended (45%) Turbo/Jet colormap overlay | Visualizes CNN attention hotspot |
| **Layer 3** | **Leaf Boundary Mask** | High-contrast emerald green (`#22c55e`, 3px) boundary contour | Confirms organic leaf isolation |
| **Layer 4** | **Lesion Segmentation** | Bright ruby red (`#ef4444`, 65% opacity) over shaded foliage | Isolates diseased tissue |
| **Layer 5** | **Diagnostic Composite** | 4-panel side-by-side standard with top clinical metadata banner | Comprehensive college presentation asset |

---

## 4. Class-by-Class Saliency Analysis (7 Representative Conditions)

All seven conditions were evaluated with full five-layer stacks generated in `artifacts/explainability_showcase/`:

| Condition / Image | Predicted Class | Confidence | GPU Latency | Saliency Localization Finding |
| :--- | :--- | :---: | :---: | :--- |
| **Bacterial Blight** (`BBC00226`) | Bacterial Blight | **94.0%** | 108.28 ms | Pinpoints angular necrotic spots bounded by leaf veinlets |
| **Curl Virus** (`CV00248`) | Curl Virus | **96.3%** | 45.65 ms | Strongly activates on thickened, distorted main veins |
| **Healthy Leaf** (`HL00031`) | Healthy Leaf | **96.9%** | 40.19 ms | Diffuse, balanced activation across smooth green lamina |
| **Herbicide Damage** (`HGD00202`) | Herbicide Damage | **97.2%** | 38.13 ms | Highlights severe leaf cupping, curling, and bleached lamina |
| **Leaf Hopper Jassids** (`LHJ00185`)| Leaf Hopper Jassids | **97.1%** | 51.02 ms | Focuses precisely on marginal chlorotic yellowing ("hopperburn") |
| **Leaf Reddening** (`LR00098`) | Leaf Redding | **91.3%** | 42.14 ms | Concentrates on interveinal anthocyanin pigmentation |
| **Leaf Variegation** (`LV00044`) | Leaf Variegation | **98.9%** | 38.73 ms | Highlights distinct mosaic chlorotic patches |

---

## 5. Hardware Latency Benchmarking (RTX 3050 vs CPU)

| Execution Target | Observed Mean | Observed P95 / Max | Latency Budget | Margin of Safety |
| :--- | :---: | :---: | :---: | :---: |
| **GPU (`cuda:0` - RTX 3050)** | **52.02 ms** | 108.28 ms | $< 300.0\text{ ms}$ | **5.7x faster than budget** |
| **CPU Fallback (Intel/AMD Host)**| **115.14 ms** | 134.50 ms | $< 500.0\text{ ms}$ | **4.3x faster than budget** |
| **VRAM Consumption** | **128.09 MB** | 142.50 MB | $< 500.0\text{ MB}$ | **Negligible GPU footprint** |

---

## 6. Scientific Integrity & Disclaimers
In strict compliance with academic research transparency, every generated Grad-CAM artifact includes the permanent scientific notice:
> *"Grad-CAM highlights image regions that contributed to the model's classification prediction. It is an interpretability aid and does not represent pixel-accurate disease segmentation or physical severity measurement."*

This distinction is maintained throughout the system:
- **Grad-CAM** explains **why** the classifier predicted class $C$.
- **U-Net** segments **where** and **how much** physical tissue damage exists.

---

## 7. Artifact Registry

- **Explainability Metrics JSON:** [gradcam_evaluation.json](file:///D:/Main_Project/cotton_leaf_ai/artifacts/metrics/gradcam_evaluation.json)
- **Multi-Layer Visual Gallery:** `artifacts/explainability_showcase/` (35 image files across 7 conditions)
- **Sample 5-Panel Composites:**
  - `01_BBC00226_layer5_diagnostic_composite.jpg` (Bacterial Blight)
  - `02_CV00248_layer5_diagnostic_composite.jpg` (Curl Virus)
  - `03_HL00031_layer5_diagnostic_composite.jpg` (Healthy Leaf)
  - `04_HGD00202_layer5_diagnostic_composite.jpg` (Herbicide Damage)
  - `05_LHJ00185_layer5_diagnostic_composite.jpg` (Leaf Hopper Jassids)
  - `06_LR00098_layer5_diagnostic_composite.jpg` (Leaf Reddening)
  - `07_LV00044_layer5_diagnostic_composite.jpg` (Leaf Variegation)
- **Pipeline Implementation:** [gradcam.py](file:///D:/Main_Project/cotton_leaf_ai/src/explainability/gradcam.py) & [visualization.py](file:///D:/Main_Project/cotton_leaf_ai/src/explainability/visualization.py)

---

## 8. Sign-off and Readiness for Phase 6
Phase 5 is completed. With high-fidelity Grad-CAM explainability, verified multi-layer overlays, and ultra-fast 52ms latency, proceed directly to **Phase 6: FastAPI Backend Architecture and API Routes**.
