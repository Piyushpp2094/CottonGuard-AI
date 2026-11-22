# Software, Pretrained Weight & YOLO License Audit

**Project:** Cotton Leaf AI — Leaf Disease Analysis and Severity Estimation Using U-Net and YOLO-Based Detection  
**Audit Date:** 2026-11-10  
**Policy:** Strict Zero-Cost Local Execution; Free and Open-Source Software Only.

---

## 1. Core Application Stack Licenses

| Component | Package / Technology | Verified License | Permitted Use & Restrictions |
|---|---|---|---|
| **Deep Learning** | `torch 2.15.0+cu132`, `torchvision 0.29.2+cu132` | BSD-3-Clause | Permissive; allows local, academic, and commercial use with copyright notice. |
| **API Backend** | `fastapi`, `starlette`, `pydantic`, `uvicorn` | MIT / BSD-3-Clause | Permissive; zero external API fees or license fees. |
| **Database & ORM** | SQLite / PostgreSQL, `sqlalchemy` | Public Domain / PostgreSQL / MIT | Permissive and zero-cost. |
| **Authentication** | `pyjwt`, `bcrypt` | MIT / Apache-2.0 | Permissive; local HS256 JWT + salted bcrypt. |
| **Computer Vision** | `opencv-python-headless`, `pillow`, `scikit-learn`, `scipy`, `numpy`, `pandas` | Apache-2.0 / HPND / BSD-3-Clause | Permissive; zero copyleft requirements. |
| **Explainability & Charts** | `matplotlib` | PSF-style (Matplotlib License) | Permissive. |
| **PDF Reports** | `reportlab` | BSD-3-Clause | Permissive. |
| **Frontend UI** | `react`, `react-dom`, `react-router-dom`, `vite`, `tailwindcss`, `recharts`, `axios`, `lucide-react` | MIT / ISC | Permissive; locally bundled via Vite. |
| **Annotation Tool** | CVAT Community Edition | MIT | Locally hosted Docker Compose stack; completely free. |
| **Classification Backbone Weights** | TorchVision `EfficientNet_B0_Weights` (ImageNet-1K) | BSD-3-Clause (weights released by PyTorch/TorchVision under TorchVision terms) | Used solely to train `best_model.pt` on local SAR-CLD-2024 dataset; no external API calls. |

---

## 2. YOLO Object Detection License Verification & Selection Rationale

### Licensing Risks Reviewed
Not all open-source YOLO implementations allow unrestricted distribution or network hosting:

1. **Ultralytics (`ultralytics` YOLOv8 / YOLO11):**
   - **Verified PyPI License:** `AGPL-3.0` (`GNU Affero General Public License v3 or later`).
   - **Licensing Concern:** AGPL-3.0 contains a strict network copyleft trigger. Exposing a FastAPI web application over a network that embeds `ultralytics` triggers the obligation to open-source the entire combined application source code under AGPL-3.0 (unless a paid commercial Ultralytics Enterprise license is purchased). The associated COCO pretrained weights are also tied to Ultralytics' dual-license policy.
2. **YOLOv5 / YOLOv7 / YOLOv9 Official Repositories:**
   - **License:** `GPL-3.0` (strong copyleft when distributing combined software).
3. **YOLOX (Megvii):**
   - **Verified PyPI License:** `Apache-2.0` (permissive; allows commercial/academic distribution without copyleft source-disclosure mandates). However, PyPI package `yolox==0.3.0` has legacy build dependencies (`pycocotools`, `thop`, older Cython extensions) that conflict with modern Python/PyTorch environments.

### Selected Permissive YOLO Architecture
To respect the project title (*"YOLO-Based Detection"*) while maintaining **100% permissive BSD-3 / Apache-2.0 licensing with zero AGPL-3.0/GPL-3.0 copyleft restrictions**:

- We implement a clean, pure-PyTorch **Anchor-Free Multi-Scale YOLO Architecture** (`CottonLeafYOLODetector` in `src/models/detector.py`), inspired by YOLOX/YOLOv8 anchor-free design:
  - **Backbone:** TorchVision CSP/MobileNet/EfficientNet/ResNet feature extractor (`BSD-3-Clause`, initialized from ImageNet weights or trained from scratch, keeping VRAM footprint low on RTX 3050 6 GB).
  - **Neck:** Multi-scale Feature Pyramid Network + Path Aggregation (`FPN/PANet`, strides `P3=8`, `P4=16`, `P5=32`).
  - **Head:** Decoupled YOLO Anchor-Free Classification + CIoU Bounding-Box Regression + Objectness heads with `torchvision.ops.nms` post-processing (`BSD-3-Clause`).
- **Pretrained Weight Notice:** No AGPL/GPL COCO weights are downloaded or bundled. All detection training relies exclusively on genuine human-verified cotton leaf bounding-box annotations converted from CVAT (`artifacts/checkpoints/detection/best_yolo.pt`). If annotations are absent, detection remains safely in an `UNAVAILABLE` state.
