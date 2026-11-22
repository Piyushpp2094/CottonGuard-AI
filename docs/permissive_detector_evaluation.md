# Permissively Licensed Object Detector Alternatives Evaluation (Phase 6)

**Status:** Active research reference for CottonGuard AI Phase 6  
**Constraint:** Do **not** install `ultralytics` or adopt AGPL-3.0 copyleft obligations without separate written user approval. Keep existing [src/models/detector.py](file:///D:/Main_Project/cotton_leaf_ai/src/models/detector.py) intact.

---

## 1. Environment Check (Observed Installed Packages in `.venv`)

Observed directly in `D:\Main_Project\cotton_leaf_ai\.venv`:
- `torch==2.15.0+cu132` (**BSD-3-Clause**)
- `torchvision==0.29.2+cu132` (**BSD-3-Clause**) — includes native `torchvision.models.detection` architectures with pretrained ImageNet/COCO backbones (`FCOS`, `RetinaNet`, `FasterRCNN`, `SSD300`, `SSDLite320`) under BSD-3-Clause!
- `timm==1.0.31` (**Apache-2.0**)
- `segmentation-models-pytorch==0.6.0` (**MIT**)
- `albumentations==2.0.8` (**MIT**)
- `ultralytics`: **NOT installed** (`AGPL-3.0` network copyleft avoided).

---

## 2. Architecture Comparison for 6-Class Cotton Leaf Lesion Detection (RTX 3050 6 GB)

| Candidate Architecture | Source / Library | License | Pretrained Weights Available Locally/BSD-Apache | RTX 3050 6 GB Suitability | Pros & Limitations |
|---|---|---|---|---|---|
| **1. `CottonLeafYOLODetector` (Existing Custom)** | `src/models/detector.py` (PyTorch/TorchVision CSP + PANet) | **BSD-3-Clause / Apache-2.0** | ImageNet backbone optional; zero external copyleft | Native FP16, tiny VRAM (`640×640`, batch `4–8`) | Preserves project's pure-PyTorch YOLO anchor-free architecture with zero third-party license risk; needs more annotated samples than COCO-pretrained detectors because detection head initializes from scratch. |
| **2. `FCOS_ResNet50_FPN` (Anchor-Free TorchVision)** | `torchvision.models.detection.fcos_resnet50_fpn` | **BSD-3-Clause** | COCO-pretrained backbone + FPN weights (`BSD-3-Clause`) | ~32M params; runs smoothly at `640×640` with batch `4` + AMP on 6 GB VRAM | Strongest anchor-free drop-in alternative already installed in `.venv` (`0` new packages); transfers well onto small labeled agricultural datasets (`70–200` images) due to COCO FPN weights. |
| **3. `RetinaNet_ResNet50_FPN_V2` / `FasterRCNN_MobileNet_V3_Large_FPN`** | `torchvision.models.detection` | **BSD-3-Clause** | COCO V2 / MobileNetV3 weights (`BSD-3-Clause`) | Low–Moderate VRAM (`batch=4–8` FP16 on 6 GB) | Mature TorchVision CIoU/GIoU/Focal-Loss implementations; `MobileNetV3` variant offers sub-15 ms mobile/edge latency. |
| **4. `YOLOX` / `RT-DETR` (`transformers` or `timm`)** | Official Megvii YOLOX / HuggingFace `transformers` (`RTDetrForObjectDetection`) | **Apache-2.0** | COCO Apache-2.0 checkpoints | Moderate VRAM (`batch=4` at `640×640`) | `RT-DETR` (`Apache-2.0`) delivers transformer-grade small-lesion recall without NMS; requires adding `transformers` (`Apache-2.0`) or `onnxruntime`. |
| **5. `YOLOv8n` (`ultralytics`) — Gated / Deferred** | `ultralytics` PyPI package | **GNU AGPL-3.0** | Ultralytics dual-licensed weights | Very fast convergence & low VRAM (~3.2M params) | **Copyleft Blocker:** Serving `ultralytics` inside the FastAPI web backend over a network triggers AGPL-3.0 full-application source disclosure unless commercial Ultralytics Enterprise licensing is purchased. Remains **blocked/not installed** until user decision. |

---

## 3. Recommendation for Phase 6 / Phase 7 Decision Gate

1. **Immediate (Phase 6):** Complete the **21-image pilot** and scale annotations to **70+ verified images** using the split-clean CVAT protocol. Neither custom YOLO nor TorchVision FCOS nor YOLOv8n can train meaningfully with `0` verified bounding boxes.
2. **If user keeps zero-AGPL permissive licensing (Recommended for Web Deployment):**
   - Keep `CottonLeafYOLODetector` intact **and** optionally benchmark `torchvision.models.detection.fcos_resnet50_fpn` (`BSD-3-Clause`, already installed, anchor-free) vs custom YOLO on the exact same train/validation annotated split once `>= 50` annotated images exist.
3. **If user explicitly authorizes AGPL-3.0 compliance (Academic/Open-Source AGPL Release):**
   - Install `ultralytics` separately and execute the existing `scripts/train_yolov8.py` CLI on `configs/yolo_config.yaml`.
