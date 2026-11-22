# Permissively Licensed Object Detector Benchmark Plan (Phase 6/7)

**License Policy:** Preserve [src/models/detector.py](file:///D:/Main_Project/cotton_leaf_ai/src/models/detector.py) (`CottonLeafYOLODetector`, BSD-3/Apache-2.0 pure PyTorch) intact. Do **not** install `ultralytics` (`AGPL-3.0`) unless separately approved in writing.  
**Target Hardware:** `NVIDIA GeForce RTX 3050 6GB Laptop GPU` (`6143.5 MB` VRAM, PyTorch `2.15.0+cu132`, TorchVision `0.29.2+cu132` — **BSD-3-Clause**).

---

## 1. Fair Benchmark Candidates (Zero Copyleft Dependencies)

All four models below share permissive **BSD-3-Clause / Apache-2.0** licensing and require **no** new third-party package installations beyond the active `.venv`:

| # | Detector Candidate | Package & License | Initialization | Architecture Family |
|---|---|---|---|---|
| 1 | **`CottonLeafYOLODetector`** *(baseline custom)* | `src/models/detector.py` (**BSD-3 / Apache-2.0**) | Random head + optional ImageNet backbone | Anchor-free CSP + FPN/PANet multi-scale YOLO |
| 2 | **`FCOS_ResNet50_FPN`** | `torchvision.models.detection.fcos_resnet50_fpn` (**BSD-3-Clause**) | COCO-pretrained backbone + FPN (`FCOSHead` re-initialized to `num_classes=7` incl. background `0` -> `6` disease classes) | Anchor-free per-pixel dense FPN detector |
| 3 | **`RetinaNet_ResNet50_FPN_V2`** | `torchvision.models.detection.retinanet_resnet50_fpn_v2` (**BSD-3-Clause**) | COCO V2 FPN backbone + focal-loss classification head (`num_classes=7`) | One-stage focal-loss dense detector |
| 4 | **`FasterRCNN_MobileNet_V3_Large_FPN`** & **`FasterRCNN_ResNet50_FPN_V2`** | `torchvision.models.detection` (**BSD-3-Clause**) | COCO V2 / MobileNetV3 FPN + RoI box predictor (`6` disease classes + bg) | Two-stage RPN + RoIAlign benchmark reference |

---

## 2. Identical Split, Resolution & RTX 3050 Training Protocol

To guarantee a **strictly fair apples-to-apples benchmark** once CVAT bounding-box exports reach the reliable training threshold (`>= 50`–`140+` verified non-test annotated images across all 6 disease classes + Healthy negatives):

1. **Identical Group-Disjoint Split:**
   - All models train and validate on the **exact same** `train` / `validation` image-label manifest derived strictly from `artifacts/manifests/splits/{train,validation}.csv` (never touching `test.csv` or mixing `Augmented Dataset` into `validation`/`test`).
2. **Input Resolution & Hardware Parity:**
   - Standard resize/letterbox input: `640×640` RGB, batch size `4` (with `grad_accum=2` for effective batch size `8`), `torch.amp.autocast("cuda", dtype=torch.float16)`, sequential single-model GPU loading, and `torch.cuda.empty_cache()` between candidates.
   - Reproducible seed: `42`.
   - Early stopping on validation `mAP@0.50` (`patience=15`, max `60` epochs, AdamW + CosineAnnealingLR).
3. **Evaluation Metrics Logged Per Candidate (Mandatory):**
   - `mAP50` (`IoU=0.50`), `mAP50_95` (`IoU=0.50:0.95`), macro `precision`, macro `recall`, per-class `AP50` across all 6 symptomatic classes
   - False-positive rate on **Healthy Leaf negative images** (empty box precision guard)
   - Single-image GPU inference latency (ms mean / p95) & peak allocated VRAM (`MB`) on RTX 3050 6 GB
   - Checkpoint size (`MB`) and verified SHA-256 hash

---

## 3. Acceptance & Model-Selection Rule

- **Minimum threshold to begin benchmarking:** `>= 50` verified non-test bounding-box images across all disease classes + Healthy negatives (target stage-2 dataset: `140+` images).
- **Selection criterion:** Highest validation `mAP50` and Healthy-negative false-positive control under `<= 25 ms` mean GPU latency & `<= 2,500 MB` peak VRAM; final holdout evaluation runs **once** on quarantined `test.csv` bounding-box annotations.
- **Until then:** All detectors remain **`NOT_TRAINED`** with `mAP50=null`, `mAP50_95=null`.
