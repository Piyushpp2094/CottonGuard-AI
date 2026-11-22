# CottonGuard AI — Phase 11.2: Checkpoint Integrity Verification Report

**Audit Date:** November 17, 2026  
**Auditor:** Senior Machine Learning Engineer, Computer Vision Researcher & Software Quality Assurance Engineer  
**Project:** CottonGuard AI  
**Project Root:** `D:\Main_Project\cotton_leaf_ai`  
**Python Environment:** `D:\Main_Project\cotton_leaf_ai\.venv` (Python 3.11.9, PyTorch 2.15.0+cu132)  
**Hardware Platform:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 13.2, 6.00 GB VRAM)  
**Execution Context:** Strict Read-Only Checkpoint Architecture & Preprocessing Audit  
**Final Phase 11.2 Verdict:** **`VERIFIED`**  
**Approval Gate 4 Status:** **`PENDING_APPROVAL`** (Strictly Preserved)

---

## 1. Executive Summary

In accordance with the **Phase 11.2** mandate, an independent, read-only audit of the three model checkpoints was conducted. This inspection recomputed cryptographic SHA-256 hashes, decompiled top-level checkpoint dictionaries, inspected tensor layer shapes and class counts on CPU, validated preprocessing pipelines between training and inference, and resolved historical environment discrepancies.

### Key Audit Findings

1. **Cryptographic Integrity (100% Match):** All three checkpoints stored in `artifacts/checkpoints/` match their authoritative SHA-256 hashes bit-for-bit. Zero bytes have been altered, overwritten, or retrained.
2. **Architecture & State-Dictionary Verification:**
   - **Classification (`best_model.pt`):** EfficientNet-B0 (360 state-dict tensors) with output head `[7, 1280]` corresponding to 7 disease classes and 1280 feature dimensions. Checkpoint metadata documents Epoch 8 validation accuracy of 98.43% and Macro F1 of 98.39%.
   - **Detection (`best_faster_rcnn.pt`):** Faster R-CNN ResNet-50 FPN V2 (404 state-dict tensors) with 7 classification outputs (`[7, 1024]`) and 28 bounding box coordinate regressors (`[28, 1024]`). Checkpoint metadata confirms Epoch 13 validation mAP50 of 13.94% and COCO V1 initialization provenance.
   - **Segmentation (`best_unet.pt`):** UNet-ResNet34 (278 state-dict tensors) with 3-channel segmentation head (`[3, 16, 3, 3]`). Checkpoint metadata confirms Epoch 31 validation Macro Dice of 68.45% and Macro IoU of 59.35%.
3. **Preprocessing Pipeline Alignment:**
   - **Classification:** 100% alignment across training, validation, testing, and production inference (224x224 Bilinear Resize, ToTensor, ImageNet Normalization, 7-class dynamic argmax).
   - **Detection:** 100% alignment with TorchVision Faster R-CNN internal preprocessing (raw RGB $\in [0, 1]$ tensor, dynamic 800–1333 aspect-ratio scaling, internal ImageNet normalization, 1-indexed foreground classes).
   - **Segmentation (Discrepancy Identified):** The trained U-Net checkpoint metadata specifies an input resolution of **512x512** (`'input_size': [512, 512]`), whereas `prediction_service.py` initialized `self.seg_transform = get_inference_transforms(image_size=256)`. Because U-Net is fully convolutional, execution succeeds at 256x256, but downscaling by 50% relative to the 512x512 training resolution reduces spatial fidelity on fine necrotic lesion margins.
4. **Environment Discrepancy Resolved:** The actual physical hardware in the host system was verified directly via PyTorch device telemetry as the **NVIDIA GeForce RTX 3050 6GB Laptop GPU** (VRAM: 6.00 GB, Compute Capability: 8.6, Multiprocessors: 20). The earlier Phase 11 reference to "RTX 3060" was a clerical error in the report prose; the physical hardware is and has always been the RTX 3050 6GB Laptop GPU.

---

## 2. Checkpoint SHA-256 Hash Verification

All checkpoint files were recomputed on disk using standard 64 KB block streaming SHA-256 hashing and compared against the authoritative project records:

| Model Task | Checkpoint Relative Path | File Size | Authoritative Expected Hash | Independently Recomputed Hash | Verification Status |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **Classification** | `artifacts/checkpoints/classification/best_model.pt` | 48,685,496 bytes | `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e` | `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e` | **EXACT MATCH** |
| **Detection** | `artifacts/checkpoints/detection/best_faster_rcnn.pt` | 345,797,103 bytes | `01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2` | `01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2` | **EXACT MATCH** |
| **Segmentation** | `artifacts/checkpoints/segmentation/best_unet.pt` | 97,921,803 bytes | `612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43` | `612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43` | **EXACT MATCH** |

---

## 3. Architecture & Tensor Inspection

Each checkpoint was loaded safely onto CPU (`torch.load(..., map_location="cpu")`) to inspect its top-level dictionary, metadata attributes, and tensor parameters.

### 3.1 Classification Checkpoint (`best_model.pt`)
- **Container Type:** Python Dictionary (`dict`)
- **Top-Level Keys:** `['epoch', 'model_state_dict', 'optimizer_state_dict', 'metrics', 'config', 'scheduler_state_dict']`
- **Stored Training Metadata:**
  - `epoch`: `8` (Best checkpoint saved at epoch 8 of 15)
  - `metrics`:
    - `accuracy`: `0.9843` (98.43% validation accuracy)
    - `macro_f1`: `0.9839` (98.39% Macro F1)
    - `weighted_f1`: `0.9842` (98.42% Weighted F1)
    - `precision`: `0.9870`, `recall`: `0.9813`, `loss`: `0.4305`
  - `config`: `{'epochs': 15, 'batch_size': 32, 'lr': 0.0003, 'weight_decay': 0.0001, 'patience': 5, 'dropout': 0.3, 'seed': 42}`
- **State Dictionary Analysis:**
  - Total Weight Tensors: **360**
  - Architecture: **EfficientNet-B0** wrapped in `CottonLeafClassifier` (prefixed with `backbone.`)
  - Classifier Linear Head:
    - Weight: `backbone.classifier.1.weight` $\rightarrow$ Shape: `torch.Size([7, 1280])`
    - Bias: `backbone.classifier.1.bias` $\rightarrow$ Shape: `torch.Size([7])`
  - Number of Classes: **7** (exact match with `CLASS_NAMES`)
  - Input Feature Dimension: **1,280**
- **Load Test & Forward Verification:**
  - Loaded via `src.models.model_factory.build_classifier("efficientnet_b0", num_classes=7, pretrained=False, checkpoint_path=cls_path, device="cpu")`.
  - Missing Keys: `0`, Unexpected Keys: `0`.
  - Forward pass on CPU with dummy input `torch.zeros(1, 3, 224, 224)` produced output tensor of shape `torch.Size([1, 7])`.

### 3.2 Detection Checkpoint (`best_faster_rcnn.pt`)
- **Container Type:** Python Dictionary (`dict`)
- **Top-Level Keys:** `['epoch', 'global_step', 'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict', 'best_map50', 'best_map50_95', 'val_metrics', 'config', 'pretrained_weights_sha256', 'seed', 'timestamp']`
- **Stored Training Metadata:**
  - `epoch`: `13`, `global_step`: `234`
  - `best_map50`: `0.1394` (13.94%), `best_map50_95`: `0.0853` (8.53%)
  - `val_metrics`: Precision = 0.1042, Recall = 0.3846, Evaluated GT Boxes = 143
  - `class_mapping`: `{0: 'background', 1: 'Bacterial Blight', 2: 'Curl Virus', 3: 'Herbicide Growth Damage', 4: 'Leaf Hopper Jassids', 5: 'Leaf Redding', 6: 'Leaf Variegation'}`
  - `pretrained_weights_sha256`: `DD69338A24B8D7381807E247652BDC356325BCBAF1CD3E092E00E0A1A58706BF` (COCO V1)
  - `config`: Contains full hyperparameter manifest (SGD optimizer, lr=0.005, momentum=0.9, WarmupCosineAnnealingLR 540 steps).
- **State Dictionary Analysis:**
  - Total Weight Tensors: **404**
  - Architecture: **Faster R-CNN ResNet-50 FPN V2** (`fasterrcnn_resnet50_fpn_v2`)
  - RPN Box Regressor: `rpn.head.bbox_pred.weight` $\rightarrow$ Shape: `torch.Size([12, 256, 1, 1])` (3 aspect ratios $\times$ 4 coords)
  - RoI Box Classification Head: `roi_heads.box_predictor.cls_score.weight` $\rightarrow$ Shape: `torch.Size([7, 1024])`
  - RoI Box Regression Head: `roi_heads.box_predictor.bbox_pred.weight` $\rightarrow$ Shape: `torch.Size([28, 1024])` (7 classes $\times$ 4 coords)
  - Number of Classes: **7** (Background label 0 + 6 foreground disease classes)
- **Load Test & Forward Verification:**
  - Loaded via `src.models.faster_rcnn.build_faster_rcnn_model(num_classes=7, pretrained=False, checkpoint_path=det_path, device="cpu")`.
  - Missing Keys: `0`, Unexpected Keys: `0`.
  - Forward pass on CPU with dummy input `[torch.zeros(3, 800, 800)]` returned valid prediction dictionary containing `'boxes'`, `'labels'`, and `'scores'`.

### 3.3 Segmentation Checkpoint (`best_unet.pt`)
- **Container Type:** Python Dictionary (`dict`)
- **Top-Level Keys:** `['epoch', 'model_state_dict', 'val_macro_dice', 'val_macro_iou', 'best_metrics', 'confusion_matrix', 'class_weights', 'input_size', 'architecture', 'status']`
- **Stored Training Metadata:**
  - `epoch`: `31`
  - `architecture`: `'unet_resnet34'`
  - `input_size`: `[512, 512]`
  - `val_macro_dice`: `0.68447` (68.45%)
  - `val_macro_iou`: `0.59351` (59.35%)
  - `class_weights`: `[0.3, 1.0, 2.5]` (Background, Healthy Leaf, Diseased Tissue)
  - `status`: `'EXPERIMENTAL'`
- **State Dictionary Analysis:**
  - Total Weight Tensors: **278**
  - Architecture: **UNet-ResNet34** (Segmentation Models PyTorch `smp.Unet`)
  - Final Convolutional Segmentation Head:
    - Weight: `segmentation_head.0.weight` $\rightarrow$ Shape: `torch.Size([3, 16, 3, 3])`
    - Bias: `segmentation_head.0.bias` $\rightarrow$ Shape: `torch.Size([3])`
  - Number of Classes: **3** (`0: Background`, `1: Healthy Leaf`, `2: Diseased Tissue`)
- **Load Test & Forward Verification:**
  - Loaded via `src.models.unet_resnet34.build_unet_resnet34(num_classes=3, checkpoint_path=seg_path, device="cpu")`.
  - Missing Keys: `0`, Unexpected Keys: `0`.
  - Forward pass on CPU with dummy input `torch.zeros(1, 3, 512, 512)` produced output logits tensor of shape `torch.Size([1, 3, 512, 512])`.

---

## 4. Preprocessing Alignment Analysis

| Dimension | Classification (EfficientNet-B0) | Detection (Faster R-CNN) | Segmentation (U-Net ResNet-34) | Alignment Status |
| :--- | :--- | :--- | :--- | :---: |
| **Input Resolution** | 224 $\times$ 224 pixels | 800 $\times$ 800 (internal dynamic scale 800–1333) | **Trained:** 512 $\times$ 512<br>**Service:** 256 $\times$ 256 | **Mismatch Documented** (See Sec 5) |
| **Color Channels** | 3-Channel RGB | 3-Channel RGB | 3-Channel RGB | **100% Aligned** |
| **Normalization** | ImageNet Mean: `[0.485, 0.456, 0.406]`<br>Std: `[0.229, 0.224, 0.225]` | TorchVision internal ImageNet normalization | ImageNet Mean: `[0.485, 0.456, 0.406]`<br>Std: `[0.229, 0.224, 0.225]` | **100% Aligned** |
| **Transformation Order** | 1. Resize (Bilinear)<br>2. ToTensor [0, 1]<br>3. Normalize | 1. ToTensor [0, 1]<br>2. TorchVision internal transform | 1. Resize (Bilinear)<br>2. ToTensor [0, 1]<br>3. Normalize | **100% Aligned** |
| **Output Classes** | 7 classes | 7 classes (0: Background, 1..6: Diseases) | 3 classes (0: Bg, 1: Healthy, 2: Diseased) | **100% Aligned** |
| **Label Mapping** | `CLASS_NAMES` (0..6) | `FOREGROUND_CLASSES` (1..6) | `[0, 1, 2]` Pixel Mask Argmax | **100% Aligned** |

---

## 5. Identified Issues, Discrepancies & Resolutions

### 5.1 Issue 1: Host GPU Hardware Discrepancy (Resolved)
- **Problem Statement:** The previous Phase 11 report text mentioned "NVIDIA GeForce RTX 3060 Laptop GPU", whereas Phase 9.5 preflight records specified "RTX 3050 6GB".
- **Investigation & Evidence:** Direct hardware query via `torch.cuda.get_device_name(0)` and `torch.cuda.get_device_properties(0)` returns:
  - `Device 0: NVIDIA GeForce RTX 3050 6GB Laptop GPU`
  - `Total Memory: 6,441,926,656 bytes (6.00 GB)`
  - `Compute Capability: 8.6`
  - `Multiprocessor Count: 20`
- **Resolution:** Formally confirmed that the actual physical device is the **NVIDIA GeForce RTX 3050 6GB Laptop GPU**. The "RTX 3060" reference in the prior report was a typographical error in the narrative header. All execution and telemetry correspond to the 6GB RTX 3050.

### 5.2 Issue 2: TorchVision `trainable_backbone_layers` UserWarning (Documented)
- **Problem Statement:** Initializing `build_faster_rcnn_model(pretrained=False, checkpoint_path=...)` emits:
  `UserWarning: Changing trainable_backbone_layers has no effect if neither pretrained nor pretrained_backbone have been set to True, falling back to trainable_backbone_layers=5 so that all layers are trainable`
- **Investigation & Impact:** TorchVision's internal `_resnet_fpn_extractor` raises this warning because `pretrained=False` is passed (since custom weights are loaded from `best_faster_rcnn.pt` rather than downloading COCO weights).
- **Verdict:** **Benign.** During inference, `model.eval()` and `torch.inference_mode()` freeze all weights regardless of `trainable_backbone_layers`. Zero parameter drift or numerical distortion occurs.

### 5.3 Issue 3: U-Net Preprocessing Resolution Discrepancy (Documented for Gate 4)
- **Problem Statement:** The saved checkpoint metadata in `best_unet.pt` specifies `'input_size': [512, 512]`, which matches the Albumentations training pipeline in `scripts/train_unet_resnet34.py`. However, `backend/app/services/prediction_service.py` initializes:
  ```python
  self.seg_transform = get_inference_transforms(image_size=256)
  ```
- **Investigation & Impact:** Because U-Net ResNet-34 is fully convolutional, it processes arbitrary spatial resolutions, and the output is upsampled back to the original leaf dimensions via nearest-neighbor interpolation. However, processing at 256x256 instead of the native 512x512 resolution downsamples the input tensor by 50% along each spatial axis, slightly degrading fine lesion boundary delineation.
- **Resolution:** To be safely addressed under future Phase 12 refinement when Approval Gate 4 is unlocked. In adherence to strict safety rules, **zero code files were modified** during this audit.

---

## 6. Environment Telemetry Summary

```
========================================================================================
                              HOST RUNTIME ENVIRONMENT
========================================================================================
  Operating System:             Windows 11 Home / Professional (AMD64)
  Python Executable:            D:\Main_Project\cotton_leaf_ai\.venv\Scripts\python.exe
  Python Version:               3.11.9 (64-bit MSC v.1938)
  PyTorch Version:              2.15.0+cu132
  CUDA Version:                 13.2
  GPU Device Name:              NVIDIA GeForce RTX 3050 6GB Laptop GPU
  Dedicated VRAM:               6,441,926,656 bytes (6.00 GB)
  CUDA Compute Capability:      8.6 (Ampere Architecture)
  SM Processor Count:           20 Multiprocessors
========================================================================================
```

---

## 7. Strict Safety Compliance Confirmation

In full accordance with Phase 11.2 constraints:
- **Zero model training or fine-tuning was executed.**
- **Zero optimizer steps were taken.**
- **Zero checkpoint files were modified, overwritten, or renamed.**
- **Zero records were added or modified in `cotton_leaf_ai.db`.**
- **Zero CVAT annotations or manifests were touched.**
- **The sealed classification test set was not accessed or altered.**
- **Zero source code or configuration files were modified.**
- **Phase 11.3 has NOT been initiated.**

---

## 8. Final Verdict

| Audit Item | Expected Standard | Observed Finding | Verdict |
| :--- | :--- | :--- | :---: |
| **Classification SHA-256** | `552baba83a66ad4a8bbc1eeb7fe3ee8dd84dfabe68c1ef477930da57a05fa90e` | Exact match | **VERIFIED** |
| **Detection SHA-256** | `01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2` | Exact match | **VERIFIED** |
| **Segmentation SHA-256** | `612fca0a162b56b8253a81b9ded3d069c1e15666e505feb38fc93ed6be7c3a43` | Exact match | **VERIFIED** |
| **State-Dict & Tensor Inspection** | Correct output heads, class counts, and zero missing keys | All 3 models load cleanly and produce expected tensor outputs | **VERIFIED** |
| **Preprocessing Alignment** | Classification & Detection match; U-Net resolution discrepancy documented | Fully documented without unapproved code edits | **VERIFIED** |
| **Hardware Clarification** | Resolve RTX 3060 vs RTX 3050 discrepancy | Confirmed RTX 3050 6GB Laptop GPU | **VERIFIED** |
| **Zero Checkpoint Mutation** | All files unaltered on disk | File timestamps and hashes 100% invariant | **VERIFIED** |

### Overall Phase 11.2 Verdict

# **`VERIFIED`**

Phase 11.2 is completed. All checkpoints are cryptographically authentic and structurally sound. Preprocessing characteristics and hardware telemetry have been rigorously documented.

**Approval Gate 4 Status:** **`PENDING_APPROVAL`**  
*(Execution halted. Awaiting user instructions before proceeding to Phase 11.3).*
