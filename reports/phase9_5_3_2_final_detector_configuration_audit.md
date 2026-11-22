# CottonGuard AI — Phase 9.5.3.2 Technical Report
## Final Detector Configuration and Initialization Verification

**Date:** 2026-11-17  
**Project:** CottonGuard AI (`cotton_leaf_ai`)  
**Phase:** Phase 9.5.3.2 — Final Detector Configuration & Initialization Verification  
**Author:** Autonomous Senior AI/ML & Computer Vision Engineer  
**Status / Executive Verdict:** **`A. CONFIGURATION VERIFIED — READY FOR EXPLICIT TRAINING APPROVAL`**  
**Formal Safety Governance:** Strictly non-executing configuration audit. Zero training epochs executed. Zero optimizer updates. Zero checkpoint modifications. Zero database updates. CVAT and sealed test set untouched.

---

## 1. Executive Summary & Verdict

Phase 9.5.3.2 performs the final technical and source-code verification of the Faster R-CNN detection pipeline against the proposed experiment run card ahead of Approval Gate 4. Every potential ambiguity—including class mappings, model weight initialization provenance, optimizer/scheduler source code synchronization, dataset staging, GPU memory uncertainties, background execution boundaries, and evaluation reporting—has been audited, resolved, and empirically verified.

### Key Audit Findings:
1. **Class Mapping Standardized:**
   - Authoritative mapping verified: Class 0 = Background, Class 1 = Bacterial Blight, Class 2 = Curl Virus, Class 3 = Herbicide Growth Damage, Class 4 = Leaf Hopper Jassids, Class 5 = Leaf Redding, Class 6 = Leaf Variegation (`num_classes = 7`).
   - Healthy Leaf images ($N=15$) are negative controls (0 detection targets), NOT foreground classes.
   - Ambiguous historical comments (such as `0: healthy/background`) have been eradicated.
   - Label alignment between prediction post-processing (`p_lbl - 1`) and evaluation ground truth has been verified via synthetic integration test (1.0000 mAP@0.50).
2. **Model Initialization Provenance Established:**
   - **Option B (COCO-pretrained Faster R-CNN ResNet-50 FPN V2 with adapted 7-class FastRCNNPredictor)** is formally recommended.
   - Weights sourced directly from local PyTorch cache: `fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth` (SHA-256: `DD69338A24B8D7381807E247652BDC356325BCBAF1CD3E092E00E0A1A58706BF`).
   - Option A (ImageNet backbone alone) was rejected due to lack of multiscale anchor pre-training. Option C (`best_faster_rcnn.pt`) was rejected to prevent carrying over negative supervision bias against diffuse symptoms.
3. **Optimizer & Scheduler Source-Code Synchronization:**
   - Resolved the SGD vs. AdamW discrepancy: `scripts/train_faster_rcnn.py` has been updated to dynamically instantiate `torch.optim.AdamW(lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4)` from configuration.
   - The per-batch `LambdaLR` scheduler mathematically implements a 105-step linear warmup ($3.33 \times 10^{-5} \to 1.0 \times 10^{-4}$) followed by a 945-step cosine decay ($1.0 \times 10^{-4} \to 1.0 \times 10^{-6}$), matching the exact 1,050 total steps across 30 epochs (35 steps/epoch).
4. **Mixed Precision (AMP) Step Detection & Skipped Step Instrumentation:**
   - Verified that scheduler advancement is strictly guarded by `optimizer_stepped = (scale_after >= scale_before)`.
   - Added `epoch_skipped_steps` and `total_skipped_steps` counter instrumentation to track AMP gradient overflow events.
5. **Tier 2 Ignore-Region Handling Verified:**
   - `IgnoreAwareRPN` and `IgnoreAwareRoIHeads` verified with IoU/IoA threshold 0.30 and `~is_positive` invariant in `src/models/ignore_region_faster_rcnn.py`.
   - 13/13 unit tests passed in `tests/models/test_ignore_region_faster_rcnn.py`.
6. **Dataset Staging & Loader Verification:**
   - Staged isolated dataset directory `data/detection_consolidated_100/` containing all 100 images, 1,009 focal lesion boxes, and 130 diffuse ignore boxes.
   - `CottonDetectionDataset` tested and verified: 70 train images (683 focal, 62 ignore) and 30 validation images (326 focal, 68 ignore).
   - Group isolation verified: 0 image overlap between train and val; perfect class balance (10 train images per class).
7. **Dry-Run Check Successful:**
   - `python scripts/train_faster_rcnn.py --config configs/phase9_5_faster_rcnn_retrain_config.yaml --dry-run-check` passed with exit code 0.
   - Output paths isolated to `artifacts/checkpoints/detection/phase9_5_retrain/`, protecting baseline checkpoint `best_faster_rcnn.pt` (SHA-256: `01d82b32...`).
8. **Final Preflight Verdict:** **`A. CONFIGURATION VERIFIED — READY FOR EXPLICIT TRAINING APPROVAL`**.

---

## 2. Environment Health & Process Diagnostics

### 2.1 Runtime Environment
- **Operating System:** Windows 11 Home (x86_64, Kernel 10.0.26100)
- **Python Version:** 3.11.9 (64-bit) at `D:\Main_Project\cotton_leaf_ai\.venv\Scripts\python.exe`
- **PyTorch / TorchVision:** PyTorch 2.15.0+cu132, TorchVision 0.29.2+cu132, CUDA 13.2 available (`torch.cuda.is_available() == True`)
- **GPU Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6,143.5 MB total dedicated VRAM, 20 SMs, Compute Capability 8.6)
- **Host RAM:** 23.64 GB total, 5.57 GB available (76.4% utilization)

### 2.2 Investigation and Safe Resolution of Hung Pytest Process
During earlier testing, background task `task-8880` running pytest appeared unresponsive for >28 minutes. An autonomous investigation was conducted:
1. **Process Inspection:** Background task `task-8880` was inspected. The command being executed was `pytest tests/backend tests/test_backend_api.py`.
2. **Exact Point of Stoppage:** Execution was frozen in `cotton_leaf_ai/tests/backend/test_phase3_backend.py` inside `TestPhase3Backend.setUpClass()` at the instantiation of Starlette `TestClient(app)`.
3. **Traceback & Root Cause Diagnosis:**
   - Starlette's `TestClient` uses `anyio.from_thread.start_blocking_portal()`.
   - `anyio` creates an internal pair of loopback sockets on `127.0.0.1` and calls `connect()`.
   - Inside the Windows AppContainer sandbox, local socket connections are blocked by the OS network isolation boundary, raising `PermissionError: [WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions`.
   - Because `anyio` catches certain socket errors and retries without a bounded timeout, the worker thread entered an indefinite wait state on the kernel socket handle.
4. **Resolution:**
   - Background task `task-8880` was safely cancelled via the task management interface.
   - No zombie processes or orphan handles remain active.
   - **Boundary Established:** All PyTorch model architectures, loss functions, Tier 2 ignore-region masking, dataset loaders, and annotation validators execute 100% inside the standard sandbox with zero network requirements. Backend HTTP integration tests using Starlette `TestClient` require unsandboxed execution where local loopback socket binding is permitted.

---

## 3. Project Structure & Source-of-Truth Architecture

To ensure total code alignment ahead of training, the 5 core source files governing detection were cross-audited:

| Component | Source File Path | Architectural Role & Implementation Details | Resolved Discrepancies |
| :--- | :--- | :--- | :--- |
| **Baseline Architecture** | `src/models/faster_rcnn.py` | Factory `build_faster_rcnn()` using standard TorchVision `fasterrcnn_resnet50_fpn_v2`. Replaces final box predictor head with 7-class linear layer. | Served as baseline reference. Maintained unmodified for backward compatibility. |
| **Ignore-Region Architecture** | `src/models/ignore_region_faster_rcnn.py` | Subclasses `RegionProposalNetwork` into `IgnoreAwareRPN` and `RoIHeads` into `IgnoreAwareRoIHeads`. Masks anchors and proposals with $\text{IoA} \ge 0.30$ against Tier 2 diffuse regions while enforcing `~is_positive` invariant. Factory `build_faster_rcnn_model()`. | Verified 13/13 unit tests pass. Parameterized ignore thresholds (default 0.30) exposed to config. |
| **Dataset Loader** | `src/data/detection_dataset.py` | `CottonDetectionDataset` parses YOLO `.txt` bounding boxes and optional `ignore_labels/` YOLO `.txt` files. Scales normalized $[x_c, y_c, w, h]$ to absolute $[x_1, y_1, x_2, y_2]$. Converts Healthy Leaf samples to empty target tensors. | Verified loading of 70 train / 30 val split. Group isolation verified. Healthy controls correctly emit 0 boxes. |
| **Training Pipeline** | `scripts/train_faster_rcnn.py` | Full training orchestration: YAML config loader, optimizer dispatch (AdamW/SGD), `LambdaLR` scheduler, AMP `autocast` + `GradScaler`, validation metric computation, early stopping, checkpoint serialization. | **Hardening Applied:** Dispatched AdamW dynamically; parameterized dataset assertions; wired `enable_ignore_regions`; added AMP skipped-step counters. |
| **Authoritative Config** | `configs/phase9_5_faster_rcnn_retrain_config.yaml` | Master experiment configuration (`EXP-FRCNN-20261117-01`). Defines paths, 7-class mapping, AdamW hyperparameters, cosine schedule (105 warmup / 1,050 total steps), and isolated checkpoint paths. | Created and verified via `--dry-run-check`. Baseline checkpoint protected. |

---

## 4. Correct Detector Class Mapping & Negative Supervision

### 4.1 Authoritative Faster R-CNN Mapping
The Faster R-CNN ResNet-50 FPN V2 detector operates on exactly **7 classes** (`num_classes = 7`):

| Class ID | Semantic Entity | Role in Detector | CVAT Source Attribute | Output Projection |
| :---: | :--- | :--- | :--- | :---: |
| **0** | **Background** | Non-lesion / Negative Supervision | None (RPN background anchors) | $p_{\text{bg}}$ |
| **1** | **Bacterial Blight** | Foreground Disease Lesion | `disease_type="Bacterial Blight"` | $p_1$ |
| **2** | **Curl Virus** | Foreground Disease Lesion | `disease_type="Curl Virus"` | $p_2$ |
| **3** | **Herbicide Growth Damage** | Foreground Disease Lesion | `disease_type="Herbicide Growth Damage"` | $p_3$ |
| **4** | **Leaf Hopper Jassids** | Foreground Disease Lesion | `disease_type="Leaf Hopper Jassids"` | $p_4$ |
| **5** | **Leaf Redding** | Foreground Disease Lesion | `disease_type="Leaf Redding"` | $p_5$ |
| **6** | **Leaf Variegation** | Foreground Disease Lesion | `disease_type="Leaf Variegation"` | $p_6$ |

### 4.2 Role of Healthy Leaf Negative Controls
- **Healthy Leaf is NOT a detection class.**
- Healthy Leaf images ($N=15$, exactly 10 in train and 5 in val) are negative controls.
- When parsed by `src/data/detection_dataset.py`, Healthy Leaf samples emit:
  - `target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)`
  - `target["labels"] = torch.zeros((0,), dtype=torch.int64)`
  - `target["ignore_boxes"] = torch.zeros((0, 4), dtype=torch.float32)`
- During RPN anchor assignment, all anchors generated across the healthy leaf lamina have an IoU of $0.0$ with ground-truth boxes. Consequently, they provide negative supervision (class 0), training the network to suppress false positives on asymptomatic cotton foliage.

### 4.3 Metric Class Mapping and Prediction Post-Processing
TorchVision's Faster R-CNN outputs bounding boxes with predicted class labels in the range $\{1..6\}$ (excluding background 0). In `scripts/train_faster_rcnn.py`:
- In `evaluate_detection_epoch()`, model predictions are shifted to 0-based indices via `int(p_lbl - 1)` before being collected into `sample_preds`.
- Ground truth labels (already $1..6$) are similarly shifted via `int(gt_lbl - 1)` before being collected into `sample_targets`.
- Both lists are passed to `compute_detection_metrics(sample_preds, sample_targets, num_classes=6)`.
- Verified via `scripts/test_metric_mapping.py`: A synthetic 6-class batch was evaluated, confirming that predictions for classes $1..6$ match targets $1..6$ perfectly, yielding 1.0000 AP for each class and 1.0000 mAP@0.50.

---

## 5. Model Initialization Provenance

Three candidate weight initialization strategies were inspected and compared:

### 5.1 Comparative Analysis

| Feature | Option A: ImageNet ResNet-50 Backbone | Option B: COCO Pretrained Faster R-CNN V2 (RECOMMENDED) | Option C: Pre-existing `best_faster_rcnn.pt` Checkpoint |
| :--- | :--- | :--- | :--- |
| **Weights Source** | `torchvision.models.ResNet50_Weights.IMAGENET1K_V2` | `torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1` | `artifacts/checkpoints/detection/best_faster_rcnn.pt` |
| **Local File Path** | N/A (requires network download) | `C:\Users\Haker's pc\.cache\torch\hub\checkpoints\fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth` | `D:\Main_Project\cotton_leaf_ai\artifacts\checkpoints\detection\best_faster_rcnn.pt` |
| **Weights SHA-256** | N/A | `DD69338A24B8D7381807E247652BDC356325BCBAF1CD3E092E00E0A1A58706BF` | `01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2` |
| **Backbone Weights** | Pretrained on ImageNet-1K | Pretrained on COCO train2017 | Trained for 13 epochs on 35 cotton images |
| **FPN & RPN Weights** | **Randomly Initialized** | **Pretrained on COCO train2017** | Trained for 13 epochs on 35 cotton images |
| **Box Predictor Head** | Randomly initialized (7 classes) | Randomly initialized (7 classes) | Trained for 13 epochs (7 classes) |
| **Pretrained Discarded** | Classification head (`fc`) | 91-class `cls_score` and `bbox_pred` projections | None |
| **Newly Initialized** | ~16,200,000 parameters (FPN + RPN + Head) | **35,875 parameters (0.082% of model)** | 0 parameters |
| **State-Dict Compatibility** | Backbone only | 100% backbone, FPN, and RPN | 100% full architecture |
| **Academic Verdict** | **REJECTED (Under-constrained)** | **RECOMMENDED (Gold Standard)** | **REJECTED (Contaminated Supervision)** |

### 5.2 Detailed Justification
1. **Why Option A was rejected:** Initializing FPN and RPN from scratch requires millions of object bounding-box instances to learn multiscale anchor proposal generation. Training these layers from scratch on 70 images results in anchor collapse or severe overfitting.
2. **Why Option C was rejected:** The existing `best_faster_rcnn.pt` model was trained in Phase 9.2c *without* Tier 2 ignore-region supervision. In that run, diffuse disease patches in Tasks 4 and 5 were supervised as negative background. Fine-tuning from that checkpoint inherits a learned suppression bias against diffuse symptoms. Furthermore, initializing from COCO weights preserves scientific attribution: it enables a rigorous head-to-head comparison against the baseline to isolate the exact impact of the expanded dataset and Tier 2 ignore masking.
3. **Why Option B is recommended:** COCO V1 weights provide a fully trained feature pyramid and proposal network. Adapting the final `FastRCNNPredictor` (1024 $\to$ 7 for classification, 1024 $\to$ 28 for bounding-box regression) initializes only 35,875 parameters via Gaussian distributions ($\mathcal{N}(0, 0.01)$ and $\mathcal{N}(0, 0.001)$), allowing stable, rapid convergence on 70 training samples.

---

## 6. Optimizer and Scheduler Source-Code Verification

### 6.1 Resolution of the SGD vs. AdamW Discrepancy
- **Background:** Phase 9.2A originally used SGD ($lr=0.005, momentum=0.9, weight\_decay=0.0005$), following TorchVision's standard reference. However, under batch size 2 on a single laptop GPU, SGD required aggressive GradScaler scale dampening (1024.0) to prevent FP16 overflow and triggered early stopping after only 13 epochs (baseline mAP50 = 0.1394).
- **Modern Transfer Learning Rationale:** For fine-tuning deep convolutional backbones on small target datasets, AdamW ($lr=1.0 \times 10^{-4}, \beta_1=0.9, \beta_2=0.999, weight\_decay=1.0 \times 10^{-4}$) provides per-parameter adaptive learning rates, stabilizing sparse anchor gradient updates.
- **Source Code Verification & Hardening:**
  `scripts/train_faster_rcnn.py` has been updated to dynamically read `cfg["training"]["optimizer"]["type"]`. When set to `"AdamW"`, it instantiates:
  ```python
  optimizer = torch.optim.AdamW(
      [p for p in model.parameters() if p.requires_grad],
      lr=base_lr,
      betas=(float(opt_cfg.get("beta1", 0.9)), float(opt_cfg.get("beta2", 0.999))),
      weight_decay=weight_decay,
  )
  ```

### 6.2 Scheduler Implementation & Mathematics
The learning rate schedule is implemented via `build_warmup_cosine_scheduler()` using PyTorch's `LambdaLR`:

- **Stepping Frequency:** **Per batch** (specifically: inside the batch loop, conditioned on `if optimizer_stepped:` to protect against AMP skipped steps).
- **Dataset Size:** 70 training images.
- **Batch Size:** 2 (`drop_last=False`).
- **Batches per Epoch:** $\lceil 70 / 2 \rceil = \mathbf{35\text{ steps/epoch}}$.
- **Warmup Epochs:** 3 epochs $\to 3 \times 35 = \mathbf{105\text{ warmup steps}}$.
- **Maximum Epochs:** 30 epochs $\to 30 \times 35 = \mathbf{1,050\text{ total steps}}$.
- **Decay Phase:** $1,050 - 105 = \mathbf{945\text{ decay steps}}$.

### 6.3 Learning Rate Milestone Progression
Verified via `scripts/verify_lr_progression.py`:

| Global Step | Epoch | Phase | Mathematical Formula | Effective Learning Rate |
| :---: | :---: | :---: | :--- | :---: |
| **0** | Pre-train | Initial | `base_lr * warmup_factor` ($1.0e-4 \times 0.333333$) | $3.333 \times 10^{-5}$ |
| **1** | Epoch 1 | Warmup | Linear interpolation step 1/105 | $3.397 \times 10^{-5}$ |
| **35** | Epoch 1 End | Warmup | Linear interpolation step 35/105 | $5.556 \times 10^{-5}$ |
| **70** | Epoch 2 End | Warmup | Linear interpolation step 70/105 | $7.778 \times 10^{-5}$ |
| **105** | Epoch 3 End | Warmup End | Linear interpolation step 105/105 (`base_lr`) | $1.000 \times 10^{-4}$ |
| **106** | Epoch 4 Start | Cosine Decay | Step 1 of 945 decay steps | $9.999 \times 10^{-5}$ |
| **577** | Epoch 17 Mid | Cosine Decay | Midpoint of decay ($\text{progress} = 0.5$) | $5.050 \times 10^{-5}$ |
| **1,050** | Epoch 30 End | Final Step | Complete cosine cycle (`min_lr`) | $1.000 \times 10^{-6}$ |

---

## 7. Mixed Precision (AMP) Step-Detection & Skipped-Step Instrumentation

Under mixed-precision training with `torch.cuda.amp.GradScaler`, gradients that contain `inf` or `nan` cause `scaler.step(optimizer)` to skip the parameter update, subsequently downscaling the scale factor:
- **Safety Mechanism:** In `scripts/train_faster_rcnn.py`, the scale factor before stepping is compared with the scale factor after stepping:
  ```python
  scale_before = scaler.get_scale()
  scaler.step(optimizer)
  scaler.update()
  scale_after = scaler.get_scale()
  optimizer_stepped = (scale_after >= scale_before)
  ```
- **Scheduler Protection:** The LR scheduler is advanced if and only if `optimizer_stepped` is `True`. If a step is skipped, the LR does not advance, preventing schedule desynchronization.
- **Skipped-Step Instrumentation:** Counters `epoch_skipped_steps` and `total_skipped_steps` were integrated into `train_faster_rcnn.py` and saved to `epoch_record` to maintain full observability over gradient stability.
- **Verification:** Unit test `scripts/test_amp_step_detection.py` injected synthetic NaNs and verified that the step was skipped, `epoch_skipped_steps` incremented, and the learning rate remained unadvanced.

---

## 8. Tier 2 Ignore-Region Handling Verification

The Tier 2 ignore-region mechanism was engineered to solve the negative-background contamination identified in Phase 9.5.1:
- **RPN Ignore Logic (`IgnoreAwareRPN`):**
  - Standard Faster R-CNN RPN classifies anchors that have $\text{IoU} < 0.30$ with focal boxes as negative background.
  - In images with diffuse symptoms (Curl Virus, Reddening, Variegation), anchors placed on diffuse lesions would receive negative supervision.
  - `IgnoreAwareRPN` computes the Intersection-over-Anchor (IoA) between all candidate anchors and ground-truth `ignore_boxes`.
  - Any anchor with $\text{IoA} \ge 0.30$ is masked out of RPN classification loss, provided it is not already an assigned positive anchor (`~is_positive` invariant).
- **RoI Heads Ignore Logic (`IgnoreAwareRoIHeads`):**
  - RoI proposals generated by the RPN are checked against `ignore_boxes`.
  - Negative proposals that overlap diffuse ignore regions with $\text{IoA} \ge 0.30$ are filtered out before computing RoI cross-entropy loss.
- **Unit Test Verification:**
  - 13/13 unit tests passed in `tests/models/test_ignore_region_faster_rcnn.py`.
  - Tests confirm that focal boxes are never masked, ignore boxes correctly suppress negative penalties, and backward gradients flow cleanly without NaN or zero-proposal crashes.

---

## 9. Dataset Staging and Split Provenance

To eliminate file lookup errors without modifying legacy 50-image directories, an isolated staging directory was constructed using `scripts/prepare_consolidated_100_detection_data.py`:

- **Location:** `data/detection_consolidated_100/`
  - `images/`: 100 images (50 from Tasks 4 & 5, 50 from Task 6).
  - `labels/`: 100 YOLO `.txt` files containing 1,009 normalized focal bounding boxes.
  - `ignore_labels/`: 100 ignore `.txt` files containing 130 normalized diffuse ignore boxes.

### 9.1 Dataset Split Verification

| Split | Image Count | Focal Boxes | Ignore Boxes | Healthy Negative Controls | Split SHA-256 Hash |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | 70 | 683 | 62 | 10 | `4dcbac115a2e717ce10229ff1309ab7f5d3bc8f6fa5515bfa170934bfd9c04a7` |
| **Validation** | 30 | 326 | 68 | 5 | `78b216ea96204ba8ec59b70e3866ba54edbe1a4b33e0c17b967edd0f0061dff3` |
| **Total** | **100** | **1,009** | **130** | **15** | `faef40caf326c6f958eebb1813c37a2d5561947003bb5c3bc78c177a3743e285` |

- **Group Isolation Verified:** $\text{Train} \cap \text{Validation} = \emptyset$ (zero image overlap).
- **Class Balance:**
  - Train: exactly 10 images per class across all 7 classes.
  - Validation: 4 images for Bacterial Blight, Curl Virus, Herbicide Growth Damage, Leaf Hopper Jassids, Leaf Redding; 5 images for Healthy Leaf and Leaf Variegation.

---

## 10. Hardware Dry-Run Integrity & Memory Interpretation

### 10.1 Review of Phase 9.5.3.1 Hardware Dry-Run
- **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (6,143.5 MB total dedicated VRAM).
- **Measured VRAM Metrics (under `torch.cuda.amp.autocast(float16)`):**
  - Baseline VRAM: 0.00 MB
  - Model Loaded VRAM: 169.37 MB
  - Peak Forward Pass Allocated: 1,855.98 MB
  - Peak Backward Pass Allocated: 1,906.91 MB
  - Peak Backward Pass Reserved: 2,210.00 MB
  - Measured Safety Headroom: 4,236.6 MB (69.0% buffer)

### 10.2 Parameter vs. Buffer State Integrity
- **Model Parameters:** In the dry run, all 209 parameter tensors maintained an identical SHA-256 hash (`eccdc3ea...`) before and after the backward pass, confirming that omitting `optimizer.step()` prevented any weight drift.
- **BatchNorm Mutable Buffers:** In PyTorch, running a forward pass in `.train()` mode updates BatchNorm running statistics (`running_mean`, `running_var`, `num_batches_tracked`). In the dry run, this occurred on an **ephemeral in-memory model instance** that was never written to disk and was immediately discarded upon process exit. No checkpoint on disk was touched.

### 10.3 Remaining Uncertainties & Real Training VRAM Estimation
While the dry run confirmed hardware feasibility, claiming "zero OOM risk" is unscientific. The following variables will increase memory consumption during full training:
1. **Lesion Density Variance:** Real images have variable lesion counts (e.g., Frame 34 has 42 lesions). Higher lesion counts produce more anchor-to-ground-truth matches and more RoIAlign extractions.
2. **Optimizer State Buffers:** AdamW maintains two state tensors (`exp_avg` and `exp_avg_sq`) for each of the ~21.3M trainable parameters. In float32, this adds approximately $2 \times 85\text{ MB} \approx 170\text{ MB}$ of allocated VRAM.
3. **Windows WDDM Display Driver Overhead:** Operating system display composition on Windows 11 can cause transient GPU allocation spikes.
4. **Realistic Estimated Peak VRAM:** Peak allocated VRAM during full training is projected at **2.4 GB to 2.8 GB**, reserving ~3.2 GB. This maintains an estimated safety buffer of **>50%**, indicating very low risk of OOM at batch size 2.

---

## 11. Evaluation and Research Reporting Standard

To uphold academic integrity and rigorous research reporting:
1. **Validation Metrics Only:** All detection performance figures produced by this experiment will be explicitly documented and cited as **30-image validation metrics**.
2. **No Test Set Conflation:** These metrics will NOT be described as "test accuracy", "independent test mAP", or "generalization benchmark".
3. **Classification Test Set Preserved:** The sealed 321-image classifier test set (`data/splits/test.csv`, SHA-256 `6025d86f...`) remains untouched with zero data leakage.
4. **Future Work Framing:** The final project dissertation and technical reports will explicitly state that evaluating the detector on a large, separately collected field test set ($N > 500$) is planned future work.

---

## 12. Final Validated Experiment Run Card

The finalized, validated run card is saved at `configs/phase9_5_faster_rcnn_retrain_config.yaml`:

```yaml
experiment:
  name: "phase9_5_faster_rcnn_ignore_retrain"
  run_id: "EXP-FRCNN-20261117-01"
  phase: "Phase 9.5 Retrain with Tier 2 Ignore Supervision"
  reproducibility_seed: 42
  deterministic: true

model:
  architecture: "fasterrcnn_resnet50_fpn_v2"
  torchvision_version: "0.29.2+cu132"
  initialization_provenance: "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1"
  weights_file: "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
  weights_sha256: "DD69338A24B8D7381807E247652BDC356325BCBAF1CD3E092E00E0A1A58706BF"
  num_classes: 7
  background_label: 0
  trainable_backbone_layers: 3
  enable_ignore_regions: true
  rpn_ignore_threshold: 0.30
  roi_ignore_threshold: 0.30

class_mapping:
  0: "background"
  1: "Bacterial Blight"
  2: "Curl Virus"
  3: "Herbicide Growth Damage"
  4: "Leaf Hopper Jassids"
  5: "Leaf Redding"
  6: "Leaf Variegation"

dataset:
  manifest_path: "artifacts/manifests/consolidated_100_annotation_manifest.csv"
  manifest_sha256: "faef40caf326c6f958eebb1813c37a2d5561947003bb5c3bc78c177a3743e285"
  data_dir: "data/detection_consolidated_100"
  train_samples: 70
  val_samples: 30
  train_split_sha256: "4dcbac115a2e717ce10229ff1309ab7f5d3bc8f6fa5515bfa170934bfd9c04a7"
  val_split_sha256: "78b216ea96204ba8ec59b70e3866ba54edbe1a4b33e0c17b967edd0f0061dff3"
  group_isolation_verified: true
  total_focal_boxes: 1009
  total_ignore_boxes: 130
  healthy_control_samples: 15

training:
  batch_size: 2
  num_workers: 0
  max_epochs: 30
  optimizer:
    type: "AdamW"
    lr: 0.0001
    beta1: 0.9
    beta2: 0.999
    weight_decay: 0.0001
  lr_scheduler:
    type: "WarmupCosineAnnealingLR"
    warmup_epochs: 3
    warmup_steps: 105
    total_steps: 1050
    warmup_factor: 0.333333
    base_lr: 0.0001
    min_lr: 0.000001
    stepping: "per_batch"
  mixed_precision:
    enabled: true
    dtype: "float16"
    grad_scaler_init_scale: 1024.0

evaluation:
  eval_batch_size: 1
  reporting_standard: "30-image validation metrics"
  checkpoint_policy:
    primary_metric: "val_map50"
    mode: "max"
    best_checkpoint_path: "artifacts/checkpoints/detection/phase9_5_retrain/best_faster_rcnn_ignore.pt"
    latest_checkpoint_path: "artifacts/checkpoints/detection/phase9_5_retrain/latest_faster_rcnn_ignore.pt"
    early_stopping_patience: 5

preservation_and_rollback:
  baseline_checkpoint_protected: "artifacts/checkpoints/detection/best_faster_rcnn.pt"
  baseline_sha256: "01d82b3282509c543c1bd3a48a46cc03581b3ce573a8d8b77b85be8e2b36b5d2"
  rollback_trigger: "If retrained model val mAP@0.50 regresses or fails stability checks, baseline checkpoint remains production default."
```

---

## 13. Issue Register Summary

A dedicated issue register has been compiled at `reports/phase9_5_3_2_issue_register.md`. All 8 detected configuration and environmental issues have been audited and resolved or bounded:

| Issue ID | Severity | Component | Issue Description | Safe Correction Applied | Verification Result |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `ISSUE-01` | **HIGH** | `scripts/train_faster_rcnn.py` | `torch.optim.SGD` was hardcoded, ignoring AdamW in config | Dynamic optimizer dispatch implemented for AdamW & SGD | Verified via `--dry-run-check` |
| `ISSUE-02` | **CRITICAL** | `data/detection/` | Task 6 images and labels not staged in dataset directory | Created isolated `data/detection_consolidated_100/` | Verified all 100 images/labels load |
| `ISSUE-03` | **HIGH** | `scripts/train_faster_rcnn.py` | Hardcoded assertions `assert len == 35/15` | Parameterized assertions to read from config | Verified 70/30 counts pass |
| `ISSUE-04` | **HIGH** | `scripts/train_faster_rcnn.py` | Missing `enable_ignore_regions` in model builder call | Wired `enable_ignore_regions` and thresholds from config | Verified `IgnoreAware` modules equip |
| `ISSUE-05` | **MEDIUM** | `scripts/train_faster_rcnn.py` | AMP skipped steps untracked in metrics/logs | Added `epoch_skipped_steps` and `total_skipped_steps` counters | Verified in `test_amp_step_detection.py` |
| `ISSUE-06` | **CRITICAL** | `configs/` | Risk of overwriting production baseline `best_faster_rcnn.pt` | Created `phase9_5_faster_rcnn_retrain_config.yaml` with isolated paths | Baseline SHA-256 protected |
| `ISSUE-07` | **LOW** | `reports/` | Confusing comments stating `(0: healthy/background)` | Standardized documentation: Class 0 = strictly Background | Verified in config and metric tests |
| `ISSUE-08` | **HIGH** | `tests/backend/` | Sandbox loopback socket block causes Starlette `TestClient` hang | Diagnosed `WinError 10013`; bounded test boundaries | Unit/model tests pass in sandbox |

---

## 14. Proposed Master Project Task Tracker Corrections

The following corrections are proposed for `reports/MASTER_PROJECT_TASK_TRACKER.md` to reflect recent Phase 9.5 and 9.5.3 audit reconciliations:

### 14.1 Reconcile Frame 20 Historical Severity Record
- **Historical Inconsistency:** Phase 9.3/9.4 documentation listed Frame 20 (`LR00001.jpg`) severity as **30.40%**, derived from an unconstrained bounding box containing 145,920 background pixels ($[0, 0, 800, 800]$ image rectangle).
- **Corrected Audit Value:** Phase 9.5.1 established the precision polygon-restricted leaf boundary (area 344,057 px) and leaf-restricted disease mask (area 148,600 px). The true leaf-restricted severity is:
  $$\text{Severity} = \frac{148,600\text{ px}}{344,057\text{ px}} = \mathbf{43.19\%}$$
- **Proposed Tracker Update:** Reconcile all historical task references for Frame 20 to record 43.19% severity.

### 14.2 Maintain Approval Gate 4 Status as Pending
- **Task Reference:** `TASK-TRAIN-EVALUATE-NEED` / Approval Gate 4.
- **Current Status:** **`PENDING_APPROVAL`**.
- **Audit Requirement:** The status must remain **`PENDING_APPROVAL`** until the user explicitly issues training authorization. This configuration audit does NOT constitute training authorization.

---

## 15. Test Suite Execution Results

To verify detector integrity, 58 specialized test cases were executed across model, architecture, dataset, and validation suites:

| Test Suite File | Tested Architecture / Component | Tests Run | Tests Passed | Tests Failed | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `tests/models/test_faster_rcnn.py` | Standard Faster R-CNN factory, FPN features, output shapes | 23 | 23 | 0 | **`100% PASSED`** |
| `tests/models/test_ignore_region_faster_rcnn.py` | `IgnoreAwareRPN`, `IgnoreAwareRoIHeads`, IoA masking, gradients | 13 | 13 | 0 | **`100% PASSED`** |
| `tests/data/test_detection_dataset.py` | `CottonDetectionDataset`, YOLO parsing, ignore boxes, Healthy controls | 5 | 5 | 0 | **`100% PASSED`** |
| `tests/test_validation_rules_v3.py` | Geometric validation, 15 Stage 1 checks, coordinate clamping | 15 | 15 | 0 | **`100% PASSED`** |
| `scripts/test_metric_mapping.py` | Prediction label shift ($p_{lbl}-1$), ground truth alignment, mAP@0.50 | 1 | 1 | 0 | **`100% PASSED`** |
| `scripts/test_amp_step_detection.py` | AMP `GradScaler` scale tracking, NaN skipped steps, counter accuracy | 1 | 1 | 0 | **`100% PASSED`** |
| **Total Test Suite** | **Comprehensive Detector Configuration & Pipeline Verification** | **58** | **58** | **0** | **`100% PASSED`** |

---

## 16. Conclusion & Final Preflight Recommendation

### Final Recommendation:
# **`A. CONFIGURATION VERIFIED — READY FOR EXPLICIT TRAINING APPROVAL.`**

All technical, dataset, code, optimizer, scheduler, and provenance requirements have been audited and verified without starting model training or modifying existing production assets.

**Safety Assurance Summary:**
- **Zero training epochs or optimizer updates executed.**
- **Zero checkpoints overwritten (baseline SHA-256 `01d82b32...` protected).**
- **Zero CVAT task or annotation modifications.**
- **Zero classification test set leakage (321 images preserved, SHA-256 `6025d86f...`).**
- **Hardware memory headroom verified (>50% buffer on RTX 3050).**
- **Dry-run check passed cleanly.**

**Next Step:** Awaiting explicit user authorization at Approval Gate 4 to execute Faster R-CNN retraining.
