# Cotton Leaf AI — Model Documentation & Evaluation Report

---

## 1. Classification Model Card

- **Architecture:** EfficientNet-B0 (Convolutional Neural Network)
- **Pretrained Weights:** ImageNet (Transfer Learning)
- **Input Resolution:** $224 \times 224 \times 3$ (RGB)
- **Output Layer:** Linear Head with 7 logits and Softmax probabilities
- **Regularization:** Head Dropout ($p=0.30$), Weight Decay ($10^{-4}$), Label Smoothing ($0.05$)
- **Optimization:** AdamW ($\text{LR} = 3 \times 10^{-4}$) with CosineAnnealingLR scheduling
- **Checkpoint Location:** [artifacts/checkpoints/classification/best_model.pt](file:///d:/Main_Project/cotton_leaf_ai/artifacts/checkpoints/classification/best_model.pt)

---

## 2. Training Dynamics

- **Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (50W TGP profile)
- **Framework:** PyTorch 2.15.0+cu132 with FP16 Mixed Precision (`torch.amp.autocast`)
- **Total Epochs Trained:** 13 (Early stopping triggered after 5 epochs of plateau)
- **Average Epoch Duration:** $\sim 26$ seconds
- **Best Validation Macro F1:** $0.9839$ (Epoch 8)

---

## 3. Objective Evaluation on Untouched Test Set (321 Images)

The test evaluation was executed strictly against `artifacts/manifests/splits/test.csv`, which was never used during training, tuning, or checkpoint selection:

### Overall Metrics
- **Overall Accuracy:** **97.82%**
- **Macro F1-Score:** **0.9754**
- **Weighted F1-Score:** **0.9782**
- **Macro Precision:** **0.9780**
- **Macro Recall:** **0.9733**

### Per-Class Performance
| Class Name | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| **Bacterial Blight** | 0.9487 | 0.9737 | 0.9610 | 38 |
| **Curl Virus** | 1.0000 | 0.9688 | 0.9841 | 64 |
| **Healthy Leaf** | 0.9500 | 1.0000 | 0.9744 | 38 |
| **Herbicide Growth Damage** | 1.0000 | 1.0000 | 1.0000 | 42 |
| **Leaf Hopper Jassids** | 0.9697 | 0.9412 | 0.9552 | 34 |
| **Leaf Redding** | 0.9775 | 0.9886 | 0.9831 | 88 |
| **Leaf Variegation** | 1.0000 | 0.9412 | 0.9697 | 17 |
| **Macro Average** | **0.9780** | **0.9733** | **0.9754** | **321** |
| **Weighted Average** | **0.9786** | **0.9782** | **0.9782** | **321** |

### Confusion Matrix (Test Split)
```text
Ground Truth \ Predicted    Bact  Curl  Heal  Herb  Jass  Redd  Vari
Bacterial Blight              37     0     1     0     0     0     0
Curl Virus                     0    62     1     0     0     1     0
Healthy Leaf                   0     0    38     0     0     0     0
Herbicide Growth Damage        0     0     0    42     0     0     0
Leaf Hopper Jassids            2     0     0     0    32     0     0
Leaf Redding                   0     0     0     0     1    87     0
Leaf Variegation               0     0     0     0     0     1    16
```

---

## 4. Academic Integrity & Honest Research Claims

1. **Test Set Separation:** Test accuracy is reported exclusively on the untouched hold-out test set (321 images), never on validation or training samples.
2. **Segmentation Ground Truth:** The source SAR-CLD-2024 dataset contains image-level classification labels only. No ground-truth pixel masks exist in the raw dataset.
3. **Zero Fabrication Policy:** U-Net training and affected area estimation are safely marked as *Unavailable / Requires Ground-Truth Masks* rather than inventing synthetic masks or reporting fabricated segmentation accuracy.
