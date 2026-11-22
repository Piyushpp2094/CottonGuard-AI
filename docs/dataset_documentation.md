# Cotton Leaf Dataset Documentation

**Dataset Reference:** `SAR-CLD-2024: A Comprehensive Dataset for Cotton Leaf Disease Detection`  
**Dataset Location:** `D:\Main_Project\dataset\SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection\Original Dataset`

---

## 1. Class Distribution & Inventory

The original dataset was audited non-destructively to ensure zero data corruption, unreadable images, or unintentional modifications:

| Class Index | Class Name | Filename Prefix | Total Images | Distribution % |
|:---:|---|:---:|---:|---:|
| `0` | Bacterial Blight | `BBC` | 250 | 11.73% |
| `1` | Curl Virus | `CV` | 431 | 20.22% |
| `2` | Healthy Leaf | `HL` | 252 | 11.82% |
| `3` | Herbicide Growth Damage | `HGD` | 280 | 13.13% |
| `4` | Leaf Hopper Jassids | `LHJ` | 225 | 10.55% |
| `5` | Leaf Redding | `LR` | 578 | 27.11% |
| `6` | Leaf Variegation | `LV` | 116 | 5.44% |
| **Total** | **7 Classes** | — | **2,132** | **100.00%** |

---

## 2. Duplicate Detection & Group-Aware Isolation

- **Perceptual Hash Audit:** 7 candidate near-duplicate pairs were discovered by dHash analysis.
- **Human Verification:**
  - 6 pairs were confirmed as identical photographic capture sessions taken from multiple angles or slight zoom variations.
  - 1 pair was confirmed as distinct leaves.
- **Leakage Prevention:** All 6 verified same-photo pairs were assigned unified group IDs. Splitting was performed at the **group level** rather than the image level, guaranteeing that related images never appear across different splits.

---

## 3. Verified Split Distribution

Splits are strictly stratified across all seven classes and recorded in `artifacts/manifests/splits/`:

| Split | Images | Groups | Percentage | Purpose |
|---|---:|---:|---:|---|
| **Training** | 1,492 | 1,488 | ~70.0% | Model parameter optimization and feature learning |
| **Validation** | 319 | 319 | ~15.0% | Hyperparameter tuning and early stopping guidance |
| **Testing** | 321 | 319 | ~15.0% | **Untouched hold-out test set** for final objective scoring |
| **Total** | **2,132** | **2,126** | **100.0%** | Zero cross-split group leakage |

---

## 4. Class-by-Split Breakdown

| Class Name | Train Samples | Validation Samples | Test Samples | Total |
|---|---:|---:|---:|---:|
| Bacterial Blight | 175 | 37 | 38 | 250 |
| Curl Virus | 302 | 65 | 64 | 431 |
| Healthy Leaf | 177 | 37 | 38 | 252 |
| Herbicide Growth Damage | 196 | 42 | 42 | 280 |
| Leaf Hopper Jassids | 157 | 34 | 34 | 225 |
| Leaf Redding | 404 | 86 | 88 | 578 |
| Leaf Variegation | 81 | 18 | 17 | 116 |
| **Total** | **1,492** | **319** | **321** | **2,132** |
