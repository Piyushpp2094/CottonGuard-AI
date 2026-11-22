# CottonGuard AI — Uncertain Region Handling & Staged Review Log (Phase 9.5)

**Policy Standard:** ANNOTATION_STANDARD_V3 Section 4  
**Philosophy:** Ambiguous, low-contrast, or unresolvable regions are never forced into training sets.

---

## 1. Summary of Uncertain & Boundary Regions

| Image Filename | Frame | Class | Suspected Feature | Resolution / Action Taken | Status |
|:---|:---:|:---|:---|:---|:---:|
| `BBC00008.jpg` | 35 | Bacterial Blight | Lesion #52 ($\Delta L = 9.7$) | Contrast sub-threshold; deleted to prevent false positive detector bias | `RESOLVED_EXCLUDED` |
| `BBC00008.jpg` | 35 | Bacterial Blight | Lesion #07 | Bisected by right image border; tagged `is_boundary_truncated=True` | `BOUNDARY_TRUNCATED` |
| `LHJ00059.jpg` | 16 | Leaf Hopper Jassids | Central green veins | SAM wedges previously covered healthy lamina; replaced with 5 narrow marginal ribbons | `RESOLVED_REANNOTATED` |
| `LR00001.jpg` | 20 | Leaf Redding | Diffuse anthocyanin blush | Isolated as Tier 2 diffuse; excluded from Faster R-CNN detector boxes | `TIER2_DIFFUSE_SEG_ONLY` |
| `LR00003.jpg` | 43 | Leaf Redding | Diffuse blush | Macro view; diffuse blush segmented for U-Net, 35 dark focal necrotic spots boxed | `TIER2_DIFFUSE_SEG_ONLY` |
| `CV00431.jpg` | 09 | Curl Virus | Wrinkled petiole fold | Extreme whole-leaf distortion; annotated as morphological segmentation without artificial boxes | `MORPHOLOGICAL_SEG_ONLY` |
| `HGD00210.jpg` | 13 | Herbicide Growth Damage | Strapped narrow veinlets | Severe epinastic stunting; annotated as morphological deformation | `MORPHOLOGICAL_SEG_ONLY` |

---

## 2. Unresolved Ambiguities

- **Total Unresolved / Quarantined Images in Task #6:** **0**
- All 50 images in Task #6 possess verified clear diagnostic manifestations adhering to their class criteria.
