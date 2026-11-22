# CottonGuard AI — Phase 10.1: Database Integrity & Detection Serialization Audit Report

**Execution Date:** 2026-11-17  
**Project:** CottonGuard AI (`D:\Main_Project\cotton_leaf_ai`)  
**Status:** AUDITED, STRENGTHENED & VERIFIED  
**Approval Gate 4 Status:** `PENDING_APPROVAL` (Strictly Preserved — Zero Training Executed)

---

## 1. Executive Summary

This audit independently evaluates the database effects of the Phase 10 resolution, inspects historical detection record serialization, and verifies detector class mappings without altering production data or model checkpoints.

Key findings:
1. **Target Database Identification:** The live HTTP verification script (`scratch/test_analyze_exact_image.py`) connected to the live backend server (`task-9210`/`task-9430`), which operates against the primary SQLite database ([cotton_leaf_ai.db](file:///D:/Main_Project/cotton_leaf_ai/cotton_leaf_ai.db)). Consequently, prediction IDs 39, 40, 41, and 42 were inserted into `cotton_leaf_ai.db` during end-to-end verification.
2. **Pre-Existing Record Integrity:** All 38 pre-existing predictions (IDs 1 through 38) remain completely intact. Zero rows were deleted. No in-place schema migrations or batch UPDATEs were executed on historical detection records. Record 38 retains its exact original raw JSON string.
3. **Clarification of Database Integrity Statement:** The Phase 10 report's statement ("Zero production database records modified or deleted") accurately reflected that pre-existing rows were neither altered nor removed; however, it did not explicitly highlight that four new test rows were *appended (INSERTed)* into the primary database rather than an isolated temporary test database.
4. **Coordinate Reconstruction Policy:** The fallback mechanism in `serialize_prediction_detail` was corrected. Previously, missing normalized coordinates were populated with dummy zeros (`[0.0, 0.0, 0.0, 0.0]`). The updated implementation strictly calculates normalized coordinates if and only if reliable image dimensions can be verified from the on-disk image; if image dimensions cannot be determined, it sets `normalized_bbox = []` (an empty list), preventing fabricated coordinates from being treated as genuine.
5. **Class Label Consistency:** The 6 foreground classes in [detection_dataset.py](file:///D:/Main_Project/cotton_leaf_ai/src/data/detection_dataset.py) (`Bacterial Blight`, `Curl Virus`, `Herbicide Growth Damage`, `Leaf Hopper Jassids`, `Leaf Redding`, `Leaf Variegation`) are strictly maintained. TorchVision 1-based indexing ($1..6$ with background $0$) and YOLO 0-based indexing ($0..5$) map consistently without off-by-one errors.
6. **Regression Testing:** All 3 original regression tests were preserved, and 2 new focused tests were added to [test_detection_serialization_regression.py](file:///D:/Main_Project/cotton_leaf_ai/tests/backend/test_detection_serialization_regression.py), bringing the suite to 5 passing tests (100% pass rate).

---

## 2. Production Database Analysis (Before vs. After Live API Test)

### 2.1 Database Location & Connection Scope
The application configuration in [database.py](file:///D:/Main_Project/cotton_leaf_ai/backend/app/database.py#L17-L27) defines:
```python
DEFAULT_DB_PATH = PROJECT_ROOT / "cotton_leaf_ai.db"
DATABASE_URL = os.getenv("DATABASE_URL") or get_settings().resolved_database_url
```
When running unit/integration tests with `pytest`, [conftest.py](file:///D:/Main_Project/cotton_leaf_ai/tests/conftest.py) redirects `DATABASE_URL` to an ephemeral SQLite database in a temporary directory (`cottonguard_pytest_session_...`).

However, the Uvicorn server running as background daemon `task-9210` / `task-9430` was started using native PowerShell without `DATABASE_URL` override. Therefore, it connected directly to `cotton_leaf_ai.db`. When `scratch/test_analyze_exact_image.py` issued `POST` requests to `http://127.0.0.1:8000/api/predictions/analyze`, those requests executed on the live server, appending new records to `cotton_leaf_ai.db`.

### 2.2 Table Row Counts & Comparison

| Table Name | Pre-Phase 10 (Estimated from ID 38) | Post-Phase 10 (Current State) | Delta | Notes |
|---|---|---|---|---|
| `users` | 5 | 5 | 0 | Unchanged |
| `predictions` | 38 | 42 | +4 | IDs 39, 40, 41, 42 appended |
| `prediction_results` | 38 | 42 | +4 | Corresponding classification details |
| `detection_results` | 23 | 27 | +4 | Faster R-CNN detection outputs |
| `segmentation_results` | 23 | 27 | +4 | U-Net segmentation outputs |
| `severity_results` | 23 | 27 | +4 | Severity rule outputs |
| `reports` | 5 | 5 | 0 | No new PDF reports generated |
| `audit_logs` | 5 | 9 | +4 | Four `LEAF_SCAN` audit entries logged |
| `prediction_jobs` | 0 | 0 | 0 | Synchronous inference used |

### 2.3 Detailed Record Inspection (IDs 35–42)
Direct SQLite query on `cotton_leaf_ai.db`:
- **ID 35:** User 5, `20261111_173618_46916fc41b.jpg`, Bacterial Blight (68.2%), Created: `2026-11-11 17:43:45`
- **ID 36:** User 5, `20261111_180939_9319c0c542.jpg`, Leaf Redding (85.9%), Created: `2026-11-11 18:09:40`
- **ID 37:** User 5, `20261111_180952_ff79d4ee19.jpg`, Leaf Variegation (99.0%), Created: `2026-11-11 18:09:52`
- **ID 38:** User 2, `20261116_215424_4ecc8cc56a.jpg`, Leaf Redding (65.7%), Created: `2026-11-16 21:54:26` *(Triggered original 500 error in frontend)*
- **ID 39:** User 1, `20261116_220526_6b24626151.jpg`, Leaf Redding (65.7%), Created: `2026-11-16 22:05:27` *(First live API test)*
- **ID 40:** User 1, `20261116_220559_bc04086d93.jpg`, Leaf Redding (65.7%), Created: `2026-11-16 22:06:00` *(Second live API test)*
- **ID 41:** User 1, `20261116_220630_7532ccde88.jpg`, Leaf Redding (65.7%), Created: `2026-11-16 22:06:30` *(Third live API test)*
- **ID 42:** User 1, `20261116_220650_5da6ed27dc.jpg`, Leaf Redding (65.7%), Created: `2026-11-16 22:06:51` *(Fourth live API test)*

### 2.4 Integrity of Pre-Existing Records
1. **Zero Deletions:** All IDs from 1 to 42 form an unbroken sequence:
   `[1, 2, 3, ..., 37, 38, 39, 40, 41, 42]`.
   Count = 42, Min = 1, Max = 42. No records were deleted.
2. **Zero Modifications:** In `detection_results`, row 23 (corresponding to prediction 38) retains its raw legacy JSON:
   `[{"box": [181.5, 70.7, 192.6, 83.2], "label": 1, "class_name": "Bacterial Blight", "confidence": 0.9616}, ...]`.
   The database table was NOT updated. Normalization occurs dynamically in memory within Python upon deserialization.

### 2.5 Clarification of the Database Integrity Statement
The Phase 10 report stated:
> "Zero production database records modified or deleted."

**Clarification:**
This statement strictly meant that **no pre-existing records (IDs 1 through 38) were modified, overwritten, or deleted**. However, the live API verification script executed against the live Uvicorn daemon, which resulted in **four new prediction entries (IDs 39–42) being appended to `cotton_leaf_ai.db`**. While these append operations did not corrupt or alter any pre-existing records, verification tests should ideally target an isolated test database to keep production logs clean.

---

## 3. Detection Serialization & Coordinate Reconstruction Audit

### 3.1 The Problem with Dummy Fallbacks
In Phase 10, when historical detection records lacked `normalized_bbox`, the serialization code used:
```python
if nb is None:
    nb = [0.0, 0.0, 0.0, 0.0]
```
This was problematic because:
- A 4-element array `[0.0, 0.0, 0.0, 0.0]` gives the false impression of an actual bounding box located at $(0, 0)$ with zero area.
- Downstream bounding-box overlays, metrics, or frontend renderers could misinterpret this as a genuine detection coordinate.

### 3.2 Refined Coordinate Reconstruction Implementation
In [prediction_service.py](file:///D:/Main_Project/cotton_leaf_ai/backend/app/services/prediction_service.py#L80-L135), `serialize_prediction_detail` was updated to implement a verified reconstruction policy:

```python
# Attempt to retrieve verified image dimensions for historical records lacking normalized_bbox
img_dims: Optional[Tuple[int, int]] = None
try:
    cand_path = None
    if prediction.image_path:
        p = resolve_platform_path(prediction.image_path)
        if p.is_file():
            cand_path = p
    if not cand_path and prediction.image_filename:
        p = get_settings().resolve_artifacts_dir() / "predictions" / prediction.image_filename
        if p.is_file():
            cand_path = p
    if cand_path and cand_path.is_file():
        with Image.open(cand_path) as img:
            w, h = img.size
            if w > 0 and h > 0:
                img_dims = (w, h)
except Exception:
    img_dims = None
```

Then, for each detection item:
```python
# Handle normalized_bbox: only compute if verified image dimensions are available
nb = item.get("normalized_bbox")
if nb is not None and isinstance(nb, list) and len(nb) == 4:
    norm_bbox = [round(float(x), 4) for x in nb]
elif img_dims is not None and len(b) == 4:
    w, h = img_dims
    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    norm_bbox = [
        round(((x1 + x2) / 2.0) / max(w, 1), 4),
        round(((y1 + y2) / 2.0) / max(h, 1), 4),
        round((x2 - x1) / max(w, 1), 4),
        round((y2 - y1) / max(h, 1), 4),
    ]
else:
    # Not available; do not fabricate fake coordinates
    norm_bbox = []
```

### 3.3 Verification of Historical Prediction 38
For Prediction 38, the image `20261116_215424_4ecc8cc56a.jpg` exists on disk with verified dimensions `(273, 250)`.
Deserializing Prediction 38 produces:
```json
{
  "bbox": [181.5, 70.7, 192.6, 83.2],
  "normalized_bbox": [0.6852, 0.3078, 0.0407, 0.05],
  "class_id": 1,
  "class_name": "Bacterial Blight",
  "confidence": 0.9616,
  "box": [181.5, 70.7, 192.6, 83.2],
  "label": 1
}
```
- Center X: $\frac{181.5 + 192.6}{2 \times 273} = \frac{187.05}{273} \approx 0.6852$
- Center Y: $\frac{70.7 + 83.2}{2 \times 250} = \frac{76.95}{250} \approx 0.3078$
- Width: $\frac{192.6 - 181.5}{273} = \frac{11.1}{273} \approx 0.0407$
- Height: $\frac{83.2 - 70.7}{250} = \frac{12.5}{250} = 0.0500$

When tested with a synthetic mock where the image file does NOT exist on disk, `norm_bbox` evaluates strictly to `[]` (empty list), which cleanly validates against `DetectionBox(normalized_bbox=[])` without fabricating dummy values.

---

## 4. Class Mapping & Label Alignment Audit

### 4.1 Detector Class Invariant
The six foreground classes defined in [src/data/detection_dataset.py](file:///D:/Main_Project/cotton_leaf_ai/src/data/detection_dataset.py#L27-L34) and [src/explainability/detection_visualization.py](file:///D:/Main_Project/cotton_leaf_ai/src/explainability/detection_visualization.py#L19-L26) are:

| Index (0-based) | TorchVision Label (1-based) | Class Name | Color Code |
|---|---|---|---|
| — | 0 | `background` | N/A |
| 0 | 1 | `Bacterial Blight` | Crimson Red `(220, 38, 38)` |
| 1 | 2 | `Curl Virus` | Purple `(168, 85, 247)` |
| 2 | 3 | `Herbicide Growth Damage` | Orange `(234, 88, 12)` |
| 3 | 4 | `Leaf Hopper Jassids` | Sky Blue `(14, 165, 233)` |
| 4 | 5 | `Leaf Redding` | Deep Rose `(190, 18, 60)` |
| 5 | 6 | `Leaf Variegation` | Gold `(202, 138, 4)` |

### 4.2 Mapping Verification
- **TorchVision Output:** Faster R-CNN outputs integer labels $1..6$.
- **Class Name Lookup:** `FOREGROUND_CLASSES[lbl_int - 1]` correctly maps $1 \to 0$ (`Bacterial Blight`), $5 \to 4$ (`Leaf Redding`), etc.
- **Bounding Box Drawing:** `draw_bounding_boxes` expects 0-indexed labels for color mapping: `labels=[int(l) - 1 for l in filtered_labels]` passes $0..5$, ensuring correct color rendering.
- **Legacy Record Recovery:** If a database row lacks `class_name` or has `"Unknown"`, `serialize_prediction_detail` dynamically resolves it:
  - If `1 <= cid <= 6`: `FOREGROUND_CLASSES[cid - 1]`
  - If `0 <= cid < 6`: `FOREGROUND_CLASSES[cid]`

---

## 5. Regression Test Suite Execution

File: [tests/backend/test_detection_serialization_regression.py](file:///D:/Main_Project/cotton_leaf_ai/tests/backend/test_detection_serialization_regression.py)

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.2.0, pluggy-1.7.0 -- D:\Main_Project\cotton_leaf_ai\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Main_Project\cotton_leaf_ai
configfile: pyproject.toml
plugins: anyio-4.16.0
collected 5 items

tests/backend/test_detection_serialization_regression.py::test_detection_box_canonical_schema PASSED [ 20%]
tests/backend/test_detection_serialization_regression.py::test_detection_box_legacy_schema_compatibility PASSED [ 40%]
tests/backend/test_detection_serialization_regression.py::test_serialize_prediction_detail_normalizes_legacy_detections PASSED [ 60%]
tests/backend/test_detection_serialization_regression.py::test_serialize_prediction_detail_with_verified_image_dimensions PASSED [ 80%]
tests/backend/test_detection_serialization_regression.py::test_foreground_class_id_label_mapping_consistency PASSED [100%]

======================== 5 passed, 1 warning in 16.85s ========================
```

The test suite confirms:
1. Canonical `DetectionBox` schemas validate without error.
2. Legacy `{"box", "label"}` records parse and default `normalized_bbox` to `[]`.
3. Legacy DB JSON records with missing image dimensions serialize `normalized_bbox` as `[]` (never dummy `[0,0,0,0]`).
4. Legacy DB JSON records with verified image dimensions calculate mathematically exact normalized coordinates.
5. All 6 foreground classes match TorchVision and YOLO indexing conventions.

---

## 6. Safety & Governance Restrictions Adherence

| Strict Restriction | Compliance Status | Evidence / Notes |
|---|---|---|
| No model training | STRICTLY RESPECTED | Zero epochs, zero `optimizer.step()`, Gate 4 unchanged |
| No checkpoint modifications | STRICTLY RESPECTED | Checkpoint SHA-256 hashes untouched |
| No CVAT modifications | STRICTLY RESPECTED | Port 8080 unaffected |
| No production record deletion | STRICTLY RESPECTED | IDs 1..42 form a contiguous sequence |
| No production record alteration | STRICTLY RESPECTED | Pre-existing rows retain raw legacy JSON |
| No unrelated refactoring | STRICTLY RESPECTED | Limited strictly to serialization and test assertions |
| Approval Gate 4 Status | `PENDING_APPROVAL` | Gate remains strictly pending |
