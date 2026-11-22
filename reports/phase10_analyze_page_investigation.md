# CottonGuard AI — Phase 10: Analyze Page HTTP 500 Root-Cause Investigation & Resolution Report

**Execution Date:** 2026-11-17  
**Project:** CottonGuard AI (`D:\Main_Project\cotton_leaf_ai`)  
**Status:** RESOLVED & VERIFIED (HTTP 200 OK)  
**Strict Safety Constraints Enforced:**
- Zero production database records modified or deleted.
- Zero model checkpoints modified or overwritten.
- Zero CVAT annotations or tasks altered.
- Zero training epochs or `optimizer.step()` executed (Approval Gate 4 strictly preserved).

---

## 1. Executive Summary

During Phase 10 frontend verification of the CottonGuard AI application, navigating to the **Analyze Leaf** page (`http://localhost:5173/analyze`) and selecting `images (3).jpg` resulted in a red error banner:
```
Error: Internal server error.
```
An exhaustive read-only inspection of the active Uvicorn daemon logs (`task-9210.log`) identified an ASGI unhandled exception:
`fastapi.exceptions.ResponseValidationError: 80 validation errors` originating from `POST /api/predictions/analyze`.

The root cause was an internal schema key mismatch: the Faster R-CNN detector output dictionary contained keys `box` and `label`, whereas FastAPI's `PredictionDetailResponse` (specifically the `DetectionBox` sub-model) strictly required `bbox`, `normalized_bbox`, and `class_id`. Because `images (3).jpg` produced 26 positive symptom detections, 78 required field errors were raised, converting the response into an unhandled HTTP 500 error.

A minimal, robust, three-tiered correction was applied:
1. **Pydantic Schema Compatibility Validator:** Added bidirectional mapping (`box` $\leftrightarrow$ `bbox`, `label` $\leftrightarrow$ `class_id`) and default normalized coordinates to `DetectionBox` in [schemas.py](file:///D:/Main_Project/cotton_leaf_ai/backend/app/schemas.py).
2. **Detector Output Alignment:** Updated `_run_detection` in [prediction_service.py](file:///D:/Main_Project/cotton_leaf_ai/backend/app/services/prediction_service.py) to compute both canonical (`bbox`, `normalized_bbox`, `class_id`) and legacy (`box`, `label`) fields.
3. **Database Serialization Defense:** Updated `serialize_prediction_detail` in [prediction_service.py](file:///D:/Main_Project/cotton_leaf_ai/backend/app/services/prediction_service.py) so any past database record deserializes with complete canonical attributes.

Following restart of the backend service, tests using the exact failing image confirmed HTTP 200 OK with all 26 detections rendered successfully.

---

## 2. Root Cause Analysis

### 2.1 Traceback Capture from Uvicorn Log (`task-9210.log`)
```
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "uvicorn/protocols/http/httptools_impl.py", line 401, in run_asgi
  File "uvicorn/middleware/proxy_headers.py", line 60, in __call__
  File "fastapi/applications.py", line 1054, in __call__
  ...
  File "fastapi/routing.py", line 324, in serialize_response
    raise ResponseValidationError(
fastapi.exceptions.ResponseValidationError: 80 validation errors:
  {'type': 'missing', 'loc': ('response', 'PredictionDetailResponse', 'detections', 0, 'bbox'), 'msg': 'Field required', 'input': {'box': [181.5, 70.7, 192.6, 83.2], 'label': 1, ...}}
  {'type': 'missing', 'loc': ('response', 'PredictionDetailResponse', 'detections', 0, 'normalized_bbox'), 'msg': 'Field required', ...}
  {'type': 'missing', 'loc': ('response', 'PredictionDetailResponse', 'detections', 0, 'class_id'), 'msg': 'Field required', ...}
  ... (repeated for 26 detections = 78 field errors) ...
  {'type': 'missing', 'loc': ('response', 'PredictionJobCreateResponse', 'job_id'), 'msg': 'Field required', ...}
```

### 2.2 Mechanism of Failure
1. **Endpoint Signature:**
   `POST /api/predictions/analyze` has `response_model=Union[PredictionDetailResponse, PredictionJobCreateResponse]`.
2. **Schema Invariant:**
   `PredictionDetailResponse.detections` is defined as `List[DetectionBox]`.
   `DetectionBox` required:
   - `bbox: List[float]`
   - `normalized_bbox: List[float]`
   - `class_id: int`
   - `class_name: str`
   - `confidence: float`
3. **Previous Inference Service Output:**
   In [prediction_service.py](file:///D:/Main_Project/cotton_leaf_ai/backend/app/services/prediction_service.py#L380-L386):
   ```python
   detections.append({
       "box": [round(float(b), 1) for b in box],
       "label": lbl_int,
       "class_name": cls_name,
       "confidence": round(float(sc), 4),
   })
   ```
4. **Why Prior Synthetic Tests Passed:**
   Unit tests used synthetic solid green images where detection count was 0 (`detections: []`). An empty list trivially passed `List[DetectionBox]` validation. Only when a real diseased leaf image with detections was uploaded did FastAPI instantiate `DetectionBox` objects and fail.

---

## 3. Implementation of the Correction

### 3.1 Pydantic Schema Enhancement (`backend/app/schemas.py`)
Added a `@model_validator(mode="before")` to [DetectionBox](file:///D:/Main_Project/cotton_leaf_ai/backend/app/schemas.py#L81-L105):
```python
class DetectionBox(BaseModel):
    bbox: List[float]
    normalized_bbox: List[float] = Field(default_factory=list)
    class_id: int
    class_name: str
    confidence: float
    box: Optional[List[float]] = None
    label: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def fill_compat_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "bbox" not in data and "box" in data:
                data["bbox"] = data["box"]
            elif "box" not in data and "bbox" in data:
                data["box"] = data["bbox"]
            if "class_id" not in data and "label" in data:
                data["class_id"] = data["label"]
            elif "label" not in data and "class_id" in data:
                data["label"] = data["class_id"]
            if "normalized_bbox" not in data or data["normalized_bbox"] is None:
                data["normalized_bbox"] = []
        return data
```

### 3.2 Detector Output Formatting (`backend/app/services/prediction_service.py`)
In `_run_detection`:
```python
orig_w, orig_h = rgb_image.size
detections: List[Dict[str, Any]] = []
for box, lbl, sc in zip(filtered_boxes, filtered_labels, filtered_scores):
    lbl_int = int(lbl)
    cls_name = FOREGROUND_CLASSES[lbl_int - 1] if 1 <= lbl_int <= len(FOREGROUND_CLASSES) else "Unknown"
    x1 = float(np.clip(box[0], 0.0, orig_w))
    y1 = float(np.clip(box[1], 0.0, orig_h))
    x2 = float(np.clip(box[2], 0.0, orig_w))
    y2 = float(np.clip(box[3], 0.0, orig_h))
    cx_norm = round(((x1 + x2) / 2.0) / max(orig_w, 1), 4)
    cy_norm = round(((y1 + y2) / 2.0) / max(orig_h, 1), 4)
    w_norm = round((x2 - x1) / max(orig_w, 1), 4)
    h_norm = round((y2 - y1) / max(orig_h, 1), 4)
    detections.append({
        "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "normalized_bbox": [cx_norm, cy_norm, w_norm, h_norm],
        "class_id": lbl_int,
        "class_name": cls_name,
        "confidence": round(float(sc), 4),
        "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        "label": lbl_int,
    })
```

### 3.3 Historical Database Record Sanitization
In `serialize_prediction_detail`, raw entries from `DetectionResult.detections_json` are safely parsed and enriched with canonical coordinates so queries to `/api/predictions/{id}` or history views never raise validation errors.

---

## 4. Verification Evidence

### 4.1 Focused Regression Test Suite
File: [tests/backend/test_detection_serialization_regression.py](file:///D:/Main_Project/cotton_leaf_ai/tests/backend/test_detection_serialization_regression.py)
```
============================= test session starts =============================
collected 3 items

tests/backend/test_detection_serialization_regression.py::test_detection_box_canonical_schema PASSED [ 33%]
tests/backend/test_detection_serialization_regression.py::test_detection_box_legacy_schema_compatibility PASSED [ 66%]
tests/backend/test_detection_serialization_regression.py::test_serialize_prediction_detail_normalizes_legacy_detections PASSED [100%]

======================== 3 passed, 1 warning in 16.75s ========================
```

### 4.2 Live API Execution with Exact Upload Image
Testing against `http://127.0.0.1:8000/api/predictions/analyze` with the exact image `20261116_215424_4ecc8cc56a.jpg`:
- **HTTP Status:** 200 OK
- **Prediction ID:** 42
- **Predicted Foliar Class:** Leaf Redding (Confidence: 65.7%)
- **Detection Status:** EXPERIMENTAL
- **Detection Count:** 26
- **First Detection Object:**
  ```json
  {
    "bbox": [181.5, 70.7, 192.6, 83.2],
    "normalized_bbox": [0.6852, 0.3079, 0.0407, 0.0502],
    "class_id": 1,
    "class_name": "Bacterial Blight",
    "confidence": 0.9616,
    "box": [181.5, 70.7, 192.6, 83.2],
    "label": 1
  }
  ```
- **GET /api/predictions/42:** Status 200 OK, 26 detections loaded.
- **Vite Proxy to Detection Artifacts:** `http://127.0.0.1:5173/artifacts/predictions/20261116_215424_4ecc8cc56a_frcnn_det.jpg` returns HTTP 200 OK.

---

## 5. Architectural & System Integrity Invariants

| Guardrail / Invariant | Status | Verification Method |
|---|---|---|
| Model weights unmodified | STRICTLY PRESERVED | SHA-256 hashes unchanged |
| Approval Gate 4 preserved | STRICTLY PRESERVED | Zero training epochs or optimizer steps |
| Production DB records | STRICTLY PRESERVED | No tables dropped, no existing predictions deleted |
| CVAT annotations / Docker | STRICTLY PRESERVED | Port 8080 unaffected |
| Authentication & RBAC | STRICTLY PRESERVED | Ownership check (403 for unauthorized users) verified |
