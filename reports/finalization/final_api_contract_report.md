# CottonGuard AI — Final API Contract Validation Report

**Document Reference:** `reports/finalization/final_api_contract_report.md`  
**API Specification:** OpenAPI 3.1.0 / Swagger UI (`http://localhost:8000/docs`)  
**Backend Framework:** FastAPI / Pydantic V2  
**Evaluation Standard:** Final B.Tech REST API Protocol & Contract Compliance Review  
**Date:** November 19, 2026  
**Auditor:** Senior API Architect & QA Specialist  
**Status:** **VERIFIED WORKING — 100% CONTRACT COMPLIANCE**

---

## 1. Executive Summary

A contract validation audit was conducted across all REST API endpoints of the CottonGuard AI backend. The audit verified:
1. Pydantic request/response schema serialization.
2. HTTP status code conformity (200, 201, 400, 401, 404, 422, 503).
3. Image upload boundary validation and MIME-type enforcement.
4. Token authentication and permission handling.
5. Swagger/OpenAPI documentation accuracy.

---

## 2. Endpoint Contract Inventory & Verification

### A. Health & Telemetry Contracts
- **Endpoint:** `GET /api/health`
  - **Contract:** Returns `{ "status": "healthy", "service": "cotton-leaf-ai", "version": "1.0.0", "timestamp": "...", "database": "connected" }`
  - **Status Code:** `200 OK`
  - **Verification:** Verified passing in both live runtime and `test_health.py`.
- **Endpoint:** `GET /api/health/models`
  - **Contract:** Returns live status of all models, checkpoint paths, CUDA availability, and allocated VRAM.
  - **Status Code:** `200 OK`
  - **Verification:** Accurately reflects loaded EfficientNet-B0 and U-Net checkpoints on `cuda:0`.

### B. Authentication & Identity Contracts
- **Endpoint:** `POST /api/auth/token`
  - **Contract:** Form-encoded `username` and `password`. Returns `{ "access_token": "<jwt>", "token_type": "bearer", "role": "..." }`.
  - **Status Codes:** `200 OK` on valid credentials; `401 UNAUTHORIZED` on invalid credentials.
  - **Verification:** Tested with bcrypt hash authentication; rejects non-matching credentials.
- **Endpoint:** `GET /api/auth/me`
  - **Contract:** Requires `Authorization: Bearer <jwt>`. Returns user ID, username, email, and role.
  - **Status Codes:** `200 OK` with valid token; `401 UNAUTHORIZED` if token is expired, tampered, or missing.
  - **Verification:** Tested across role hierarchies.

### C. Inference & Prediction Contracts
- **Endpoint:** `POST /api/predict`
  - **Contract:** Multipart form upload (`file: UploadFile`, optional `seg_alpha: float`). Returns full `PredictionDetailResponse` JSON:
    - `id`: integer
    - `predicted_class`: string (one of 7 valid disease names)
    - `confidence`: float ($0.0 \le c \le 1.0$)
    - `all_probabilities`: dictionary of 7 classes $\rightarrow$ float
    - `leaf_pixels`: integer $\ge 0$
    - `disease_pixels`: integer $\ge 0$
    - `healthy_pixels`: integer $\ge 0$
    - `affected_area_percentage`: float ($0.0 \le p \le 100.0$)
    - `severity_level`: string (`Healthy`, `Low`, `Moderate`, `High`, `Severe`, `Critical`)
    - `overlay_url`: string URI
    - `leaf_mask_url`: string URI
    - `disease_mask_url`: string URI
    - `gradcam_url`: string URI
    - `recommendations`: dictionary (cultural, chemical, biological controls)
    - `uncertainty_notes`: scientific disclaimer string
  - **Status Codes:**
    - `200 OK` on successful inference.
    - `400 BAD REQUEST` if non-image file uploaded (e.g., `.txt`, `.exe`).
    - `413 PAYLOAD TOO LARGE` if file size $> 10\text{ MB}$.
    - `503 SERVICE UNAVAILABLE` if checkpoint file is missing.
  - **Verification:** Validated across all 7 disease reference classes and error edge cases.

### D. History & Reporting Contracts
- **Endpoint:** `GET /api/predictions`
  - **Parameters:** `skip: int = 0`, `limit: int = 20`, `disease_class: Optional[str]`, `severity_level: Optional[str]`.
  - **Contract:** Returns `{ "total": int, "page": int, "page_size": int, "items": List[PredictionSummaryResponse] }`.
  - **Status Code:** `200 OK`.
  - **Verification:** Confirmed pagination and multi-parameter filtering over all 64 database records.
- **Endpoint:** `GET /api/predictions/{id}`
  - **Contract:** Returns single `PredictionDetailResponse`.
  - **Status Codes:** `200 OK` on existing ID; `404 NOT FOUND` on nonexistent ID.
  - **Verification:** Verified ID lookups and 404 error boundaries.
- **Endpoint:** `GET /api/predictions/{id}/report`
  - **Contract:** Returns binary `application/pdf` stream with `Content-Disposition: attachment; filename="report_prediction_{id}.pdf"`.
  - **Status Codes:** `200 OK` with valid PDF binary; `404 NOT FOUND` on nonexistent prediction.
  - **Verification:** Verified PDF generation and headers.

---

## 3. Schema & Validation Robustness

- **Pydantic V2 Models:** Request and response schemas enforce strict typing, bounding ranges, and nullability guards.
- **Automatic Documentation:** Interactive Swagger documentation at `http://localhost:8000/docs` and ReDoc at `http://localhost:8000/redoc` are completely synchronized with Pydantic schemas.

---

## 4. API Contract Verdict

The CottonGuard AI API contract is **VERIFIED WORKING AND FULLY COMPLIANT**. All endpoints exhibit predictable HTTP status codes, robust input sanitization, and clean JSON payload structures.
