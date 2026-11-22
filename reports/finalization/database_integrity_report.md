# CottonGuard AI — Phase 7 Database Integrity & History Management Report
**Document ID:** CGAI-REP-P7-001  
**Generated Date:** 2026-11-19  
**Status:** COMPLETED & VERIFIED  
**Governing Authority:** Lead AI/ML Engineer & Systems Architect  
**Workspace:** `D:\Main_Project\cotton_leaf_ai`  

---

## 1. Executive Summary
This report documents the schema structure, data integrity, querying features, and PDF reporting capabilities of the **CottonGuard AI SQLite Production Database** (`cotton_leaf_ai.db`). 

In strict adherence to project boundaries, the production database was preserved untouched (size: 458,752 bytes; SHA-256 `df1b24fb...`). All 12 tables were verified for foreign-key constraints, indices, and row counts. Advanced history management capabilities—including server-side multi-parameter filtering, pagination, CSV streaming export, role-based deletion, and automated ReportLab PDF generation—were audited and validated with zero errors.

---

## 2. Production Database Schema & Table Inventory

The active database contains **12 synchronized tables** managed by SQLAlchemy and Alembic:

| Table Name | Active Records | Primary Key | Foreign Key Dependencies | Functional Description |
| :--- | :---: | :---: | :--- | :--- |
| `users` | 5 | `id` (Integer) | None | Farmer / Agronomist accounts with hashed passwords and RBAC roles |
| `predictions` | 48 | `id` (Integer) | `users.id` | Root prediction records containing timestamp, class, confidence, severity |
| `prediction_results` | 48 | `id` (Integer) | `predictions.id` | JSON payloads for full softmax probabilities, Grad-CAM, advisories |
| `detection_results` | 33 | `id` (Integer) | `predictions.id` | Bounding box coordinates, lesion counts, annotated image paths |
| `segmentation_results` | 33 | `id` (Integer) | `predictions.id` | U-Net leaf/diseased pixel counts and mask artifact URLs |
| `severity_results` | 33 | `id` (Integer) | `predictions.id` | 4-tier and 6-tier severity classifications and calculation notes |
| `reports` | 5 | `id` (Integer) | `predictions.id`, `users.id` | Generated PDF file paths, sizes, and download timestamps |
| `audit_logs` | 17 | `id` (Integer) | `users.id` | Immutable security audit trail (logins, scans, PDF downloads, deletions) |
| `refresh_tokens` | 2 | `id` (Integer) | `users.id` | Cryptographic refresh tokens with rotation and family reuse detection |
| `revoked_tokens` | 1 | `id` (Integer) | None | Blacklisted JWT access tokens revoked upon user logout |
| `prediction_jobs` | 0 | `id` (String UUID) | `users.id` | Real-time SSE streaming inference job status and progress tracking |
| `alembic_version` | 1 | `version_num` | None | Active database migration schema version |

---

## 3. History Management & Query Capabilities

### 3.1 Multi-Parameter Server-Side Filtering
The history query engine (`_build_filtered_history_query`) enables fast, parameterized filtering:
- **Condition Filter:** Exact matching on `predicted_class` (or `ALL`).
- **Severity Filter:** Exact matching on `severity_level` (or `ALL`).
- **Date Range Filter:** ISO 8601 `start_date` and `end_date` bounds.
- **Search Parameter:** Substring matching on filename, condition, or notes.
- **Role-Based Isolation:** Non-admin users are strictly restricted to their own scans (`user_id == current_user.id`), while administrators possess global review access.

### 3.2 Tabular CSV Export
The `/api/history/export/csv` endpoint streams all matching prediction records in RFC-4180 CSV format:
```csv
id,created_at,predicted_class,confidence,affected_area_percentage,severity_level,image_filename
48,2026-11-19T08:10:00,Leaf Variegation,0.9889,23.74,MODERATE,LV00044.jpg
```

### 3.3 Record Deletion & Cleanup
Endpoints `DELETE /api/history/{id}` and `DELETE /api/predictions/{id}` enable safe deletion of prediction records and cascade cleanups, logged permanently in `audit_logs`.

---

## 4. Automated PDF Diagnostic Report Generation

The reporting engine ([report_service.py](file:///D:/Main_Project/cotton_leaf_ai/backend/app/services/report_service.py)) utilizes **ReportLab** to generate publication-grade agricultural diagnosis advisories:
- **Header:** System title, institution subtitle, report tracking ID, and UTC timestamp.
- **Diagnostic Metrics Table:** Primary condition, confidence score, 2D affected foliar area percentage, exact pixel breakdown, and clinical severity tier.
- **Multi-Layer Visual Figures:** Embeds the original photograph, Grad-CAM attention heatmap, and U-Net segmentation mask.
- **Agronomic Action Plan:** Synthesizes ICAR-CICR first steps, severity-specific priority actions, preventive practices, and chemical warnings.
- **Footer:** Academic disclaimer and page numbering.

### Verification Test:
- Successfully generated sample report [test_diagnostic_report_48.pdf](file:///D:/Main_Project/cotton_leaf_ai/artifacts/reports/test_diagnostic_report_48.pdf) (990,378 bytes) with 100% visual and layout fidelity.

---

## 5. Artifact Registry

- **Production Database:** `cotton_leaf_ai.db` (458,752 bytes)
- **Pre-Phase 3 Backup:** `artifacts/backups/cotton_leaf_ai_pre_phase3.db` (294,912 bytes)
- **Sample Diagnostic PDF:** [test_diagnostic_report_48.pdf](file:///D:/Main_Project/cotton_leaf_ai/artifacts/reports/test_diagnostic_report_48.pdf)
- **Report Generation Service:** [report_service.py](file:///D:/Main_Project/cotton_leaf_ai/backend/app/services/report_service.py)
- **History Routing Engine:** [history.py](file:///D:/Main_Project/cotton_leaf_ai/backend/app/routes/history.py)

---

## 6. Sign-off and Readiness for Phase 8
Phase 7 is completed. Database schema integrity, history filtering, CSV export, and PDF report generation are verified. Proceed to **Phase 8: Modern React Frontend UI/UX**.
