# Cotton Leaf AI — REST API Specification

- **Local Base URL:** `http://127.0.0.1:8000` (Proxied via Vite at `http://localhost:5173/api/...`)
- **Interactive Docs:** `http://127.0.0.1:8000/docs` | `http://127.0.0.1:8000/redoc`
- **Authentication:** `Authorization: Bearer <JWT>`

---

## Endpoints Summary

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/api/health` | Public | Live DB status, CUDA RTX 3050 VRAM telemetry, and readiness of Classification, YOLO Detection, U-Net Segmentation, and Grad-CAM. |
| `POST` | `/api/auth/register` | Public | Registers a `farmer` account (public registration locks role to `farmer`). |
| `POST` | `/api/auth/login` | Public | Authenticates credentials and returns JWT Bearer token. |
| `GET` | `/api/auth/me` | Bearer | Returns authenticated user profile. |
| `POST` | `/api/predictions/analyze` | Bearer | Unified pipeline: Classification + Grad-CAM + optional YOLO & U-Net + Severity + ICAR-CICR guidance + DB persistence. |
| `POST` | `/api/classification/predict` | Bearer | Standalone EfficientNet-B0 7-class prediction with full class probabilities & latency. |
| `POST` | `/api/detection/predict` | Bearer | Standalone YOLO region localization (returns honest `UNAVAILABLE` payload until `best_yolo.pt` is trained). |
| `POST` | `/api/segmentation/predict` | Bearer | Standalone U-Net 3-class segmentation (returns honest `UNAVAILABLE` payload until `best_unet.pt` is trained). |
| `POST` | `/api/severity/calculate` | Bearer | Computes 2D severity category from leaf/disease pixel counts or area percentage. |
| `POST` | `/api/explainability/gradcam` | Bearer | Generates Grad-CAM heatmap, overlay, 3-panel comparison, metadata JSON, and scientific disclaimer. |
| `GET` | `/api/history` | Bearer | Paginated history (`page`, `page_size`, `search`, `disease`, `severity`, `start_date`, `end_date`). |
| `GET` | `/api/history/export/csv` | Bearer | Streams filtered prediction history as a downloadable CSV spreadsheet. |
| `GET` | `/api/history/{id}` | Bearer | Full stored scan detail (detections, masks, Grad-CAM, recommendations); owner or admin only. |
| `DELETE` | `/api/history/{id}` | Bearer | Deletes a prediction record and cascades child rows (owner or admin only). |
| `GET` | `/api/predictions/history` | Bearer | Legacy list endpoint preserved for backward compatibility. |
| `GET` | `/api/predictions/{id}` | Bearer | Retrieves individual prediction detail; owner or admin only. |
| `GET` | `/api/dashboard/stats` | Bearer | Farmer KPIs, condition/severity distributions, chronological date trends, model readiness, and recent reports. |
| `GET` | `/api/dashboard/statistics` | Bearer | Alias to `/api/dashboard/stats`. |
| `GET` | `/api/reports/{id}` | Bearer | Generates, logs in `reports` table, and streams diagnostic PDF (`application/pdf`). |
| `GET` | `/api/admin/overview` | Admin | Global counts across all users, global disease breakdown, and live GPU/model status. |
| `GET` | `/api/admin/users` | Admin | User directory with account status (`is_active`) and scan counts. |
| `PATCH` | `/api/admin/users/{id}/role` | Admin | Updates user role (`farmer` or `admin`). |
| `PATCH` | `/api/admin/users/{id}/status` | Admin | Enables or disables a user account (`is_active`). |
| `GET` | `/api/admin/audit-logs` | Admin | Chronological activity & security audit log entries with optional `action` filter. |
| `GET` | `/api/admin/system` | Admin | Dataset manifest audit summary, per-class metrics, evaluation chart URLs, provenance/leakage reports, and annotation readiness. |
