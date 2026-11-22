"""
Comprehensive Integration Tests for Cotton Leaf AI Backend API.

Covers:
- Root & Health telemetry (/api/health)
- User registration (enforced farmer role), login, and JWT validation
- User ownership enforcement (403 across private scans)
- Unified inference (/api/predictions/analyze) + component endpoints
  (/api/classification/predict, /api/detection/predict, /api/segmentation/predict,
   /api/severity/calculate, /api/explainability/gradcam)
- Paginated & filtered history (/api/history, /api/history/{id}, /api/predictions/history)
- Dashboard analytics (/api/dashboard/stats, /api/dashboard/statistics)
- Admin RBAC enforcement (/api/admin/overview, /api/admin/users)
- PDF ReportLab diagnostic report generation (/api/reports/{id})
"""

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_TEMP_DB_DIR = None
if "DATABASE_URL" not in os.environ:
    _TEMP_DB_DIR = tempfile.TemporaryDirectory(prefix="cotton_leaf_pytest_")
    os.environ["DATABASE_URL"] = f"sqlite:///{Path(_TEMP_DB_DIR.name) / 'test_api.db'}"
    os.environ.setdefault("ARTIFACTS_DIR", str(Path(_TEMP_DB_DIR.name) / "artifacts"))

from backend.app.config import get_settings
get_settings.cache_clear()

from fastapi.testclient import TestClient
from backend.app.auth import hash_password
from backend.app.database import Base, SessionLocal, engine, init_db
from backend.app.main import app
from backend.app.models import User


class TestBackendAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)

        # Ensure isolated test accounts exist in DB
        db = SessionLocal()
        try:
            admin = db.query(User).filter(User.username == "pytest_admin").first()
            if not admin:
                db.add(User(
                    username="pytest_admin",
                    email="pytest_admin@cotton.local",
                    password_hash=hash_password("AdminSecret123!"),
                    role="admin",
                ))
            farmer = db.query(User).filter(User.username == "demofarmer").first()
            if not farmer:
                db.add(User(
                    username="demofarmer",
                    email="farmer@example.com",
                    password_hash=hash_password("SecretPassword123!"),
                    role="farmer",
                ))
            other = db.query(User).filter(User.username == "otherfarmer").first()
            if not other:
                db.add(User(
                    username="otherfarmer",
                    email="other@example.com",
                    password_hash=hash_password("SecretPassword123!"),
                    role="farmer",
                ))
            db.commit()
        finally:
            db.close()

    def _make_leaf_bytes(self, color=(60, 140, 50)) -> io.BytesIO:
        cand = Path("D:/Main_Project/dataset/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection/Original Dataset/Bacterial Blight/BBC00045.jpg")
        if cand.exists():
            buf = io.BytesIO()
            with open(cand, "rb") as f:
                buf.write(f.read())
            buf.seek(0)
            return buf
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), color=color).save(buf, format="JPEG")
        buf.seek(0)
        return buf

    def test_01_root_and_health(self):
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)

        res_health = self.client.get("/api/health")
        self.assertEqual(res_health.status_code, 200)
        health = res_health.json()
        self.assertEqual(health["status"], "healthy")
        self.assertTrue(health["database_connected"])
        self.assertTrue(health["classification_checkpoint"])
        self.assertIn("models", health)

    def test_02_auth_and_role_hardening(self):
        # Attempt self-assigning admin via registration -> server must enforce 'farmer'
        res_reg = self.client.post("/api/auth/register", json={
            "username": "demofarmer",
            "email": "farmer@example.com",
            "password": "SecretPassword123!",
            "role": "admin",
        })
        self.assertIn(res_reg.status_code, [201, 400])
        if res_reg.status_code == 201:
            self.assertEqual(res_reg.json()["role"], "farmer")

        # Register second farmer for ownership isolation check
        self.client.post("/api/auth/register", json={
            "username": "otherfarmer",
            "email": "other@example.com",
            "password": "SecretPassword123!",
            "role": "farmer",
        })

        res_login = self.client.post("/api/auth/login", json={
            "username_or_email": "demofarmer",
            "password": "SecretPassword123!",
        })
        self.assertEqual(res_login.status_code, 200)
        token = res_login.json()["access_token"]
        self.assertEqual(res_login.json()["user"]["role"], "farmer")

        res_me = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(res_me.status_code, 200)
        self.assertEqual(res_me.json()["username"], "demofarmer")

    def test_03_unified_pipeline_component_routes_and_security(self):
        tok_farmer = self.client.post("/api/auth/login", json={
            "username_or_email": "demofarmer",
            "password": "SecretPassword123!",
        }).json()["access_token"]
        tok_other = self.client.post("/api/auth/login", json={
            "username_or_email": "otherfarmer",
            "password": "SecretPassword123!",
        }).json()["access_token"]
        tok_admin = self.client.post("/api/auth/login", json={
            "username_or_email": "pytest_admin",
            "password": "AdminSecret123!",
        }).json()["access_token"]

        h_farmer = {"Authorization": f"Bearer {tok_farmer}"}
        h_other = {"Authorization": f"Bearer {tok_other}"}
        h_admin = {"Authorization": f"Bearer {tok_admin}"}

        # 1. Unified analyze endpoint
        res_analyze = self.client.post(
            "/api/predictions/analyze",
            files={"file": ("test_leaf.jpg", self._make_leaf_bytes(), "image/jpeg")},
            headers=h_farmer,
        )
        self.assertEqual(res_analyze.status_code, 200)
        pred_data = res_analyze.json()
        pid = pred_data["id"]
        self.assertIn("predicted_class", pred_data)
        self.assertEqual(len(pred_data["all_probabilities"]), 7)
        self.assertTrue(pred_data["gradcam_comparison_url"].startswith("/artifacts/explainability/"))
        self.assertIn(pred_data["detection_status"], ["UNAVAILABLE", "EXPERIMENTAL", "SUCCESS"])
        self.assertIn(pred_data["segmentation_status"], ["UNAVAILABLE", "EXPERIMENTAL", "SUCCESS"])

        # 2. Component endpoints
        cls_res = self.client.post(
            "/api/classification/predict",
            files={"file": ("cls.jpg", self._make_leaf_bytes(), "image/jpeg")},
            headers=h_farmer,
        )
        self.assertEqual(cls_res.status_code, 200)
        self.assertEqual(len(cls_res.json()["all_probabilities"]), 7)

        det_res = self.client.post(
            "/api/detection/predict",
            files={"file": ("det.jpg", self._make_leaf_bytes(), "image/jpeg")},
            headers=h_farmer,
        )
        self.assertEqual(det_res.status_code, 200)
        self.assertIn(det_res.json()["status"], ["UNAVAILABLE", "EXPERIMENTAL", "SUCCESS"])

        seg_res = self.client.post(
            "/api/segmentation/predict",
            files={"file": ("seg.jpg", self._make_leaf_bytes(), "image/jpeg")},
            headers=h_farmer,
        )
        self.assertEqual(seg_res.status_code, 200)
        self.assertIn(seg_res.json()["status"], ["UNAVAILABLE", "EXPERIMENTAL", "SUCCESS"])

        sev_res = self.client.post(
            "/api/severity/calculate",
            json={
                "predicted_class": "Bacterial Blight",
                "confidence": 0.94,
                "leaf_pixels": 1000,
                "disease_pixels": 200,
                "healthy_pixels": 800,
            },
            headers=h_farmer,
        )
        self.assertEqual(sev_res.status_code, 200)
        self.assertEqual(sev_res.json()["affected_area_percentage"], 20.0)
        self.assertEqual(sev_res.json()["level"], "MODERATE")

        cam_res = self.client.post(
            "/api/explainability/gradcam",
            files={"file": ("cam.jpg", self._make_leaf_bytes(), "image/jpeg")},
            headers=h_farmer,
        )
        self.assertEqual(cam_res.status_code, 200)
        self.assertIn("heatmap_url", cam_res.json())

        # 3. Ownership isolation (other farmer blocked from viewing demofarmer's scan & PDF)
        self.assertEqual(self.client.get(f"/api/predictions/{pid}", headers=h_other).status_code, 403)
        self.assertEqual(self.client.get(f"/api/history/{pid}", headers=h_other).status_code, 403)
        self.assertEqual(self.client.get(f"/api/reports/{pid}", headers=h_other).status_code, 403)

        # Owner can retrieve detail & paginated history
        self.assertEqual(self.client.get(f"/api/history/{pid}", headers=h_farmer).status_code, 200)
        hist_pag = self.client.get("/api/history?page=1&page_size=5", headers=h_farmer)
        self.assertEqual(hist_pag.status_code, 200)
        self.assertGreater(hist_pag.json()["total"], 0)

        # Legacy history route preserved
        self.assertEqual(self.client.get("/api/predictions/history", headers=h_farmer).status_code, 200)

        # 4. Dashboard routes
        dash_res = self.client.get("/api/dashboard/stats", headers=h_farmer)
        self.assertEqual(dash_res.status_code, 200)
        self.assertIn("model_readiness", dash_res.json())
        self.assertIn("trends", dash_res.json())

        # 5. Admin RBAC protection
        self.assertEqual(self.client.get("/api/admin/overview", headers=h_farmer).status_code, 403)
        self.assertEqual(self.client.get("/api/admin/users", headers=h_farmer).status_code, 403)
        admin_ov = self.client.get("/api/admin/overview", headers=h_admin)
        self.assertEqual(admin_ov.status_code, 200)
        self.assertGreater(admin_ov.json()["total_users"], 0)

        # 6. PDF Download
        rep_res = self.client.get(f"/api/reports/{pid}", headers=h_farmer)
        self.assertEqual(rep_res.status_code, 200)
        self.assertEqual(rep_res.headers["content-type"], "application/pdf")
        self.assertTrue(rep_res.content.startswith(b"%PDF-"))

        # 7. Filtered History CSV Export
        csv_res = self.client.get("/api/history/export/csv", headers=h_farmer)
        self.assertEqual(csv_res.status_code, 200)
        self.assertIn("text/csv", csv_res.headers["content-type"])
        self.assertIn("prediction_id,image_filename,predicted_class", csv_res.text)
        self.assertIn(f"{pid},", csv_res.text)

        # 8. Admin system evaluation charts, audit logs & account status toggles
        sys_res = self.client.get("/api/admin/system", headers=h_admin)
        self.assertEqual(sys_res.status_code, 200)
        sys_json = sys_res.json()
        self.assertIn("metric_charts", sys_json)
        self.assertIn("classifier_full_metrics", sys_json)
        if sys_json["metric_charts"].get("confusion_matrix"):
            chart_res = self.client.get(sys_json["metric_charts"]["confusion_matrix"])
            self.assertEqual(chart_res.status_code, 200)

        users_res = self.client.get("/api/admin/users", headers=h_admin)
        self.assertEqual(users_res.status_code, 200)
        other_user = next(u for u in users_res.json() if u["username"] == "otherfarmer")
        admin_user_row = next(u for u in users_res.json() if u["username"] == "pytest_admin")

        # Self-deactivation must be rejected
        self.assertEqual(
            self.client.patch(
                f"/api/admin/users/{admin_user_row['id']}/status",
                json={"is_active": False},
                headers=h_admin,
            ).status_code,
            400,
        )

        # Deactivate otherfarmer -> login fails with 403 -> reactivate -> login succeeds
        deact_res = self.client.patch(
            f"/api/admin/users/{other_user['id']}/status",
            json={"is_active": False},
            headers=h_admin,
        )
        self.assertEqual(deact_res.status_code, 200)
        self.assertFalse(deact_res.json()["is_active"])
        self.assertEqual(
            self.client.post(
                "/api/auth/login",
                json={"username_or_email": "otherfarmer", "password": "SecretPassword123!"},
            ).status_code,
            403,
        )
        react_res = self.client.patch(
            f"/api/admin/users/{other_user['id']}/status",
            json={"is_active": True},
            headers=h_admin,
        )
        self.assertEqual(react_res.status_code, 200)
        self.assertTrue(react_res.json()["is_active"])

        # 9. Delete history ownership enforcement & deletion
        self.assertEqual(self.client.delete(f"/api/history/{pid}", headers=h_other).status_code, 403)
        del_res = self.client.delete(f"/api/history/{pid}", headers=h_farmer)
        self.assertEqual(del_res.status_code, 200)
        self.assertEqual(del_res.json()["status"], "deleted")
        self.assertEqual(self.client.get(f"/api/history/{pid}", headers=h_farmer).status_code, 404)

        # 10. Audit logs RBAC & verification
        self.assertEqual(self.client.get("/api/admin/audit-logs", headers=h_farmer).status_code, 403)
        audit_res = self.client.get("/api/admin/audit-logs?limit=25", headers=h_admin)
        self.assertEqual(audit_res.status_code, 200)
        actions = {row["action"] for row in audit_res.json()}
        self.assertTrue({"LEAF_SCAN", "PDF_REPORT_DOWNLOAD", "HISTORY_CSV_EXPORT", "USER_STATUS_CHANGE", "PREDICTION_DELETED"}.issubset(actions))

    @classmethod
    def tearDownClass(cls):
        if _TEMP_DB_DIR is not None:
            engine.dispose()
            _TEMP_DB_DIR.cleanup()


if __name__ == "__main__":
    unittest.main()
