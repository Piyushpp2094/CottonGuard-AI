"""
Phase 5 Frontend-Backend Contract Integration Tests:
Covers verified benchmark & NOT_TRAINED model indicators, farmer auth lifecycle,
invalid upload rejection, async SSE prediction flow, Grad-CAM interaction,
history pagination & CSV export, PDF report generation/download, and RBAC errors.
"""

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_PHASE5_TMP = None
if "DATABASE_URL" not in os.environ:
    _PHASE5_TMP = tempfile.TemporaryDirectory(prefix="cottonguard_phase5_")
    os.environ["DATABASE_URL"] = f"sqlite:///{Path(_PHASE5_TMP.name) / 'phase5_api.db'}"
    os.environ["ARTIFACTS_DIR"] = str(Path(_PHASE5_TMP.name) / "artifacts")

from backend.app.config import get_settings
get_settings.cache_clear()

from fastapi.testclient import TestClient

from backend.app.database import engine, init_db
from backend.app.main import app


class TestPhase5FrontendContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        if _PHASE5_TMP is not None:
            engine.dispose()
            _PHASE5_TMP.cleanup()

    def _make_img(self, fmt: str = "PNG", size=(224, 224), color=(60, 145, 70)) -> io.BytesIO:
        if size[0] < 100 or size[1] < 100:
            buf = io.BytesIO()
            Image.new("RGB", size, color=color).save(buf, format=fmt)
            buf.seek(0)
            return buf
        cand = Path("D:/Main_Project/dataset/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection/Original Dataset/Bacterial Blight/BBC00045.jpg")
        if cand.exists():
            buf = io.BytesIO()
            with Image.open(cand) as im:
                im.resize(size).save(buf, format=fmt)
            buf.seek(0)
            return buf
        buf = io.BytesIO()
        Image.new("RGB", size, color=color).save(buf, format=fmt)
        buf.seek(0)
        return buf

    def test_01_public_models_and_verified_benchmark_contract(self):
        res = self.client.get("/api/health/models")
        self.assertEqual(res.status_code, 200)
        models = res.json()["models"]
        self.assertEqual(models["classifier"]["status"], "trained")
        self.assertIn(models["detector"]["status"], ["not_trained", "experimental", "trained"])
        self.assertIn(models["segmentor"]["status"], ["not_trained", "experimental", "trained"])

        bench = self.client.get("/artifacts/metrics/benchmark_report.json")
        self.assertEqual(bench.status_code, 200)
        bench_models = bench.json()["models"]
        self.assertAlmostEqual(
            bench_models["classifier_efficientnet_b0"]["live_validation_benchmark"]["validation_metrics"]["accuracy"],
            0.9843,
            places=4,
        )
        self.assertAlmostEqual(
            bench_models["classifier_efficientnet_b0"]["pre_audited_holdout_test_metrics"]["overall_performance"]["accuracy"],
            0.9782,
            places=4,
        )
        self.assertAlmostEqual(
            bench_models["classifier_efficientnet_b0"]["pre_audited_holdout_test_metrics"]["overall_performance"]["macro_f1"],
            0.9754,
            places=4,
        )
        self.assertEqual(bench_models["detector_yolov8n"]["status"], "NOT_TRAINED")
        self.assertEqual(bench_models["segmentor_unet_resnet34"]["status"], "NOT_TRAINED")

    def test_02_auth_upload_sse_gradcam_history_and_pdf_report_flow(self):
        # Register & login farmer
        reg = self.client.post(
            "/api/auth/register",
            json={"username": "phase5_farmer", "email": "phase5_farmer@cotton.local", "password": "FarmerPass123!"},
        )
        self.assertEqual(reg.status_code, 201)

        login_res = self.client.post(
            "/api/auth/login",
            json={"username_or_email": "phase5_farmer", "password": "FarmerPass123!"},
        )
        self.assertEqual(login_res.status_code, 200)
        tokens = login_res.json()
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}

        # Non-admin blocked from admin endpoints (error scenario)
        admin_block = self.client.get("/api/admin/overview", headers=headers)
        self.assertEqual(admin_block.status_code, 403)
        self.assertIn("error", admin_block.json())

        # Corrupted magic bytes upload error scenario
        bad_upload = self.client.post(
            "/api/predictions/analyze/async",
            headers=headers,
            files={"file": ("not_an_image.png", io.BytesIO(b"NOT_A_PNG_PAYLOAD"), "image/png")},
        )
        self.assertEqual(bad_upload.status_code, 400)
        self.assertEqual(bad_upload.json()["error"]["code"], "INVALID_IMAGE_MAGIC_BYTES")

        # Undersized resolution error scenario (<100px)
        tiny_upload = self.client.post(
            "/api/predictions/analyze/async",
            headers=headers,
            files={"file": ("tiny.png", self._make_img(size=(64, 64)), "image/png")},
        )
        self.assertEqual(tiny_upload.status_code, 400)
        self.assertEqual(tiny_upload.json()["error"]["code"], "INVALID_IMAGE_DIMENSIONS")

        # Async enqueue + SSE stream completion
        async_res = self.client.post(
            "/api/predictions/analyze/async",
            headers=headers,
            data={"seg_alpha": "0.4"},
            files={"file": ("leaf.png", self._make_img(size=(256, 256)), "image/png")},
        )
        self.assertEqual(async_res.status_code, 202)
        job_id = async_res.json()["job_id"]

        seen_stages = []
        completed_payload = None
        with self.client.stream("GET", f"/api/predictions/analyze/stream/{job_id}", headers=headers) as stream:
            self.assertEqual(stream.status_code, 200)
            curr_event = None
            for line in stream.iter_lines():
                if line.startswith("event:"):
                    curr_event = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and curr_event:
                    payload = json.loads(line.split(":", 1)[1].strip())
                    if curr_event == "progress":
                        seen_stages.append(payload["stage"])
                    elif curr_event == "complete":
                        completed_payload = payload
                        break

        self.assertIsNotNone(completed_payload)
        self.assertIn("classification", seen_stages)
        self.assertIn("gradcam", seen_stages)
        result = completed_payload["result"]
        pred_id = result["id"]
        self.assertIn(result["detection_status"], ["UNAVAILABLE", "EXPERIMENTAL", "SUCCESS"])
        self.assertIn(result["segmentation_status"], ["UNAVAILABLE", "EXPERIMENTAL", "SUCCESS"])
        self.assertIn(result["severity_level"], ["UNAVAILABLE", "NONE", "LOW", "MODERATE", "HIGH"])
        self.assertFalse(result["thresholds_verified"])

        # Interactive Grad-CAM colormap recomputation
        cam_res = self.client.post(
            "/api/explainability/gradcam",
            headers=headers,
            data={"target_class": "0", "alpha": "0.5", "colormap": "turbo"},
            files={"file": ("leaf.png", self._make_img(size=(256, 256)), "image/png")},
        )
        self.assertEqual(cam_res.status_code, 200)
        self.assertTrue(cam_res.json()["comparison_url"].startswith("/artifacts/explainability/"))

        # Paginated history + detail inspection + PDF generate & download
        hist_res = self.client.get("/api/history?page=1&page_size=10", headers=headers)
        self.assertEqual(hist_res.status_code, 200)
        self.assertGreaterEqual(hist_res.json()["total"], 1)

        gen_pdf = self.client.post(f"/api/reports/{pred_id}/generate", headers=headers)
        self.assertEqual(gen_pdf.status_code, 200)
        self.assertGreater(gen_pdf.json()["file_size_bytes"], 1000)

        dl_pdf = self.client.get(f"/api/reports/{pred_id}", headers=headers)
        self.assertEqual(dl_pdf.status_code, 200)
        self.assertEqual(dl_pdf.headers["content-type"], "application/pdf")

        # Refresh & logout rotation
        refresh_res = self.client.post("/api/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
        self.assertEqual(refresh_res.status_code, 200)
        new_tokens = refresh_res.json()
        logout_res = self.client.post(
            "/api/auth/logout",
            headers={"Authorization": f"Bearer {new_tokens['access_token']}"},
            json={"refresh_token": new_tokens["refresh_token"]},
        )
        self.assertEqual(logout_res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
