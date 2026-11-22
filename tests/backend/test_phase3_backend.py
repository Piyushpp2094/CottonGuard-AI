"""
Phase 3 Backend Tests: Settings, Alembic Fresh Migration, Model Registry,
Token Refresh/Logout Revocation, Strict Upload Magic-Byte/Dimension Validation,
Structured Error Envelopes, SSE Streaming Jobs, Report Generate, and CUDA OOM CPU Fallback.
"""

import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_PHASE3_TMP = None
if "DATABASE_URL" not in os.environ:
    _PHASE3_TMP = tempfile.TemporaryDirectory(prefix="cottonguard_phase3_")
    os.environ["DATABASE_URL"] = f"sqlite:///{Path(_PHASE3_TMP.name) / 'phase3_api.db'}"
    os.environ["ARTIFACTS_DIR"] = str(Path(_PHASE3_TMP.name) / "artifacts")

from backend.app.config import get_settings
get_settings.cache_clear()

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from backend.app.auth import hash_password
from backend.app.database import SessionLocal, engine, init_db
from backend.app.main import app
from backend.app.models import User
from backend.app.services.prediction_service import prediction_service


class TestPhase3Backend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)
        cls.client.__enter__()

        db = SessionLocal()
        try:
            admin = db.query(User).filter(User.username == "phase3_admin").first()
            if not admin:
                db.add(
                    User(
                        username="phase3_admin",
                        email="phase3_admin@cotton.local",
                        password_hash=hash_password("AdminPass123!"),
                        role="admin",
                    )
                )
                db.commit()
        finally:
            db.close()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)
        if _PHASE3_TMP is not None:
            engine.dispose()
            _PHASE3_TMP.cleanup()

    def _make_img(self, fmt: str = "PNG", size=(224, 224), color=(50, 130, 60)) -> io.BytesIO:
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

    def _register_and_login(self, username: str, email: str, password: str = "FarmerPass123!"):
        self.client.post(
            "/api/auth/register",
            json={"username": username, "email": email, "password": password},
        )
        res = self.client.post(
            "/api/auth/login",
            json={"username_or_email": username, "password": password},
        )
        self.assertEqual(res.status_code, 200)
        return res.json()

    def test_01_model_registry_honest_status_and_admin_endpoint(self):
        res = self.client.get("/api/health/models")
        self.assertEqual(res.status_code, 200)
        models = res.json()["models"]

        self.assertEqual(models["classifier"]["status"], "trained")
        self.assertEqual(len(models["classifier"]["checkpoint_sha256"]), 64)
        self.assertAlmostEqual(models["classifier"]["test_accuracy"], 0.9782, places=3)
        self.assertIn(models["detector"]["status"], ["not_trained", "experimental", "trained"])
        self.assertIn(models["segmentor"]["status"], ["not_trained", "experimental", "trained"])
        if models["detector"]["status"] == "experimental":
            self.assertEqual(models["detector"]["name"], "Faster R-CNN ResNet-50 FPN V2")
            self.assertEqual(len(models["detector"]["checkpoint_sha256"]), 64)
        if models["segmentor"]["status"] == "experimental":
            self.assertEqual(models["segmentor"]["name"], "UNet-ResNet34")
            self.assertEqual(len(models["segmentor"]["checkpoint_sha256"]), 64)

        admin_tok = self.client.post(
            "/api/auth/login",
            json={"username_or_email": "phase3_admin", "password": "AdminPass123!"},
        ).json()["access_token"]
        res_adm = self.client.get("/api/admin/models", headers={"Authorization": f"Bearer {admin_tok}"})
        self.assertEqual(res_adm.status_code, 200)
        self.assertIn(res_adm.json()["registry"]["models"]["detector"]["status"], ["not_trained", "experimental", "trained"])

    def test_02_refresh_rotation_reuse_detection_and_logout_revocation(self):
        tokens = self._register_and_login("rot_farmer", "rot_farmer@cotton.local")
        access1 = tokens["access_token"]
        refresh1 = tokens["refresh_token"]
        self.assertIsNotNone(refresh1)

        # Rotate refresh1 -> refresh2
        res_ref = self.client.post("/api/auth/refresh", json={"refresh_token": refresh1})
        self.assertEqual(res_ref.status_code, 200)
        access2 = res_ref.json()["access_token"]
        refresh2 = res_ref.json()["refresh_token"]
        self.assertNotEqual(refresh1, refresh2)

        # Replay of old refresh1 must fail with 401 and invalidate refresh2
        replay = self.client.post("/api/auth/refresh", json={"refresh_token": refresh1})
        self.assertEqual(replay.status_code, 401)
        self.assertIn("error", replay.json())
        self.assertEqual(
            self.client.post("/api/auth/refresh", json={"refresh_token": refresh2}).status_code,
            401,
        )

        # Logout revokes access token immediately
        me_ok = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {access2}"})
        self.assertEqual(me_ok.status_code, 200)
        out = self.client.post(
            "/api/auth/logout",
            json={},
            headers={"Authorization": f"Bearer {access2}"},
        )
        self.assertEqual(out.status_code, 200)
        me_revoked = self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {access2}"})
        self.assertEqual(me_revoked.status_code, 401)
        self.assertEqual(
            self.client.get("/api/auth/me", headers={"Authorization": f"Bearer {access1}"}).status_code,
            200,
        )

    def test_03_strict_upload_validation_and_error_envelope(self):
        tok = self._register_and_login("val_farmer", "val_farmer@cotton.local")["access_token"]
        headers = {"Authorization": f"Bearer {tok}"}

        # Fake image bytes disguised as .jpg
        bad_magic = self.client.post(
            "/api/classification/predict",
            files={"file": ("fake.jpg", io.BytesIO(b"not-an-image-payload"), "image/jpeg")},
            headers=headers,
        )
        self.assertEqual(bad_magic.status_code, 400)
        self.assertEqual(bad_magic.json()["error"]["code"], "INVALID_IMAGE_MAGIC_BYTES")

        # Undersized image (< 100x100)
        too_small = self.client.post(
            "/api/classification/predict",
            files={"file": ("tiny.png", self._make_img("PNG", size=(64, 64)), "image/png")},
            headers=headers,
        )
        self.assertEqual(too_small.status_code, 400)
        self.assertEqual(too_small.json()["error"]["code"], "INVALID_IMAGE_DIMENSIONS")

        # Oversized payload (> 10 MB)
        oversize = self.client.post(
            "/api/classification/predict",
            files={"file": ("huge.jpg", io.BytesIO(b"\xff\xd8\xff" + b"\x00" * (10 * 1024 * 1024 + 64)), "image/jpeg")},
            headers=headers,
        )
        self.assertEqual(oversize.status_code, 413)
        self.assertEqual(oversize.json()["error"]["code"], "IMAGE_TOO_LARGE")

    def test_04_sse_streaming_job_pipeline_and_report_generate(self):
        owner_tok = self._register_and_login("sse_owner", "sse_owner@cotton.local")["access_token"]
        other_tok = self._register_and_login("sse_other", "sse_other@cotton.local")["access_token"]

        res_async = self.client.post(
            "/api/predictions/analyze",
            data={"stream": "true", "seg_alpha": "0.4"},
            files={"file": ("leaf.png", self._make_img("PNG", size=(240, 240)), "image/png")},
            headers={"Authorization": f"Bearer {owner_tok}"},
        )
        self.assertEqual(res_async.status_code, 202)
        job_info = res_async.json()
        job_id = job_info["job_id"]
        self.assertTrue(job_info["stream_url"].endswith(job_id))

        # Unauthorized subscriber must get 403
        self.assertEqual(
            self.client.get(
                f"/api/predictions/analyze/stream/{job_id}",
                headers={"Authorization": f"Bearer {other_tok}"},
            ).status_code,
            403,
        )

        # Subscribe via SSE and capture real progress + complete event
        events = []
        with self.client.stream(
            "GET",
            f"/api/predictions/analyze/stream/{job_id}",
            headers={"Authorization": f"Bearer {owner_tok}"},
        ) as stream_resp:
            self.assertEqual(stream_resp.status_code, 200)
            cur_event = None
            for line in stream_resp.iter_lines():
                if line.startswith("event:"):
                    cur_event = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and cur_event:
                    events.append((cur_event, json.loads(line.split(":", 1)[1].strip())))
                    if cur_event == "complete":
                        break

        event_names = [e[0] for e in events]
        self.assertIn("progress", event_names)
        self.assertIn("complete", event_names)
        complete_payload = next(p for name, p in events if name == "complete")
        pred_id = complete_payload["prediction_id"]

        # Job status persisted cleanly in DB
        job_status = self.client.get(
            f"/api/predictions/jobs/{job_id}",
            headers={"Authorization": f"Bearer {owner_tok}"},
        )
        self.assertEqual(job_status.status_code, 200)
        self.assertEqual(job_status.json()["status"], "completed")
        self.assertEqual(job_status.json()["prediction_id"], pred_id)

        # Explicit report generation endpoint works and returns PDF filename
        gen_res = self.client.post(
            f"/api/reports/{pred_id}/generate",
            headers={"Authorization": f"Bearer {owner_tok}"},
        )
        self.assertEqual(gen_res.status_code, 200)
        self.assertTrue(gen_res.json()["pdf_filename"].startswith(f"cottonguard_report_{pred_id}_"))

    def test_05_cuda_oom_cpu_fallback(self):
        tok = self._register_and_login("oom_farmer", "oom_farmer@cotton.local")["access_token"]
        orig_classifier = prediction_service.classifier

        calls = {"n": 0}

        def oom_once(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise torch.cuda.OutOfMemoryError("Simulated 6GB VRAM OOM")
            return orig_classifier(*args, **kwargs)

        with mock.patch.object(prediction_service, "classifier", side_effect=oom_once):
            # Preserve .to() method on patched mock wrapper
            prediction_service.classifier.to = orig_classifier.to
            res = self.client.post(
                "/api/classification/predict",
                files={"file": ("leaf.jpg", self._make_img("JPEG", size=(224, 224)), "image/jpeg")},
                headers={"Authorization": f"Bearer {tok}"},
            )

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["device"], "cpu")
        self.assertIn("CPU fallback", body["warning"])

    def test_06_alembic_upgrade_on_fresh_database(self):
        with tempfile.TemporaryDirectory(prefix="alembic_fresh_") as tmp_dir:
            fresh_db = Path(tmp_dir) / "fresh.db"
            prev_url = os.environ.get("DATABASE_URL")
            os.environ["DATABASE_URL"] = f"sqlite:///{fresh_db}"
            try:
                cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
                cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
                cfg.set_main_option("sqlalchemy.url", f"sqlite:///{fresh_db}")
                command.upgrade(cfg, "head")

                conn = sqlite3.connect(fresh_db)
                tables = {
                    row[0]
                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                }
                conn.close()
                self.assertTrue(
                    {
                        "users",
                        "predictions",
                        "prediction_results",
                        "detection_results",
                        "segmentation_results",
                        "severity_results",
                        "reports",
                        "audit_logs",
                        "refresh_tokens",
                        "revoked_tokens",
                        "prediction_jobs",
                    }.issubset(tables)
                )
            finally:
                if prev_url is not None:
                    os.environ["DATABASE_URL"] = prev_url


if __name__ == "__main__":
    unittest.main()
