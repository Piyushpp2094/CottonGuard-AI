"""
Automated Verification & Regression Test Suite for CottonGuard AI.

Validates the mandatory Cotton Leaf Verification Gate:
- Quality gate catches low-resolution, blurry, dark, and blank images.
- Domain verifier blocks human faces, documents, objects, and non-cotton leaves.
- Verifier strictly halts downstream disease classification, U-Net segmentation,
  YOLO detection, Grad-CAM generation, and agronomic guidance for invalid images.
- Standalone sub-endpoints (/classification/predict, /detection/predict, /segmentation/predict,
  /explainability/gradcam) enforce the verification gate.
- PDF diagnostic reports are strictly blocked (HTTP 400) for unverified/rejected images.
- Valid cotton leaves proceed through full 7-stage diagnostic analysis.
"""

import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

PROJECT_ROOT = Path("D:/Main_Project/cotton_leaf_ai")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.auth import create_access_token, hash_password
from backend.app.config import get_settings
from backend.app.database import SessionLocal, init_db
from backend.app.main import app
from backend.app.models import Prediction, User
from backend.app.services.prediction_service import prediction_service
from src.data.dataset import CLASS_NAMES
from src.validation.image_quality import ImageQualityValidator


class TestCottonVerificationPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)
        cls.db = SessionLocal()

        # Set up test user
        cls.test_user = cls.db.query(User).filter(User.username == "verify_farmer").first()
        if not cls.test_user:
            cls.test_user = User(
                username="verify_farmer",
                email="verify_farmer@cotton.local",
                password_hash=hash_password("VerifyPass123!"),
                role="farmer",
                is_active=True,
            )
            cls.db.add(cls.test_user)
            cls.db.commit()
            cls.db.refresh(cls.test_user)

        res_login = cls.client.post("/api/auth/login", json={
            "username_or_email": "verify_farmer",
            "password": "VerifyPass123!",
        })
        cls.token = res_login.json()["access_token"]
        cls.headers = {"Authorization": f"Bearer {cls.token}"}

        # Locate test fixture images
        cls.non_leaf_face = next((PROJECT_ROOT / "data" / "verification" / "images" / "non_leaf" / "faces").glob("*.jpg"), None)
        cls.non_leaf_doc = next((PROJECT_ROOT / "data" / "verification" / "images" / "non_leaf" / "documents").glob("*.png"), None)
        cls.non_cotton_leaf = next((PROJECT_ROOT / "data" / "verification" / "images" / "non_cotton_leaf").glob("*.jpg"), None)
        if not cls.non_cotton_leaf:
            cls.non_cotton_leaf = next((PROJECT_ROOT / "data" / "verification" / "images" / "non_cotton_leaf").glob("*.JPG"), None)

        cls.cotton_leaf = Path("D:/Main_Project/dataset/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection/Original Dataset/Bacterial Blight/BBC00045.jpg")
        if not cls.cotton_leaf.exists():
            # Fallback to any cotton image in dataset
            cls.cotton_leaf = next(Path("D:/Main_Project/dataset").rglob("*.jpg"), None)

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_01_image_quality_validator_rules(self):
        """Test deterministic quality validation catches blurry, dark, and blank images."""
        validator = ImageQualityValidator()

        # Sub-resolution
        small_img = Image.new("RGB", (64, 64), (100, 100, 100))
        res_small = validator.validate(small_img)
        self.assertFalse(res_small.is_valid)
        self.assertEqual(res_small.error_code, "IMAGE_TOO_SMALL")

        # Extremely dark
        dark_img = Image.new("RGB", (300, 300), (10, 10, 10))
        res_dark = validator.validate(dark_img)
        self.assertFalse(res_dark.is_valid)
        self.assertEqual(res_dark.error_code, "IMAGE_UNDEREXPOSED")

        # Extremely overexposed / white
        white_img = Image.new("RGB", (300, 300), (250, 250, 250))
        res_white = validator.validate(white_img)
        self.assertFalse(res_white.is_valid)
        self.assertEqual(res_white.error_code, "IMAGE_OVEREXPOSED")

        # Low contrast flat grey
        grey_img = Image.new("RGB", (300, 300), (128, 128, 128))
        res_grey = validator.validate(grey_img)
        self.assertFalse(res_grey.is_valid)
        self.assertEqual(res_grey.error_code, "IMAGE_LOW_CONTRAST")

        # Sharp textured image (should pass)
        np_img = np.random.randint(40, 200, (400, 400, 3), dtype=np.uint8)
        good_img = Image.fromarray(np_img)
        res_good = validator.validate(good_img)
        self.assertTrue(res_good.is_valid)
        self.assertEqual(res_good.status, "PASS")

    def test_02_pipeline_rejection_on_human_photo(self):
        """Human photo must be rejected as non-plant; disease inference must be halted."""
        self.assertIsNotNone(self.non_leaf_face, "Human face test image must exist")

        with open(self.non_leaf_face, "rb") as f:
            file_bytes = f.read()

        response = self.client.post(
            "/api/predictions/analyze",
            headers=self.headers,
            files={"file": ("person.jpg", file_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertFalse(data["is_cotton_leaf"])
        self.assertIn(data["status"], ["REJECTED_NON_PLANT", "REJECTED_NON_COTTON"])
        self.assertEqual(data["predicted_class"], "Not a Cotton Leaf")
        self.assertIsNotNone(data["rejection_reason"])
        self.assertIsNone(data["affected_area_percentage"])
        self.assertEqual(data["severity_level"], "UNAVAILABLE")
        self.assertIsNone(data["gradcam_comparison_url"])

    def test_03_pipeline_rejection_on_document_image(self):
        """Document / text screenshot must be rejected as non-plant."""
        self.assertIsNotNone(self.non_leaf_doc, "Document test image must exist")

        with open(self.non_leaf_doc, "rb") as f:
            file_bytes = f.read()

        response = self.client.post(
            "/api/predictions/analyze",
            headers=self.headers,
            files={"file": ("doc.png", file_bytes, "image/png")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertFalse(data["is_cotton_leaf"])
        self.assertIn(data["status"], ["REJECTED_NON_PLANT", "REJECTED_QUALITY"])
        self.assertIn(data["predicted_class"], ["Not a Cotton Leaf", "Invalid Image Quality"])
        self.assertIsNotNone(data["rejection_reason"])
        self.assertIsNone(data["gradcam_heatmap_url"])

    def test_04_pipeline_rejection_on_non_cotton_leaf(self):
        """Other crop leaves (tomato, potato, etc.) must be rejected."""
        self.assertIsNotNone(self.non_cotton_leaf, "Non-cotton leaf image must exist")

        with open(self.non_cotton_leaf, "rb") as f:
            file_bytes = f.read()

        response = self.client.post(
            "/api/predictions/analyze",
            headers=self.headers,
            files={"file": ("tomato_leaf.jpg", file_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertFalse(data["is_cotton_leaf"])
        self.assertIn(data["status"], ["REJECTED_NON_COTTON", "REJECTED_NON_PLANT"])
        self.assertEqual(data["predicted_class"], "Not a Cotton Leaf")
        self.assertIsNone(data["affected_area_percentage"])

    def test_05_pipeline_rejection_on_blurry_image(self):
        """Severely blurry / blank image must be rejected by Image Quality gate."""
        # Create solid dark image
        img = Image.new("RGB", (256, 256), (15, 15, 15))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        response = self.client.post(
            "/api/predictions/analyze",
            headers=self.headers,
            files={"file": ("dark_blurry.jpg", buf.getvalue(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertFalse(data["is_cotton_leaf"])
        self.assertEqual(data["status"], "REJECTED_QUALITY")
        self.assertEqual(data["predicted_class"], "Invalid Image Quality")
        self.assertIsNotNone(data["rejection_reason"])

    def test_06_pipeline_acceptance_on_valid_cotton_leaf(self):
        """Genuine cotton leaf must pass verification and receive full diagnostic analysis."""
        self.assertIsNotNone(self.cotton_leaf, "Cotton leaf test image must exist")

        with open(self.cotton_leaf, "rb") as f:
            file_bytes = f.read()

        response = self.client.post(
            "/api/predictions/analyze",
            headers=self.headers,
            files={"file": ("cotton_leaf.jpg", file_bytes, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["is_cotton_leaf"])
        self.assertEqual(data["verification_status"], "COTTON_LEAF_VERIFIED")
        self.assertIn(data["predicted_class"], CLASS_NAMES)
        self.assertGreater(data["confidence"], 0.0)
        self.assertIn(data["severity_level"], ["LOW", "MEDIUM", "HIGH", "UNAVAILABLE"])
        self.assertIsNotNone(data["recommendations"])

    def test_07_pdf_report_blocked_for_rejected_predictions(self):
        """PDF Report download MUST return HTTP 400 for rejected predictions."""
        # 1. Submit an invalid image to obtain rejected prediction id
        with open(self.non_leaf_face, "rb") as f:
            file_bytes = f.read()

        res_analyze = self.client.post(
            "/api/predictions/analyze",
            headers=self.headers,
            files={"file": ("rejected_face.jpg", file_bytes, "image/jpeg")},
        )
        pred_id = res_analyze.json()["id"]

        # 2. Attempt to download PDF
        res_report = self.client.get(
            f"/api/reports/{pred_id}",
            headers=self.headers,
        )
        self.assertEqual(res_report.status_code, 400)
        self.assertIn("not verified as a valid cotton leaf", res_report.json()["detail"])

    def test_08_sub_endpoints_blocked_on_invalid_images(self):
        """Direct access to component endpoints without cotton leaf verification must fail with HTTP 400."""
        with open(self.non_leaf_face, "rb") as f:
            face_bytes = f.read()

        # Classification
        res_cls = self.client.post(
            "/api/classification/predict",
            headers=self.headers,
            files={"file": ("face.jpg", face_bytes, "image/jpeg")},
        )
        self.assertEqual(res_cls.status_code, 400)

        # Detection
        res_det = self.client.post(
            "/api/detection/predict",
            headers=self.headers,
            files={"file": ("face.jpg", face_bytes, "image/jpeg")},
        )
        self.assertEqual(res_det.status_code, 400)

        # Segmentation
        res_seg = self.client.post(
            "/api/segmentation/predict",
            headers=self.headers,
            files={"file": ("face.jpg", face_bytes, "image/jpeg")},
        )
        self.assertEqual(res_seg.status_code, 400)

        # Grad-CAM
        res_cam = self.client.post(
            "/api/explainability/gradcam",
            headers=self.headers,
            files={"file": ("face.jpg", face_bytes, "image/jpeg")},
        )
        self.assertEqual(res_cam.status_code, 400)

    def test_09_spies_assert_downstream_models_never_invoked(self):
        """Spy assertion: Downstream models must never be invoked when verification gate rejects input."""
        with open(self.non_leaf_doc, "rb") as f:
            doc_bytes = f.read()

        with patch.object(prediction_service, "_run_classifier_inference") as mock_cls, \
             patch.object(prediction_service, "_run_unet_segmentation") as mock_unet, \
             patch.object(prediction_service, "_run_yolo_detection") as mock_yolo:

            response = self.client.post(
                "/api/predictions/analyze",
                headers=self.headers,
                files={"file": ("document.png", doc_bytes, "image/png")},
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertFalse(data["is_cotton_leaf"])

            # Assert downstream models were NEVER called!
            mock_cls.assert_not_called()
            mock_unet.assert_not_called()
            mock_yolo.assert_not_called()


if __name__ == "__main__":
    unittest.main()
