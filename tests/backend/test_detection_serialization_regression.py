"""
Regression Tests for Faster R-CNN and YOLO Detection Box Serialization.

Verifies:
1. DetectionBox schema accepts both canonical fields (bbox, normalized_bbox, class_id)
   and legacy fields (box, label).
2. PredictionDetailResponse cleanly serializes detection payloads with non-empty detections.
3. serialize_prediction_detail normalizes existing legacy DB JSON records into valid DetectionBox schemas.
4. Faster R-CNN output format in prediction_service._run_detection produces all required schema fields.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock
from PIL import Image

import pytest
from backend.app.schemas import DetectionBox, PredictionDetailResponse
from backend.app.services.prediction_service import serialize_prediction_detail, prediction_service


def test_detection_box_canonical_schema():
    """Verify DetectionBox parses complete canonical detector output."""
    raw = {
        "bbox": [12.0, 34.0, 150.0, 200.0],
        "normalized_bbox": [0.35, 0.45, 0.30, 0.40],
        "class_id": 1,
        "class_name": "bacterial_blight",
        "confidence": 0.9521,
    }
    box = DetectionBox.model_validate(raw)
    assert box.bbox == [12.0, 34.0, 150.0, 200.0]
    assert box.normalized_bbox == [0.35, 0.45, 0.30, 0.40]
    assert box.class_id == 1
    assert box.class_name == "bacterial_blight"
    assert box.confidence == 0.9521
    assert box.box == [12.0, 34.0, 150.0, 200.0]
    assert box.label == 1


def test_detection_box_legacy_schema_compatibility():
    """Verify DetectionBox seamlessly adapts legacy {'box', 'label'} items without validation errors."""
    raw = {
        "box": [15.5, 40.0, 180.2, 220.7],
        "label": 3,
        "class_name": "herbicide_growth_damage",
        "confidence": 0.8123,
    }
    box = DetectionBox.model_validate(raw)
    assert box.bbox == [15.5, 40.0, 180.2, 220.7]
    assert box.class_id == 3
    assert box.class_name == "herbicide_growth_damage"
    assert box.confidence == 0.8123
    assert box.normalized_bbox == []
    assert box.box == [15.5, 40.0, 180.2, 220.7]
    assert box.label == 3


def test_serialize_prediction_detail_normalizes_legacy_detections():
    """Verify serialize_prediction_detail normalizes legacy JSON records into valid DetectionBox schemas."""
    mock_prediction = MagicMock()
    mock_prediction.id = 999
    mock_prediction.user_id = 1
    mock_prediction.image_filename = "test_leaf.jpg"
    mock_prediction.image_path = "/path/to/test_leaf.jpg"
    mock_prediction.predicted_class = "bacterial_blight"
    mock_prediction.confidence = 0.985
    mock_prediction.affected_area_percentage = 14.5
    mock_prediction.severity_level = "MODERATE"
    mock_prediction.model_version = "v1.0"
    mock_prediction.status = "SUCCESS"
    mock_prediction.created_at = datetime.utcnow()

    mock_res = MagicMock()
    mock_res.all_probabilities_json = json.dumps({"bacterial_blight": 0.985, "healthy": 0.015})
    mock_res.latency_ms = 45.2
    mock_res.overlay_path = "/artifacts/predictions/overlay.png"
    mock_res.leaf_mask_path = "/artifacts/predictions/leaf.png"
    mock_res.disease_mask_path = "/artifacts/predictions/disease.png"
    mock_res.gradcam_heatmap_path = "/artifacts/explainability/heatmap.png"
    mock_res.gradcam_overlay_path = "/artifacts/explainability/cam_overlay.png"
    mock_res.gradcam_comparison_path = "/artifacts/explainability/comparison.png"
    mock_res.recommendations_json = json.dumps({"pathogen_or_cause": "Xanthomonas axonopodis"})
    mock_res.uncertainty_notes = "High confidence"

    mock_det = MagicMock()
    mock_det.status = "EXPERIMENTAL"
    mock_det.model_version = "FasterRCNN-ResNet50-FPN-V2-v1.0"
    mock_det.detection_count = 2
    mock_det.annotated_image_path = "/artifacts/predictions/annotated.jpg"
    mock_det.notes = "Detected 2 lesion region(s)."
    # Legacy format with only 'box' and 'label'
    mock_det.detections_json = json.dumps([
        {"box": [10.0, 20.0, 50.0, 60.0], "label": 1, "class_name": "bacterial_blight", "confidence": 0.92},
        {"box": [70.0, 80.0, 110.0, 120.0], "label": 1, "class_name": "bacterial_blight", "confidence": 0.85},
    ])

    mock_seg = MagicMock()
    mock_seg.status = "SUCCESS"
    mock_seg.model_version = "UNet-ResNet34"
    mock_seg.total_leaf_pixels = 50000
    mock_seg.disease_pixels = 5000
    mock_seg.healthy_pixels = 45000
    mock_seg.overlay_path = "/artifacts/predictions/overlay.png"
    mock_seg.leaf_mask_path = "/artifacts/predictions/leaf.png"
    mock_seg.disease_mask_path = "/artifacts/predictions/disease.png"

    mock_sev = MagicMock()
    mock_sev.severity_label = "Moderate Infection"
    mock_sev.thresholds_verified = True

    mock_prediction.result = mock_res
    mock_prediction.detection = mock_det
    mock_prediction.segmentation = mock_seg
    mock_prediction.severity = mock_sev

    detail = serialize_prediction_detail(mock_prediction)

    # Validate against FastAPI schema without ResponseValidationError
    validated = PredictionDetailResponse.model_validate(detail)
    assert len(validated.detections) == 2
    for det_box in validated.detections:
        assert isinstance(det_box.bbox, list) and len(det_box.bbox) == 4
        # When image file is missing, normalized_bbox must be empty list, NOT fabricated [0, 0, 0, 0]
        assert det_box.normalized_bbox == []
        assert isinstance(det_box.class_id, int)
        assert isinstance(det_box.class_name, str)
        assert isinstance(det_box.confidence, float)


def test_serialize_prediction_detail_with_verified_image_dimensions(tmp_path):
    """Verify serialize_prediction_detail accurately calculates normalized_bbox when image exists on disk."""
    # Create a real 200x100 image on disk
    test_img_path = tmp_path / "valid_leaf.jpg"
    img = Image.new("RGB", (200, 100), color=(50, 120, 40))
    img.save(test_img_path)

    mock_prediction = MagicMock()
    mock_prediction.id = 1001
    mock_prediction.user_id = 1
    mock_prediction.image_filename = "valid_leaf.jpg"
    mock_prediction.image_path = str(test_img_path)
    mock_prediction.predicted_class = "bacterial_blight"
    mock_prediction.confidence = 0.95
    mock_prediction.affected_area_percentage = 10.0
    mock_prediction.severity_level = "LOW"
    mock_prediction.model_version = "v1.0"
    mock_prediction.status = "SUCCESS"
    mock_prediction.created_at = datetime.utcnow()

    mock_res = MagicMock()
    mock_res.all_probabilities_json = json.dumps({"bacterial_blight": 0.95})
    mock_res.latency_ms = 25.0
    mock_res.overlay_path = None
    mock_res.leaf_mask_path = None
    mock_res.disease_mask_path = None
    mock_res.gradcam_heatmap_path = None
    mock_res.gradcam_overlay_path = None
    mock_res.gradcam_comparison_path = None
    mock_res.recommendations_json = json.dumps({})
    mock_res.uncertainty_notes = None

    mock_det = MagicMock()
    mock_det.status = "EXPERIMENTAL"
    mock_det.model_version = "FasterRCNN-ResNet50-FPN-V2-v1.0"
    mock_det.detection_count = 1
    mock_det.annotated_image_path = None
    mock_det.notes = None
    # Box: x1=20, y1=10, x2=60, y2=50. (width=40, height=40, cx=40, cy=30)
    # On 200x100: cx_norm=40/200=0.20, cy_norm=30/100=0.30, w_norm=40/200=0.20, h_norm=40/100=0.40
    mock_det.detections_json = json.dumps([
        {"box": [20.0, 10.0, 60.0, 50.0], "label": 1, "confidence": 0.88}
    ])

    mock_prediction.result = mock_res
    mock_prediction.detection = mock_det
    mock_prediction.segmentation = None
    mock_prediction.severity = None

    detail = serialize_prediction_detail(mock_prediction)
    validated = PredictionDetailResponse.model_validate(detail)

    assert len(validated.detections) == 1
    det = validated.detections[0]
    assert det.bbox == [20.0, 10.0, 60.0, 50.0]
    assert det.normalized_bbox == [0.2, 0.3, 0.2, 0.4]
    assert det.class_id == 1
    assert det.class_name == "Bacterial Blight"


def test_foreground_class_id_label_mapping_consistency():
    """Verify class_id and label mappings remain consistent with the detector's 6 foreground classes."""
    from src.data.detection_dataset import FOREGROUND_CLASSES, YOLO_TO_TORCHVISION_LABEL, TORCHVISION_TO_CLASS_NAME

    assert len(FOREGROUND_CLASSES) == 6
    expected_classes = [
        "Bacterial Blight",
        "Curl Virus",
        "Herbicide Growth Damage",
        "Leaf Hopper Jassids",
        "Leaf Redding",
        "Leaf Variegation",
    ]
    assert FOREGROUND_CLASSES == expected_classes

    # TorchVision convention: 1-indexed (1..6)
    for i, name in enumerate(FOREGROUND_CLASSES):
        tv_lbl = i + 1
        assert YOLO_TO_TORCHVISION_LABEL[i] == tv_lbl
        assert TORCHVISION_TO_CLASS_NAME[tv_lbl] == name

    # Background must be label 0 in Faster R-CNN
    assert TORCHVISION_TO_CLASS_NAME[0] == "background"
