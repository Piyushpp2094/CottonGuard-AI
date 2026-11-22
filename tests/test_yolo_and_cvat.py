"""
Unit and Integration Tests for YOLO Detection Architecture, Losses, Metrics,
and CVAT Annotation Ingestion Pipeline (Phases 6 & 8).
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_cvat_annotations import process_cvat_xml_export
from src.models.detector import CottonLeafYOLODetector, draw_detection_boxes
from src.training.detection_losses import YOLODetectionLoss, compute_detection_metrics


class TestYOLOAndCVAT(unittest.TestCase):

    def test_yolo_detector_architecture_and_inference(self):
        model = CottonLeafYOLODetector(num_classes=6, width_mult=0.5, input_size=320)
        dummy = torch.randn(2, 3, 320, 320)
        outputs = model(dummy)
        self.assertEqual(len(outputs), 3)
        # Strides 8, 16, 32 -> 40x40, 20x20, 10x10
        expected_grids = [(40, 8), (20, 16), (10, 32)]
        for (ltrb, obj, cls_log, stride), (g, exp_s) in zip(outputs, expected_grids):
            self.assertEqual(stride, exp_s)
            self.assertEqual(ltrb.shape, (2, 4, g, g))
            self.assertEqual(obj.shape, (2, 1, g, g))
            self.assertEqual(cls_log.shape, (2, 6, g, g))

        criterion = YOLODetectionLoss(num_classes=6)
        targets = [
            {"boxes": torch.tensor([[30.0, 40.0, 120.0, 130.0]]), "labels": torch.tensor([0])},
            {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)},
        ]
        loss = criterion(outputs, targets)
        self.assertGreater(loss.item(), 0.0)

        # Inference and visualization helper
        pil_img = Image.new("RGB", (400, 300), color=(70, 140, 60))
        preds = model.predict(pil_img, conf_threshold=0.99)
        self.assertIsInstance(preds, list)
        boxed = draw_detection_boxes(pil_img, [
            {"bbox": [10.0, 10.0, 80.0, 80.0], "class_id": 0, "class_name": "Bacterial Blight", "confidence": 0.85}
        ])
        self.assertEqual(boxed.size, (400, 300))

    def test_detection_metrics_computation(self):
        preds = [[
            {"bbox": [10.0, 10.0, 50.0, 50.0], "class_id": 0, "confidence": 0.9},
            {"bbox": [100.0, 100.0, 140.0, 140.0], "class_id": 1, "confidence": 0.7},
        ]]
        gts = [{
            "boxes": np.array([[10.0, 10.0, 50.0, 50.0], [100.0, 100.0, 140.0, 140.0]], dtype=np.float32),
            "labels": np.array([0, 1], dtype=np.int64),
        }]
        metrics = compute_detection_metrics(preds, gts, num_classes=6)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["mAP50"], 1.0)
        self.assertEqual(metrics["mAP50_95"], 1.0)

    def test_cvat_polygon_conversion_isolated_temp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            img_dir = tmp / "src_imgs"
            img_dir.mkdir()
            sample_img = img_dir / "synthetic_pilot_unit_test.jpg"
            Image.new("RGB", (100, 100), color=(40, 130, 45)).save(sample_img)

            xml_content = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <image id="0" name="synthetic_pilot_unit_test.jpg" width="100" height="100">
    <polygon label="Leaf" points="10,10;90,10;90,90;10,90" z_order="0"/>
    <polygon label="Diseased_Region" points="20,20;50,20;50,50;20,50" z_order="1">
      <attribute name="disease_type">Bacterial Blight</attribute>
    </polygon>
    <polygon label="Diseased_Region" points="0,0;5,0;5,5;0,5" z_order="1">
      <attribute name="disease_type">Bacterial Blight</attribute>
    </polygon>
  </image>
</annotations>"""
            xml_path = tmp / "annotations.xml"
            xml_path.write_text(xml_content, encoding="utf-8")

            seg_out = tmp / "seg"
            det_out = tmp / "det"
            qa_out = tmp / "qa"

            report = process_cvat_xml_export(
                xml_path=xml_path,
                image_search_dirs=[img_dir],
                seg_out_dir=seg_out,
                det_out_dir=det_out,
                qa_out_dir=qa_out,
                report_output_path=tmp / "cvat_report.json",
            )

            self.assertEqual(report["processed_valid_count"], 1)
            self.assertEqual(report["skipped_or_invalid_count"], 0)
            rec = report["records"][0]
            self.assertGreater(rec["leaf_pixels"], 6000)
            self.assertGreater(rec["disease_pixels"], 800)
            self.assertLess(rec["disease_pixels"], 1200)  # outside-leaf 0..5 polygon clipped cleanly!
            self.assertEqual(rec["healthy_pixels"] + rec["disease_pixels"], rec["leaf_pixels"])

            mask_path = seg_out / "masks" / "synthetic_pilot_unit_test.png"
            lbl_path = det_out / "labels" / "synthetic_pilot_unit_test.txt"
            self.assertTrue(mask_path.is_file())
            self.assertTrue(lbl_path.is_file())
            self.assertTrue((qa_out / "synthetic_pilot_unit_test_qa.png").is_file())

            mask_arr = np.array(Image.open(mask_path))
            self.assertEqual(mask_arr[0, 0], 0)  # outside leaf polygon must stay background (0) even if diseased polygon overlapped
            self.assertEqual(mask_arr[30, 30], 2)  # diseased inside leaf
            self.assertEqual(mask_arr[70, 70], 1)  # healthy inside leaf


if __name__ == "__main__":
    unittest.main()
