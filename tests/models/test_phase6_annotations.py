"""
Phase 6 Annotation Quality Validator, Zero-Leakage Pilot & Gate Tests.

Tests against temporary directories and synthetic test masks/labels without touching
real CVAT containers, cotton_leaf_ai.db, dataset splits, or model checkpoints.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_annotation_pilot import generate_annotation_pilot
from scripts.prepare_cvat_annotations import process_cvat_xml_export
from scripts.validate_annotations import PILOT_PAIR_THRESHOLD, RELIABLE_TRAINING_MIN_PAIRS, validate_annotation_directory
from scripts.validate_phase6_annotations import run_phase6_quality_validation


class TestPhase6Annotations(unittest.TestCase):
    def test_01_pilot_generation_stratified_and_zero_leakage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_csv = tmp / "pilot_manifest.csv"
            summary_json = tmp / "pilot_summary.json"

            summary = generate_annotation_pilot(
                splits_dir=PROJECT_ROOT / "artifacts" / "manifests" / "splits",
                pilot_dir=tmp / "pilot_batch",
                manifest_csv_path=manifest_csv,
                summary_json_path=summary_json,
                train_per_class=2,
                val_per_class=1,
                seed=42,
                copy_images=False,
            )

            self.assertEqual(summary["total_pilot_images"], 21)
            self.assertEqual(summary["split_counts"], {"train": 14, "validation": 7, "test": 0})
            self.assertEqual(summary["test_split_overlap_count"], 0)
            self.assertEqual(summary["augmented_split_overlap_count"], 0)
            self.assertEqual(summary["unique_group_count"], 21)
            self.assertEqual(len(summary["per_class_counts"]), 7)
            for count in summary["per_class_counts"].values():
                self.assertEqual(count, 3)

    def test_02_segmentation_validator_rejects_empty_invalid_and_ten_pair_gate(self):
        self.assertEqual(PILOT_PAIR_THRESHOLD, 21)
        self.assertGreaterEqual(RELIABLE_TRAINING_MIN_PAIRS, 50)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            imgs = tmp / "images"
            masks = tmp / "masks"
            imgs.mkdir()
            masks.mkdir()

            # Valid pair
            Image.new("RGB", (200, 200), (40, 120, 40)).save(imgs / "good_leaf.jpg")
            valid_mask = np.zeros((200, 200), dtype=np.uint8)
            valid_mask[20:180, 20:180] = 1
            valid_mask[60:90, 60:90] = 2
            Image.fromarray(valid_mask, mode="L").save(masks / "good_leaf.png")

            # Empty mask (all zeros)
            Image.new("RGB", (200, 200), (40, 120, 40)).save(imgs / "empty_leaf.jpg")
            Image.fromarray(np.zeros((200, 200), dtype=np.uint8), mode="L").save(masks / "empty_leaf.png")

            # Invalid class value (3) + mismatch dimensions
            Image.new("RGB", (200, 200), (40, 120, 40)).save(imgs / "bad_class_leaf.jpg")
            bad_mask = np.ones((200, 200), dtype=np.uint8) * 3
            Image.fromarray(bad_mask, mode="L").save(masks / "bad_class_leaf.png")

            # Missing mask
            Image.new("RGB", (200, 200), (40, 120, 40)).save(imgs / "orphan_leaf.jpg")

            summary = validate_annotation_directory(imgs, masks, tmp / "summary.json")
            self.assertEqual(summary["valid_pairs_count"], 1)
            self.assertEqual(summary["invalid_pairs_count"], 2)
            self.assertEqual(summary["empty_masks_count"], 1)
            self.assertEqual(summary["missing_masks_count"], 1)
            self.assertFalse(summary["is_ready_for_segmentation_training"])

    def test_03_yolo_and_unified_phase6_validator_catches_box_and_class_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            seg_dir = tmp / "seg"
            det_dir = tmp / "det"
            (seg_dir / "images").mkdir(parents=True)
            (seg_dir / "masks").mkdir(parents=True)
            (det_dir / "images").mkdir(parents=True)
            (det_dir / "labels").mkdir(parents=True)

            # Valid YOLO image + label
            Image.new("RGB", (400, 400), (50, 130, 50)).save(det_dir / "images" / "valid_bb.jpg")
            (det_dir / "labels" / "valid_bb.txt").write_text("0 0.500000 0.500000 0.250000 0.250000\n", encoding="utf-8")

            # Invalid YOLO class ID (class 9) + out-of-bounds box
            Image.new("RGB", (400, 400), (50, 130, 50)).save(det_dir / "images" / "invalid_box.jpg")
            (det_dir / "labels" / "invalid_box.txt").write_text("9 0.95 0.95 0.50 0.50\n", encoding="utf-8")

            # Missing label
            Image.new("RGB", (400, 400), (50, 130, 50)).save(det_dir / "images" / "missing_lbl.jpg")

            report = run_phase6_quality_validation(
                seg_dir=seg_dir,
                det_dir=det_dir,
                splits_dir=PROJECT_ROOT / "artifacts" / "manifests" / "splits",
                near_dup_csv=PROJECT_ROOT / "artifacts" / "audit" / "original" / "near_duplicates.csv",
                output_path=tmp / "phase6_quality.json",
                min_reliable_pairs=50,
            )

            yolo = report["yolo_detection_validation"]
            self.assertEqual(yolo["valid_labeled_images"], 1)
            self.assertEqual(yolo["valid_bounding_boxes"], 1)
            self.assertEqual(yolo["invalid_records_count"], 1)
            self.assertEqual(yolo["missing_labels_count"], 1)
            self.assertFalse(yolo["is_ready_for_yolo_training"])
            self.assertEqual(report["overall_status"]["unet_status"], "BLOCKED_INSUFFICIENT_ANNOTATIONS")
            self.assertEqual(report["overall_status"]["yolo_status"], "BLOCKED_INSUFFICIENT_ANNOTATIONS")

    def test_04_rejects_xml_tracks_and_cross_image_polygon_duplication(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            imgs = tmp / "imgs"
            imgs.mkdir()
            Image.new("RGB", (200, 200), (40, 120, 40)).save(imgs / "BBC00130.jpg")
            Image.new("RGB", (200, 200), (40, 120, 40)).save(imgs / "BBC00202.jpg")
            Image.new("RGB", (200, 200), (40, 120, 40)).save(imgs / "HL00031.jpg")

            seg_out, det_out, qa_out = tmp / "seg", tmp / "det", tmp / "qa"

            # 1. Track XML structure must be rejected without writing files
            track_xml = tmp / "tracks.xml"
            track_xml.write_text(
                '<?xml version="1.0" encoding="utf-8"?><annotations><track id="0" label="Leaf">'
                '<polygon frame="0" points="10,10;180,10;180,180;10,180" occluded="0" outside="0"/>'
                "</track></annotations>",
                encoding="utf-8",
            )
            rep_track = process_cvat_xml_export(
                track_xml, [imgs], seg_out, det_out, qa_out, report_output_path=tmp / "rep_track.json"
            )
            self.assertEqual(rep_track["status"], "REJECTED")
            self.assertEqual(rep_track["error_code"], "UNSUPPORTED_TRACK_STRUCTURE")
            self.assertFalse(rep_track["published_to_workspace"])

            # 2. Task #3 failure scenario: identical polygons copied onto BBC00130.jpg and BBC00202.jpg
            dup_xml = tmp / "dup_polys.xml"
            same_polys = (
                '<polygon label="Leaf" points="10,10;180,10;180,180;10,180" z_order="0"/>'
                '<polygon label="Diseased_Region" points="40,40;90,40;90,90;40,90" z_order="0">'
                '<attribute name="disease_type">Bacterial Blight</attribute></polygon>'
            )
            dup_xml.write_text(
                f'<?xml version="1.0" encoding="utf-8"?><annotations>'
                f'<image id="0" name="BBC00130.jpg" width="200" height="200">{same_polys}</image>'
                f'<image id="1" name="BBC00202.jpg" width="200" height="200">{same_polys}</image>'
                f'<image id="2" name="HL00031.jpg" width="200" height="200"/>'
                f"</annotations>",
                encoding="utf-8",
            )
            rep_dup = process_cvat_xml_export(
                dup_xml, [imgs], seg_out, det_out, qa_out, report_output_path=tmp / "rep_dup.json"
            )
            self.assertEqual(rep_dup["status"], "REVIEW_REQUIRED")
            self.assertTrue(rep_dup["suspicious_cross_image_duplication"])
            self.assertFalse(rep_dup["published_to_workspace"])
            self.assertFalse((seg_out / "masks" / "BBC00130.png").exists())

            # Explicit review override allows confirmed identical polygons cleanly, leaving unannotated HL00031.jpg untouched
            rep_dup_ok = process_cvat_xml_export(
                dup_xml,
                [imgs],
                seg_out,
                det_out,
                qa_out,
                report_output_path=tmp / "rep_dup_ok.json",
                allow_identical_signatures=True,
                publish=True,
            )
            self.assertEqual(rep_dup_ok["status"], "PUBLISHED")
            self.assertEqual(rep_dup_ok["processed_valid_count"], 2)
            self.assertEqual(rep_dup_ok["unannotated_images"], ["HL00031.jpg"])
            self.assertFalse((seg_out / "masks" / "HL00031.png").exists())

    def test_05_rejects_out_of_leaf_and_wrong_disease_type_atomically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            imgs = tmp / "imgs"
            imgs.mkdir()
            Image.new("RGB", (200, 200), (40, 120, 40)).save(imgs / "BBC00130.jpg")
            Image.new("RGB", (200, 200), (40, 120, 40)).save(imgs / "CV00248.jpg")

            seg_out, det_out, qa_out = tmp / "seg", tmp / "det", tmp / "qa"

            # Valid BBC00130.jpg paired with broken CV00248.jpg (wrong disease_type + out-of-leaf polygon) -> nothing published atomically
            mixed_xml = tmp / "mixed.xml"
            mixed_xml.write_text(
                '<?xml version="1.0" encoding="utf-8"?><annotations>'
                '<image id="0" name="BBC00130.jpg" width="200" height="200">'
                '<polygon label="Leaf" points="10,10;180,10;180,180;10,180"/>'
                '<polygon label="Diseased_Region" points="40,40;90,40;90,90;40,90"><attribute name="disease_type">Bacterial Blight</attribute></polygon>'
                "</image>"
                '<image id="1" name="CV00248.jpg" width="200" height="200">'
                '<polygon label="Leaf" points="50,50;150,50;150,150;50,150"/>'
                '<polygon label="Diseased_Region" points="0,0;40,0;40,40;0,40"><attribute name="disease_type">Bacterial Blight</attribute></polygon>'
                "</image></annotations>",
                encoding="utf-8",
            )
            rep = process_cvat_xml_export(
                mixed_xml,
                [imgs],
                seg_out,
                det_out,
                qa_out,
                report_output_path=tmp / "rep_mixed.json",
                pilot_manifest_path=PROJECT_ROOT / "artifacts" / "manifests" / "pilot_batch_manifest.csv",
                strict_disease_match=True,
                strict_leaf_containment=True,
                out_of_leaf_tolerance_ratio=0.02,
                publish=True,
            )
            self.assertEqual(rep["status"], "BLOCKED_VALIDATION_ERRORS")
            self.assertFalse(rep["published_to_workspace"])
            # Atomic staging guarantees valid image 0 was also not half-published on batch error
            self.assertFalse((seg_out / "masks" / "BBC00130.png").exists())


if __name__ == "__main__":
    unittest.main()
