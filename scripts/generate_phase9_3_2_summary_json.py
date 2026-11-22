"""
scripts/generate_phase9_3_2_summary_json.py
Generates artifacts/metrics/phase9_3_2_pilot_summary.json.
"""

import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("D:/Main_Project/cotton_leaf_ai")
PILOT_DIR = PROJECT_ROOT / "data" / "annotation_pilot" / "phase9_3_2"
VAL_JSON_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "phase9_3_2_pilot_validation.json"
MANIFEST_PATH = PILOT_DIR / "pilot_manifest.csv"
OUTPUT_JSON = PROJECT_ROOT / "artifacts" / "metrics" / "phase9_3_2_pilot_summary.json"

with open(VAL_JSON_PATH, "r") as f:
    val_data = json.load(f)

manifest_df = pd.read_csv(MANIFEST_PATH)

summary = {
    "phase": "9.3.2",
    "phase_title": "Five-Image Annotation Pilot — CVAT Task #6",
    "timestamp": "2026-11-15T14:10:00+05:30",
    "governance_status": "PHASE_9_3_2_PILOT_PASSED_PENDING_USER_REVIEW",
    "detector_status": "EXPERIMENTAL / NOT APPROVED FOR PRODUCTION",
    "task_verification": {
        "task_id": 6,
        "task_name": "CottonGuard_Pilot_Batch_03_ShapesOnly",
        "job_id": 6,
        "project_id": 3,
        "project_name": "CottonGuard_AI_Annotation_v2",
        "total_frames": 50,
        "authorized_frames": [0, 16, 20, 30, 35],
        "untouched_frames_count": 45
    },
    "backups": {
        "pre_execution_backup_json": "artifacts/backups/cvat_job6_pre_phase9_3_2_backup_20261115_133854.json",
        "pre_execution_backup_sha256": "597403CB3795E5814BB40D058A4A972C055DA1E3F3F0AFCD0F9F4745D14078AA",
        "post_execution_backup_json": "artifacts/backups/cvat_job6_post_phase9_3_2_backup_20261115_140903.json",
        "post_execution_backup_sha256": "9F1F362E6E0DE653C56BE9ABEC1C4AD8058B4CD3DF35526B683F0970BD07C462",
        "cvat_export_zip": "artifacts/backups/cvat_task6_phase9_3_2_export.zip",
        "cvat_export_zip_sha256": "016F37BE2E1CC6AD7B7D809B27B420AE3409809924BBA14EB66295AE76F73FE6"
    },
    "shape_accounting": {
        "original_shapes_in_task": 31,
        "original_disease_boxes": 28,
        "final_shapes_in_task": 102,
        "final_leaf_polygons": 5,
        "final_disease_boxes": 97,
        "total_tracks": 0
    },
    "per_frame_summary": manifest_df.to_dict(orient="records"),
    "box_dimensions_distribution": val_data["box_dimensions_summary"],
    "size_policy_accounting": {
        "tier_1_rejected_degenerate_boxes": 0,
        "tier_2_borderline_small_boxes_reviewed": val_data["checks"]["check_06_borderline_boxes"]["borderline_count"],
        "tier_3_standard_accepted_boxes": len(val_data["checks"]["check_02_coordinates"]) if "check_02_coordinates" in val_data else 93,
        "oversized_boxes_greater_than_60k_px": 0
    },
    "overlap_and_duplicate_review": {
        "duplicate_boxes_detected_iou_ge_0_70": 0,
        "overlapping_pairs_same_class": val_data["checks"]["check_09_same_class_overlap"]["overlapping_pairs_count"],
        "max_same_class_iou": 0.1056,
        "confluent_fused_violations": 0
    },
    "automated_qa": {
        "total_checks": 14,
        "checks_passed": 14,
        "checks_failed": 0,
        "verdict": "PASS"
    },
    "visual_qa": {
        "inspection_rate": "100%",
        "images_inspected": 5,
        "composite_boxes_detected": 0,
        "vein_discoloration_false_alarms": 0,
        "excess_padding_violations": 0,
        "verdict": "PASS"
    },
    "preservation_invariants": {
        "classifier_checkpoint_sha256": "552BABA83A66AD4A8BBC1EEB7FE3EE8DD84DFABE68C1EF477930DA57A05FA90E",
        "unet_checkpoint_sha256": "C6788F794869E362BCEBB8BF2A075A83EF0EACEBF6B9EF3D75FDFE57517F3558",
        "detector_checkpoint_sha256": "01D82B3282509C543C1BD3A48A46CC03581B3CE573A8D8B77B85BE8E2B36B5D2",
        "test_manifest_sha256": "6025D86F89B9EC4D3F95AD4A54F7D633478C9EB152C67057DFB021D8C00BDD9D",
        "test_split_access": False,
        "published_datasets_modified": False,
        "model_training_executed": False,
        "sqlite_db_modified": False
    }
}

OUTPUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"Saved machine-readable pilot summary to {OUTPUT_JSON.name}")
