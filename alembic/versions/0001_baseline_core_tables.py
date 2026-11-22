"""baseline_core_tables

Revision ID: 0001_baseline
Revises:
Create Date: 2026-11-12 13:05:00
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0001_baseline"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = set(sa.inspect(bind).get_table_names())

    if "users" not in existing:
        op.create_table(
            "users",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("username", sa.String(length=50), nullable=False),
            sa.Column("email", sa.String(length=100), nullable=False),
            sa.Column("password_hash", sa.String(length=255), nullable=False),
            sa.Column("role", sa.String(length=20), nullable=False, server_default="farmer"),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("1")),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_users_id", "users", ["id"], unique=False)
        op.create_index("ix_users_username", "users", ["username"], unique=True)
        op.create_index("ix_users_email", "users", ["email"], unique=True)

    if "predictions" not in existing:
        op.create_table(
            "predictions",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("image_filename", sa.String(length=255), nullable=False),
            sa.Column("image_path", sa.String(length=500), nullable=False),
            sa.Column("predicted_class", sa.String(length=100), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=False),
            sa.Column("affected_area_percentage", sa.Float(), nullable=True),
            sa.Column("severity_level", sa.String(length=50), nullable=False),
            sa.Column("model_version", sa.String(length=50), nullable=False),
            sa.Column("status", sa.String(length=50), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_predictions_id", "predictions", ["id"], unique=False)
        op.create_index("ix_predictions_user_id", "predictions", ["user_id"], unique=False)

    if "prediction_results" not in existing:
        op.create_table(
            "prediction_results",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("prediction_id", sa.Integer(), nullable=False),
            sa.Column("leaf_mask_path", sa.String(length=500), nullable=True),
            sa.Column("disease_mask_path", sa.String(length=500), nullable=True),
            sa.Column("overlay_path", sa.String(length=500), nullable=True),
            sa.Column("gradcam_heatmap_path", sa.String(length=500), nullable=True),
            sa.Column("gradcam_overlay_path", sa.String(length=500), nullable=True),
            sa.Column("gradcam_comparison_path", sa.String(length=500), nullable=True),
            sa.Column("gradcam_metadata_path", sa.String(length=500), nullable=True),
            sa.Column("all_probabilities_json", sa.Text(), nullable=True),
            sa.Column("latency_ms", sa.Float(), nullable=True),
            sa.Column("recommendations_json", sa.Text(), nullable=False),
            sa.Column("uncertainty_notes", sa.Text(), nullable=True),
            sa.ForeignKeyConstraint(["prediction_id"], ["predictions.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("prediction_id"),
        )
        op.create_index("ix_prediction_results_id", "prediction_results", ["id"], unique=False)

    if "detection_results" not in existing:
        op.create_table(
            "detection_results",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("prediction_id", sa.Integer(), nullable=False),
            sa.Column("detections_json", sa.Text(), nullable=False),
            sa.Column("detection_count", sa.Integer(), nullable=False),
            sa.Column("annotated_image_path", sa.String(length=500), nullable=True),
            sa.Column("model_version", sa.String(length=80), nullable=False),
            sa.Column("status", sa.String(length=50), nullable=False),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["prediction_id"], ["predictions.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("prediction_id"),
        )
        op.create_index("ix_detection_results_id", "detection_results", ["id"], unique=False)

    if "segmentation_results" not in existing:
        op.create_table(
            "segmentation_results",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("prediction_id", sa.Integer(), nullable=False),
            sa.Column("leaf_mask_path", sa.String(length=500), nullable=True),
            sa.Column("disease_mask_path", sa.String(length=500), nullable=True),
            sa.Column("overlay_path", sa.String(length=500), nullable=True),
            sa.Column("total_leaf_pixels", sa.Integer(), nullable=True),
            sa.Column("disease_pixels", sa.Integer(), nullable=True),
            sa.Column("healthy_pixels", sa.Integer(), nullable=True),
            sa.Column("affected_area_percentage", sa.Float(), nullable=True),
            sa.Column("model_version", sa.String(length=80), nullable=False),
            sa.Column("status", sa.String(length=50), nullable=False),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["prediction_id"], ["predictions.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("prediction_id"),
        )
        op.create_index("ix_segmentation_results_id", "segmentation_results", ["id"], unique=False)

    if "severity_results" not in existing:
        op.create_table(
            "severity_results",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("prediction_id", sa.Integer(), nullable=False),
            sa.Column("severity_level", sa.String(length=50), nullable=False),
            sa.Column("severity_label", sa.String(length=200), nullable=True),
            sa.Column("affected_area_percentage", sa.Float(), nullable=True),
            sa.Column("is_available", sa.Boolean(), nullable=False),
            sa.Column("thresholds_verified", sa.Boolean(), nullable=False),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["prediction_id"], ["predictions.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("prediction_id"),
        )
        op.create_index("ix_severity_results_id", "severity_results", ["id"], unique=False)

    if "reports" not in existing:
        op.create_table(
            "reports",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("prediction_id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("pdf_filename", sa.String(length=255), nullable=False),
            sa.Column("pdf_path", sa.String(length=500), nullable=False),
            sa.Column("file_size_bytes", sa.Integer(), nullable=False),
            sa.Column("generated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["prediction_id"], ["predictions.id"]),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_reports_id", "reports", ["id"], unique=False)
        op.create_index("ix_reports_prediction_id", "reports", ["prediction_id"], unique=False)
        op.create_index("ix_reports_user_id", "reports", ["user_id"], unique=False)

    if "audit_logs" not in existing:
        op.create_table(
            "audit_logs",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=True),
            sa.Column("username", sa.String(length=50), nullable=False),
            sa.Column("action", sa.String(length=60), nullable=False),
            sa.Column("resource", sa.String(length=120), nullable=False),
            sa.Column("details", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_audit_logs_id", "audit_logs", ["id"], unique=False)
        op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"], unique=False)
        op.create_index("ix_audit_logs_action", "audit_logs", ["action"], unique=False)
        op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"], unique=False)


def downgrade() -> None:
    pass
