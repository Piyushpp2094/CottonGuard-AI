"""
Automated PDF Diagnostic Report Generator for Cotton Leaf AI Backend.

Produces publication-grade agronomic reports embedding:
  - Classification diagnosis, confidence, and top class probabilities
  - Segmentation pixel counts & affected area percentage (when available)
  - YOLO lesion localization count & annotated image (when available)
  - Grad-CAM visual saliency comparison (with clear interpretability notice)
  - Official ICAR-CICR agronomic decision-support guidance and disclaimers
"""

import json
from pathlib import Path
from typing import List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image as ReportLabImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.app.config import get_settings
from backend.app.models import Prediction
from src.utils.paths import resolve_platform_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_artifact_file(url_or_path: Optional[str], fallback_subdir: str = "predictions", fallback_name: Optional[str] = None) -> Optional[Path]:
    """Resolves a stored Windows/POSIX/URL asset path safely inside active artifacts dir or PROJECT_ROOT."""
    artifacts_root = get_settings().resolve_artifacts_dir()
    if url_or_path:
        cleaned = url_or_path.strip()
        if cleaned.startswith("/artifacts/"):
            rel_sub = cleaned[len("/artifacts/"):]
            for base in (artifacts_root, PROJECT_ROOT / "artifacts"):
                candidate = base / rel_sub
                if candidate.is_file():
                    return candidate
        resolved = resolve_platform_path(cleaned)
        if resolved.is_file():
            return resolved
    if fallback_name:
        for base in (artifacts_root, PROJECT_ROOT / "artifacts"):
            cand = base / fallback_subdir / Path(fallback_name).name
            if cand.is_file():
                return cand
    return None


def generate_pdf_report(prediction: Prediction, output_pdf_path: Path) -> Path:
    """
    Constructs a comprehensive diagnostic PDF report for a Cotton Leaf AI analysis.
    """
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=19,
        leading=23,
        textColor=colors.HexColor("#15803d"),
        alignment=1,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#475569"),
        alignment=1,
        spaceAfter=14,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor("#1e293b"),
    )
    bullet_style = ParagraphStyle(
        "BulletDark",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#334155"),
        leftIndent=14,
        spaceAfter=2.5,
    )
    warning_style = ParagraphStyle(
        "WarningBox",
        parent=styles["Normal"],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#991b1b"),
        spaceBefore=4,
        spaceAfter=4,
    )

    story: List = []

    story.append(Paragraph("Cotton Leaf Disease Analysis & Severity Diagnostic Report", title_style))
    story.append(Paragraph("Leaf Disease Analysis and Severity Estimation Using U-Net and YOLO-Based Detection", subtitle_style))

    # Meta Table
    date_str = prediction.created_at.strftime("%B %d, %Y - %H:%M:%S UTC")
    meta_data = [
        [Paragraph(f"<b>Prediction ID:</b> #{prediction.id}", body_style), Paragraph(f"<b>Timestamp:</b> {date_str}", body_style)],
        [Paragraph(f"<b>Classifier:</b> {prediction.model_version}", body_style), Paragraph(f"<b>Status:</b> {prediction.status}", body_style)],
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Metrics Table
    story.append(Paragraph("1. AI Evaluation Metrics & Severity Summary", section_heading))
    seg = prediction.segmentation
    det = prediction.detection
    sev = prediction.severity
    res = prediction.result

    affected_str = (
        f"{prediction.affected_area_percentage:.2f}%"
        if prediction.affected_area_percentage is not None
        else "Unavailable (Requires trained U-Net mask checkpoint)"
    )
    pixel_breakdown_str = (
        f"Leaf: {seg.total_leaf_pixels:,} px | Healthy: {seg.healthy_pixels:,} px | Diseased: {seg.disease_pixels:,} px"
        if seg and seg.total_leaf_pixels is not None
        else "Unavailable (Awaiting CVAT ground-truth mask training)"
    )
    det_str = (
        f"{det.detection_count} lesion box(es) detected ({det.model_version})"
        if det and det.status == "SUCCESS"
        else "Unavailable (YOLO checkpoint not trained yet)"
    )

    results_data = [
        [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Measured Value</b>", body_style)],
        [Paragraph("Predicted Condition (EfficientNet-B0)", body_style), Paragraph(f"<b>{prediction.predicted_class}</b>", body_style)],
        [Paragraph("Classification Confidence", body_style), Paragraph(f"{prediction.confidence * 100:.2f}% (distinct from area %)", body_style)],
        [Paragraph("YOLO Region Detection", body_style), Paragraph(det_str, body_style)],
        [Paragraph("2D Affected Foliar Area %", body_style), Paragraph(affected_str, body_style)],
        [Paragraph("Pixel Breakdown", body_style), Paragraph(pixel_breakdown_str, body_style)],
        [Paragraph("Estimated Severity Category", body_style), Paragraph(sev.severity_label or prediction.severity_level if sev else prediction.severity_level, body_style)],
    ]
    results_table = Table(results_data, colWidths=[195, 345])
    results_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dcfce7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#14532d")),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 10))

    # Visual Imagery Section
    orig_file = _resolve_artifact_file(prediction.image_path, "predictions", prediction.image_filename)
    comp_file = _resolve_artifact_file(res.gradcam_comparison_path if res else None)
    seg_file = _resolve_artifact_file((seg.overlay_path if seg else None) or (res.overlay_path if res else None))
    det_file = _resolve_artifact_file(det.annotated_image_path if det else None)

    if any([orig_file, comp_file, seg_file, det_file]):
        story.append(Paragraph("2. Diagnostic Imagery & Visual Explainability", section_heading))

        if orig_file:
            try:
                story.append(Paragraph("<b>Original Uploaded Leaf Photograph:</b>", body_style))
                story.append(Spacer(1, 4))
                story.append(ReportLabImage(str(orig_file), width=180, height=180, kind="proportional"))
                story.append(Spacer(1, 8))
            except Exception:
                pass

        if comp_file:
            try:
                story.append(Paragraph("<b>Grad-CAM Explainability Overview (Model Attention Aid Only — Not Lesion Segmentation):</b>", body_style))
                story.append(Spacer(1, 4))
                story.append(ReportLabImage(str(comp_file), width=510, height=220, kind="proportional"))
                story.append(Spacer(1, 8))
            except Exception:
                pass

        if seg_file:
            try:
                story.append(Paragraph("<b>U-Net 3-Class Tissue Segmentation Overlay:</b>", body_style))
                story.append(Spacer(1, 4))
                story.append(ReportLabImage(str(seg_file), width=190, height=190, kind="proportional"))
                story.append(Spacer(1, 8))
            except Exception:
                pass

        if det_file:
            try:
                story.append(Paragraph("<b>YOLO Lesion Localization Boxes:</b>", body_style))
                story.append(Spacer(1, 4))
                story.append(ReportLabImage(str(det_file), width=190, height=190, kind="proportional"))
                story.append(Spacer(1, 8))
            except Exception:
                pass

    # Recommendations
    rec_json_str = res.recommendations_json if res and res.recommendations_json else "{}"
    try:
        rec_data = json.loads(rec_json_str)
    except Exception:
        rec_data = {}

    story.append(Paragraph("3. Agronomic Decision-Support Guidance (ICAR-CICR Aligned)", section_heading))
    if rec_data.get("immediate_first_steps"):
        story.append(Paragraph("<b>Recommended Field Scouting & Cultural Actions:</b>", body_style))
        for step in rec_data["immediate_first_steps"]:
            story.append(Paragraph(f"• {step}", bullet_style))

    if rec_data.get("preventive_management"):
        story.append(Spacer(1, 4))
        story.append(Paragraph("<b>Preventive Management Practices:</b>", body_style))
        for prev in rec_data["preventive_management"]:
            story.append(Paragraph(f"• {prev}", bullet_style))

    if rec_data.get("critical_warning"):
        story.append(Spacer(1, 5))
        story.append(Paragraph(f"<b>IMPORTANT SAFETY WARNING:</b> {rec_data['critical_warning']}", warning_style))

    story.append(Spacer(1, 10))
    disclaimer_text = (
        "<b>SCIENTIFIC & LEGAL DISCLAIMER:</b> Grad-CAM visualizations explain classifier feature saliency and do not represent "
        "pixel-accurate lesion boundaries or physical disease severity. Affected area percentages reflect 2D camera-plane "
        "visible pixels rather than physical cm² area. Always confirm Economic Threshold Levels (ETL) and consult a certified "
        "agronomist or Krishi Vigyan Kendra (KVK) before applying crop protection chemicals."
    )
    story.append(Paragraph(disclaimer_text, ParagraphStyle("Disc", parent=styles["Normal"], fontSize=7.5, leading=10, textColor=colors.HexColor("#64748b"))))

    doc.build(story)
    return output_pdf_path
