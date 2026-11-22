"""
CVAT Annotation Export Ingestion, Mask Conversion & YOLO Dataset Builder (Phase 6 Hardened).

Ingests an authentic exported CVAT annotation archive (.zip or extracted folder)
in CVAT for images 1.1 XML format (annotations.xml + images) with strict anti-corruption
safeguards before publishing any output:
  1. Rejects unsupported <track> elements immediately (tracks cause cross-frame leakage).
  2. Detects identical non-empty polygon signatures repeated across distinct images and
     requires explicit human review (--allow-identical-signatures) rather than silently
     publishing duplicated frames.
  3. Validates image names and expected disease classes against pilot_batch_manifest.csv
     and train/validation splits; validates zero overlap with test.csv.
  4. Never silently discards invalid polygons/boxes (missing disease_type, wrong class,
     <3 vertices, empty Leaf mask, or Diseased_Region outside Leaf beyond tolerance).
  5. Separates intentionally unannotated frames (0 shapes total -> skipped cleanly without
     writing outputs, supporting partial 2-image smoke tests).
  6. Stages all generated 3-class masks, YOLO labels, and QA images inside a temporary
     directory and publishes them ONLY after all validation checks pass cleanly.

Strictly refuses to fabricate annotations or synthetic masks.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.area_calculation import generate_segmentation_overlays
from src.models.detector import load_detection_classes


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_points_string(points_str: str) -> np.ndarray:
    """Parses CVAT XML 'x1,y1;x2,y2;...' polygon points string into (N, 2) int32 array."""
    pts: List[List[float]] = []
    for pair in points_str.strip().split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x_str, y_str = pair.split(",")
        pts.append([float(x_str), float(y_str)])
    if len(pts) < 3:
        return np.zeros((0, 2), dtype=np.int32)
    return np.round(np.array(pts, dtype=np.float32)).astype(np.int32)


def _find_image_in_sources(filename: str, search_dirs: List[Path]) -> Optional[Path]:
    """Safely locates a source image by exact relative path or stem/basename without modifying dataset."""
    base = Path(filename).name
    for root in search_dirs:
        if not root.is_dir():
            continue
        direct = root / filename
        if direct.is_file():
            return direct
        matches = list(root.rglob(base))
        if matches:
            return matches[0]
    return None


def _load_test_leakage_lookup() -> Tuple[set, set]:
    """Loads test split SHA-256 hashes and stems to ensure annotations never leak test samples into train."""
    test_csv = PROJECT_ROOT / "artifacts" / "manifests" / "splits" / "test.csv"
    hashes: set = set()
    names: set = set()
    if test_csv.is_file():
        df = pd.read_csv(test_csv)
        if "sha256" in df.columns:
            hashes.update(df["sha256"].dropna().astype(str).tolist())
        if "relative_path" in df.columns:
            names.update(Path(p).name for p in df["relative_path"].dropna().astype(str))
    return hashes, names


def _load_expected_class_lookup(pilot_manifest_path: Optional[Path] = None) -> Dict[str, Dict[str, str]]:
    """Loads expected class_label & split metadata from pilot_batch_manifest.csv and train/val splits."""
    lookup: Dict[str, Dict[str, str]] = {}
    splits_dir = PROJECT_ROOT / "artifacts" / "manifests" / "splits"
    for split_name in ("train", "validation"):
        sp = splits_dir / f"{split_name}.csv"
        if sp.is_file():
            df = pd.read_csv(sp)
            for _, row in df.iterrows():
                fname = Path(str(row.get("relative_path", ""))).name
                if fname:
                    lookup[fname] = {
                        "class_label": str(row.get("class_folder_label", "")),
                        "split": split_name,
                        "source": sp.name,
                        "in_pilot_manifest": False,
                    }

    if pilot_manifest_path and pilot_manifest_path.is_file():
        df_p = pd.read_csv(pilot_manifest_path)
        for _, row in df_p.iterrows():
            fname = str(row.get("filename", "")).strip()
            if fname:
                lookup[fname] = {
                    "class_label": str(row.get("class_label", "")),
                    "split": str(row.get("split", "")),
                    "source": pilot_manifest_path.name,
                    "in_pilot_manifest": True,
                }
    return lookup


def _compute_image_polygon_signature(img_elem: ET.Element) -> Optional[str]:
    """
    Computes a deterministic hash over all polygons/boxes inside an <image> element.
    Returns None when the <image> has zero annotations.
    """
    items: List[str] = []
    for poly in img_elem.findall("polygon"):
        label = poly.attrib.get("label", "").strip()
        pts = poly.attrib.get("points", "").strip()
        attrs = sorted(
            f"{a.attrib.get('name', '').strip()}={(a.text or '').strip()}"
            for a in poly.findall("attribute")
        )
        items.append(f"poly|{label}|{'&'.join(attrs)}|{pts}")
    for box in img_elem.findall("box"):
        label = box.attrib.get("label", "").strip()
        coords = ",".join(box.attrib.get(k, "") for k in ("xtl", "ytl", "xbr", "ybr"))
        attrs = sorted(
            f"{a.attrib.get('name', '').strip()}={(a.text or '').strip()}"
            for a in box.findall("attribute")
        )
        items.append(f"box|{label}|{'&'.join(attrs)}|{coords}")
    if not items:
        return None
    canonical = "\n".join(sorted(items))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def process_cvat_xml_export(
    xml_path: Path,
    image_search_dirs: List[Path],
    seg_out_dir: Path,
    det_out_dir: Path,
    qa_out_dir: Path,
    report_output_path: Optional[Path] = None,
    pilot_manifest_path: Optional[Path] = None,
    strict_manifest: bool = False,
    strict_disease_match: bool = False,
    strict_leaf_containment: bool = False,
    out_of_leaf_tolerance_ratio: float = 0.02,
    allow_identical_signatures: bool = False,
    publish: bool = True,
) -> Dict[str, Any]:
    """
    Processes CVAT for images 1.1 annotations.xml into verified U-Net masks and YOLO labels.

    All checks execute first; outputs are staged in a temporary directory and published
    atomically only if zero blocking errors occur and `publish=True`.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    report_path = report_output_path or (PROJECT_ROOT / "artifacts" / "manifests" / "cvat_annotation_quality_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Reject unsupported XML <track> structures immediately
    track_elems = root.findall(".//track")
    if track_elems:
        rejected = {
            "status": "REJECTED",
            "error_code": "UNSUPPORTED_TRACK_STRUCTURE",
            "message": (
                f"Detected {len(track_elems)} <track> element(s) in {xml_path.name}. "
                "Track structures propagate annotations across frames and are rejected. "
                "Re-annotate using independent Shape objects per image."
            ),
            "track_count": len(track_elems),
            "published_to_workspace": False,
            "processed_valid_count": 0,
            "skipped_or_invalid_count": 0,
            "ready_for_unet_training": False,
            "ready_for_yolo_training": False,
        }
        report_path.write_text(json.dumps(rejected, indent=2), encoding="utf-8")
        return rejected

    # 2. Detect identical non-empty polygon signatures across distinct images
    sig_to_images: Dict[str, List[str]] = defaultdict(list)
    image_elems = root.findall("image")
    for img_elem in image_elems:
        img_name = img_elem.attrib.get("name", "")
        sig = _compute_image_polygon_signature(img_elem)
        if sig is not None:
            sig_to_images[sig].append(img_name)

    identical_groups = {sig: names for sig, names in sig_to_images.items() if len(names) >= 2}
    suspicious_duplication = len(identical_groups) > 0

    if suspicious_duplication and not allow_identical_signatures:
        review_report = {
            "status": "REVIEW_REQUIRED",
            "error_code": "IDENTICAL_CROSS_IMAGE_POLYGONS_DETECTED",
            "message": (
                "Identical non-empty polygon signatures were detected across different images "
                "(often caused by CVAT POLYGON TRACK carryover). Explicit human review is required; "
                "no masks or YOLO labels were published. Pass --allow-identical-signatures only if confirmed genuine."
            ),
            "suspicious_cross_image_duplication": True,
            "identical_polygon_signature_groups": identical_groups,
            "requires_explicit_review": True,
            "published_to_workspace": False,
            "processed_valid_count": 0,
            "skipped_or_invalid_count": sum(len(v) for v in identical_groups.values()),
            "ready_for_unet_training": False,
            "ready_for_yolo_training": False,
        }
        report_path.write_text(json.dumps(review_report, indent=2), encoding="utf-8")
        return review_report

    det_classes = load_detection_classes()
    det_class_to_id = {name.lower(): idx for idx, name in enumerate(det_classes)}
    test_hashes, test_names = _load_test_leakage_lookup()
    expected_lookup = _load_expected_class_lookup(pilot_manifest_path)

    processed_records: List[Dict[str, Any]] = []
    unannotated_images: List[str] = []
    blocking_errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    leakage_flags: List[str] = []
    manifest_mismatches: List[str] = []

    if suspicious_duplication:
        warnings.append({
            "type": "IDENTICAL_POLYGON_SIGNATURES_ACKNOWLEDGED",
            "groups": identical_groups,
        })

    # Stage everything into an isolated temporary directory first
    with tempfile.TemporaryDirectory(prefix="cvat_stage_") as stage_tmp:
        stage_root = Path(stage_tmp)
        stage_seg_imgs = stage_root / "seg" / "images"
        stage_seg_masks = stage_root / "seg" / "masks"
        stage_det_imgs = stage_root / "det" / "images"
        stage_det_lbls = stage_root / "det" / "labels"
        stage_qa = stage_root / "qa"
        for d in (stage_seg_imgs, stage_seg_masks, stage_det_imgs, stage_det_lbls, stage_qa):
            d.mkdir(parents=True, exist_ok=True)

        for img_elem in image_elems:
            img_name = img_elem.attrib.get("name", "")
            width = int(float(img_elem.attrib.get("width", 0)))
            height = int(float(img_elem.attrib.get("height", 0)))

            polys = img_elem.findall("polygon")
            boxes = img_elem.findall("box")
            extra_masks = img_elem.findall("mask")

            if extra_masks:
                blocking_errors.append({
                    "image": img_name,
                    "error": f"Unsupported CVAT <mask> RLE element(s) ({len(extra_masks)}); convert masks to polygons.",
                })
                continue

            # Cleanly skip completely unannotated frames (supporting partial 2-image exports)
            if not polys and not boxes:
                unannotated_images.append(img_name)
                continue

            base_name = Path(img_name).name
            meta = expected_lookup.get(base_name)
            if strict_manifest and (meta is None or not meta.get("in_pilot_manifest", False)):
                manifest_mismatches.append(img_name)
                blocking_errors.append({
                    "image": img_name,
                    "error": f"Image '{base_name}' is missing from expected pilot manifest.",
                })
                continue

            src_path = _find_image_in_sources(img_name, image_search_dirs)
            if src_path is None or not src_path.is_file():
                blocking_errors.append({"image": img_name, "error": "Source image file not found in search paths."})
                continue

            with Image.open(src_path) as pil_img:
                rgb_img = pil_img.convert("RGB")
                actual_w, actual_h = rgb_img.size

            if width > 0 and height > 0 and (actual_w, actual_h) != (width, height):
                blocking_errors.append({
                    "image": img_name,
                    "error": f"Dimension mismatch: XML declares ({width}, {height}), file is ({actual_w}, {actual_h}).",
                })
                continue

            img_hash = compute_sha256(src_path)
            in_test_split = (img_hash in test_hashes) or (base_name in test_names)
            if in_test_split:
                leakage_flags.append(img_name)
                blocking_errors.append({
                    "image": img_name,
                    "error": "Holdout test split leakage: image belongs to artifacts/manifests/splits/test.csv.",
                })
                continue

            expected_class = meta.get("class_label") if meta else None
            leaf_mask = np.zeros((actual_h, actual_w), dtype=np.uint8)
            diseased_mask = np.zeros((actual_h, actual_w), dtype=np.uint8)
            raw_disease_polys: List[Tuple[int, np.ndarray, str]] = []
            yolo_lines: List[str] = []
            disease_types_found: List[str] = []
            img_errors: List[str] = []

            # Pass 1: rasterize Leaf polygons and collect / validate Diseased_Region polygons
            for p_idx, poly in enumerate(polys):
                label = poly.attrib.get("label", "").strip()
                pts = parse_points_string(poly.attrib.get("points", ""))
                if pts.shape[0] < 3:
                    img_errors.append(f"Polygon #{p_idx} ({label}) has fewer than 3 valid vertices.")
                    continue

                if label == "Leaf":
                    cv2.fillPoly(leaf_mask, [pts], 1)
                elif label == "Diseased_Region":
                    d_type = ""
                    for attr in poly.findall("attribute"):
                        if attr.attrib.get("name") == "disease_type":
                            d_type = (attr.text or "").strip()
                    cls_id = det_class_to_id.get(d_type.lower())
                    if cls_id is None:
                        img_errors.append(
                            f"Diseased_Region polygon #{p_idx} has missing or unknown disease_type '{d_type}'."
                        )
                        continue

                    canonical_d_type = det_classes[cls_id]
                    if expected_class == "Healthy Leaf":
                        img_errors.append(
                            f"Healthy Leaf image '{base_name}' contains Diseased_Region polygon #{p_idx}."
                        )
                        continue
                    if strict_disease_match and expected_class and canonical_d_type != expected_class:
                        img_errors.append(
                            f"disease_type mismatch on '{base_name}': expected '{expected_class}', got '{canonical_d_type}'."
                        )
                        continue

                    raw_disease_polys.append((p_idx, pts, canonical_d_type))
                    disease_types_found.append(canonical_d_type)
                    cv2.fillPoly(diseased_mask, [pts], 1)

                    x_min = float(np.clip(pts[:, 0].min(), 0, actual_w))
                    y_min = float(np.clip(pts[:, 1].min(), 0, actual_h))
                    x_max = float(np.clip(pts[:, 0].max(), 0, actual_w))
                    y_max = float(np.clip(pts[:, 1].max(), 0, actual_h))
                    bw = x_max - x_min
                    bh = y_max - y_min
                    if bw < 2.0 or bh < 2.0:
                        img_errors.append(f"Diseased_Region polygon #{p_idx} bounding box too small ({bw:.1f}x{bh:.1f} px).")
                        continue
                    cx = ((x_min + x_max) * 0.5) / actual_w
                    cy = ((y_min + y_max) * 0.5) / actual_h
                    nw = bw / actual_w
                    nh = bh / actual_h
                    yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                else:
                    img_errors.append(f"Unknown CVAT label '{label}' on polygon #{p_idx}.")

            # Optional explicit <box> elements
            for b_idx, box in enumerate(boxes):
                label = box.attrib.get("label", "").strip()
                if label != "Diseased_Region":
                    img_errors.append(f"Unsupported <box> label '{label}' (only Diseased_Region allowed).")
                    continue
                d_type = ""
                for attr in box.findall("attribute"):
                    if attr.attrib.get("name") == "disease_type":
                        d_type = (attr.text or "").strip()
                cls_id = det_class_to_id.get(d_type.lower())
                if cls_id is None:
                    img_errors.append(f"Diseased_Region box #{b_idx} has missing/unknown disease_type '{d_type}'.")
                    continue
                canonical_d_type = det_classes[cls_id]
                if expected_class == "Healthy Leaf":
                    img_errors.append(f"Healthy Leaf image '{base_name}' contains Diseased_Region box #{b_idx}.")
                    continue
                if strict_disease_match and expected_class and canonical_d_type != expected_class:
                    img_errors.append(
                        f"disease_type mismatch on '{base_name}': expected '{expected_class}', got '{canonical_d_type}'."
                    )
                    continue

                xtl = float(box.attrib.get("xtl", 0))
                ytl = float(box.attrib.get("ytl", 0))
                xbr = float(box.attrib.get("xbr", 0))
                ybr = float(box.attrib.get("ybr", 0))
                bw = xbr - xtl
                bh = ybr - ytl
                if bw < 2.0 or bh < 2.0:
                    img_errors.append(f"Diseased_Region box #{b_idx} too small ({bw:.1f}x{bh:.1f} px).")
                    continue
                disease_types_found.append(canonical_d_type)
                cx = ((xtl + xbr) * 0.5) / actual_w
                cy = ((ytl + ybr) * 0.5) / actual_h
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw / actual_w:.6f} {bh / actual_h:.6f}")

            # Verify Leaf polygon presence & Diseased_Region boundary containment
            leaf_area_raw = int(np.sum(leaf_mask == 1))
            if leaf_area_raw == 0:
                img_errors.append("No valid 'Leaf' polygon found; cannot construct verified leaf mask.")

            out_of_leaf_violations: List[Dict[str, Any]] = []
            for p_idx, pts, canonical_d_type in raw_disease_polys:
                poly_canvas = np.zeros((actual_h, actual_w), dtype=np.uint8)
                cv2.fillPoly(poly_canvas, [pts], 1)
                poly_px = int(np.sum(poly_canvas == 1))
                outside_px = int(np.sum((poly_canvas == 1) & (leaf_mask == 0)))
                outside_ratio = float(outside_px / max(1, poly_px))
                if outside_ratio > out_of_leaf_tolerance_ratio:
                    info = {
                        "polygon_index": p_idx,
                        "disease_type": canonical_d_type,
                        "outside_pixels": outside_px,
                        "polygon_pixels": poly_px,
                        "outside_ratio": round(outside_ratio, 4),
                        "tolerance_ratio": out_of_leaf_tolerance_ratio,
                    }
                    out_of_leaf_violations.append(info)
                    if strict_leaf_containment:
                        img_errors.append(
                            f"Diseased_Region polygon #{p_idx} has {outside_px}/{poly_px} px "
                            f"({outside_ratio*100:.1f}%) outside Leaf boundary (tolerance <= {out_of_leaf_tolerance_ratio*100:.1f}%)."
                        )

            if out_of_leaf_violations and not strict_leaf_containment:
                warnings.append({
                    "image": img_name,
                    "type": "OUT_OF_LEAF_POLYGON_WARNING",
                    "violations": out_of_leaf_violations,
                })

            if img_errors:
                blocking_errors.append({"image": img_name, "errors": img_errors})
                continue

            # Derive strict 3-class pixel mask: Healthy = Leaf - Diseased_Region, Diseased clipped to Leaf
            final_mask = np.zeros((actual_h, actual_w), dtype=np.uint8)
            final_mask[leaf_mask == 1] = 1
            final_mask[(leaf_mask == 1) & (diseased_mask == 1)] = 2

            leaf_px = int(np.sum(final_mask >= 1))
            disease_px = int(np.sum(final_mask == 2))
            healthy_px = leaf_px - disease_px

            stem = Path(img_name).stem
            shutil.copy2(src_path, stage_seg_imgs / f"{stem}{src_path.suffix.lower()}")
            shutil.copy2(src_path, stage_det_imgs / f"{stem}{src_path.suffix.lower()}")
            Image.fromarray(final_mask, mode="L").save(stage_seg_masks / f"{stem}.png")
            (stage_det_lbls / f"{stem}.txt").write_text(
                "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
                encoding="utf-8",
            )

            overlays = generate_segmentation_overlays(rgb_img, final_mask, alpha=0.45)
            qa_img = overlays["overlay"]
            if yolo_lines:
                draw = ImageDraw.Draw(qa_img)
                for line in yolo_lines:
                    _, cx, cy, nw, nh = [float(v) for v in line.split()]
                    x1 = (cx - nw / 2.0) * actual_w
                    y1 = (cy - nh / 2.0) * actual_h
                    x2 = (cx + nw / 2.0) * actual_w
                    y2 = (cy + nh / 2.0) * actual_h
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=2)
            qa_img.save(stage_qa / f"{stem}_qa.png")

            processed_records.append({
                "stem": stem,
                "source_file": img_name,
                "expected_class": expected_class,
                "sha256": img_hash,
                "polygon_signature": _compute_image_polygon_signature(img_elem),
                "in_test_split": in_test_split,
                "dimensions": [actual_w, actual_h],
                "leaf_pixels": leaf_px,
                "healthy_pixels": healthy_px,
                "disease_pixels": disease_px,
                "affected_pct": round((disease_px / leaf_px) * 100.0, 2),
                "yolo_box_count": len(yolo_lines),
                "disease_types": sorted(set(disease_types_found)),
            })

        should_publish = bool(publish and len(blocking_errors) == 0 and len(processed_records) > 0)
        if should_publish:
            seg_images_dir = seg_out_dir / "images"
            seg_masks_dir = seg_out_dir / "masks"
            det_images_dir = det_out_dir / "images"
            det_labels_dir = det_out_dir / "labels"
            for d in (seg_images_dir, seg_masks_dir, det_images_dir, det_labels_dir, qa_out_dir):
                d.mkdir(parents=True, exist_ok=True)

            for f in stage_seg_imgs.iterdir():
                shutil.copy2(f, seg_images_dir / f.name)
            for f in stage_seg_masks.iterdir():
                shutil.copy2(f, seg_masks_dir / f.name)
            for f in stage_det_imgs.iterdir():
                shutil.copy2(f, det_images_dir / f.name)
            for f in stage_det_lbls.iterdir():
                shutil.copy2(f, det_labels_dir / f.name)
            for f in stage_qa.iterdir():
                shutil.copy2(f, qa_out_dir / f.name)

            det_manifest_df = pd.DataFrame([
                {
                    "stem": r["stem"],
                    "image_path": str((det_images_dir / f"{r['stem']}{Path(r['source_file']).suffix.lower()}")),
                    "label_path": str((det_labels_dir / f"{r['stem']}.txt")),
                    "sha256": r["sha256"],
                    "in_test_split": r["in_test_split"],
                    "box_count": r["yolo_box_count"],
                }
                for r in processed_records
            ])
            det_manifest_path = det_out_dir / "detection_manifest.csv"
            if det_manifest_path.is_file():
                try:
                    old_det_df = pd.read_csv(det_manifest_path)
                    combined_det_df = pd.concat([old_det_df[~old_det_df["stem"].isin(det_manifest_df["stem"])], det_manifest_df], ignore_index=True)
                    combined_det_df.to_csv(det_manifest_path, index=False)
                except Exception:
                    det_manifest_df.to_csv(det_manifest_path, index=False)
            else:
                det_manifest_df.to_csv(det_manifest_path, index=False)

    status_str = "PUBLISHED" if should_publish else ("DRY_RUN_VALID" if not blocking_errors and len(processed_records) > 0 else "BLOCKED_VALIDATION_ERRORS")
    report = {
        "status": status_str,
        "source_xml": str(xml_path),
        "published_to_workspace": should_publish,
        "processed_valid_count": len(processed_records),
        "unannotated_images_count": len(unannotated_images),
        "unannotated_images": unannotated_images,
        "skipped_or_invalid_count": len(blocking_errors),
        "suspicious_cross_image_duplication": suspicious_duplication,
        "identical_polygon_signature_groups": identical_groups,
        "test_split_leakage_detected": len(leakage_flags) > 0,
        "test_split_overlapping_files": leakage_flags,
        "manifest_mismatches": manifest_mismatches,
        "ready_for_unet_training": len([r for r in processed_records if not r["in_test_split"]]) >= 50 and len(blocking_errors) == 0,
        "ready_for_yolo_training": len([r for r in processed_records if not r["in_test_split"] and r["yolo_box_count"] > 0]) >= 50 and len(blocking_errors) == 0,
        "warnings": warnings,
        "records": processed_records,
        "errors": blocking_errors,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(description="Convert exported CVAT archive into verified U-Net masks and YOLO labels")
    parser.add_argument("--input", type=str, required=True, help="Path to CVAT export .zip or extracted folder containing annotations.xml")
    parser.add_argument("--extra-images-dir", type=str, default=None, help="Optional directory containing source images")
    parser.add_argument("--pilot-manifest", type=str, default="artifacts/manifests/pilot_batch_manifest.csv")
    parser.add_argument("--strict-manifest", action="store_true", help="Fail if any annotated image is absent from pilot manifest")
    parser.add_argument("--strict-disease-match", action="store_true", default=True, help="Verify disease_type matches manifest class label")
    parser.add_argument("--no-strict-disease-match", action="store_false", dest="strict_disease_match")
    parser.add_argument("--out-of-leaf-tolerance", type=float, default=0.02, help="Allowed ratio of Diseased_Region polygon pixels outside Leaf (default 0.02)")
    parser.add_argument("--allow-identical-signatures", action="store_true", help="Allow repeated polygon signatures after explicit human review")
    parser.add_argument("--dry-run", action="store_true", help="Validate and stage without publishing to data/annotations/ or data/detection/")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"CVAT input path not found: {input_path}")

    extract_dir = input_path
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        extract_dir = PROJECT_ROOT / "artifacts" / "cvat_extracted" / input_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(extract_dir)

    xml_candidates = list(extract_dir.rglob("annotations.xml"))
    if not xml_candidates:
        raise FileNotFoundError(f"No annotations.xml found inside {extract_dir}. Export CVAT task in 'CVAT for images 1.1' format.")

    dataset_root = Path("D:/Main_Project/dataset/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection/Original Dataset")
    if not dataset_root.exists():
        dataset_root = Path("/mnt/d/Main_Project/dataset/SAR-CLD-2024 A Comprehensive Dataset for Cotton Leaf Disease Detection/Original Dataset")

    pilot_imgs = PROJECT_ROOT / "data" / "pilot_batch_01" / "images"
    search_dirs = [extract_dir, extract_dir / "images", pilot_imgs, dataset_root]
    if args.extra_images_dir:
        search_dirs.insert(0, Path(args.extra_images_dir))

    manifest_p = Path(args.pilot_manifest)
    if not manifest_p.is_absolute():
        manifest_p = PROJECT_ROOT / manifest_p

    report = process_cvat_xml_export(
        xml_path=xml_candidates[0],
        image_search_dirs=search_dirs,
        seg_out_dir=PROJECT_ROOT / "data" / "annotations",
        det_out_dir=PROJECT_ROOT / "data" / "detection",
        qa_out_dir=PROJECT_ROOT / "artifacts" / "annotations_qa",
        pilot_manifest_path=manifest_p if manifest_p.is_file() else None,
        strict_manifest=args.strict_manifest,
        strict_disease_match=args.strict_disease_match,
        strict_leaf_containment=True,
        out_of_leaf_tolerance_ratio=args.out_of_leaf_tolerance,
        allow_identical_signatures=args.allow_identical_signatures,
        publish=not args.dry_run,
    )
    print(json.dumps(report, indent=2))
    if report.get("status") in ("REJECTED", "REVIEW_REQUIRED", "BLOCKED_VALIDATION_ERRORS"):
        sys.exit(2)


if __name__ == "__main__":
    main()
