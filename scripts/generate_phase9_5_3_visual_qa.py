"""
Generate Diagnostic Visual Overlays for Phase 9.5.3 Pre-Training Readiness Audit.
Renders visual QA panels comparing Tier 1 focal boxes, Tier 2 diffuse polygons, and generated ignore boxes.
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "artifacts" / "visual_qa" / "phase9_5_3_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

candidate_json = PROJECT_ROOT / "data" / "annotation_pilot" / "phase9_5" / "candidate_annotations_task6.json"
manifest_path = PROJECT_ROOT / "artifacts" / "manifests" / "consolidated_100_annotation_manifest.csv"
images_dir = PROJECT_ROOT / "data" / "pilot_batch_03" / "images"

with open(candidate_json, "r", encoding="utf-8") as f:
    t6_data = json.load(f)

manifest = pd.read_csv(manifest_path)
t6_manifest = manifest[manifest["cvat_task_id"] == 6].sort_values("global_id").reset_index(drop=True)

# Select 4 representative frames:
# Frame 5: CV00001.jpg (Curl Virus large deformation ignore box)
# Frame 16: LHJ00059.jpg (Leaf Hopper Jassids marginal chlorosis ribbons)
# Frame 20: LR00001.jpg (Leaf Redding diffuse anthocyanin blush)
# Frame 21: LR00140.jpg (Mixed focal lesions + diffuse blush)
selected_frames = [5, 16, 20, 21]

for frame_idx in selected_frames:
    fn = t6_manifest.iloc[frame_idx]["filename"]
    cl = t6_manifest.iloc[frame_idx]["class_label"]
    img_path = images_dir / fn

    shapes = [s for s in t6_data["shapes"] if s["frame"] == frame_idx]
    rects = [s for s in shapes if s["type"] == "rectangle"]
    polys = [s for s in shapes if s["type"] == "polygon" and s["label_id"] == 6]
    leaf = [s for s in shapes if s["label_id"] == 5]

    # Load original image
    orig_bgr = cv2.imread(str(img_path))
    h, w = orig_bgr.shape[:2]

    # Panel 1: Native Image with Leaf Polygon
    panel1 = orig_bgr.copy()
    if len(leaf) > 0:
        l_pts = np.array(leaf[0]["points"], dtype=np.int32).reshape(-1, 2)
        cv2.polylines(panel1, [l_pts], True, (0, 255, 0), 2)

    # Panel 2: Ground Truth Annotations (Focal in Cyan, Diffuse Polys in Magenta)
    panel2 = orig_bgr.copy()
    # Match rects to polys
    matched_p = set()
    for r in rects:
        rx1, ry1, rx2, ry2 = r["points"]
        best_iou, best_idx = 0.0, -1
        for p_idx, p in enumerate(polys):
            if p_idx in matched_p:
                continue
            pts = np.array(p["points"]).reshape(-1, 2)
            px1, py1 = pts.min(axis=0)
            px2, py2 = pts.max(axis=0)
            ix1, iy1 = max(rx1, px1), max(ry1, py1)
            ix2, iy2 = min(rx2, px2), min(ry2, py2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (rx2 - rx1) * (ry2 - ry1) + (px2 - px1) * (py2 - py1) - inter
            iou = inter / max(1e-6, union)
            if iou > best_iou:
                best_iou, best_idx = iou, p_idx
        if best_idx >= 0 and best_iou > 0.5:
            matched_p.add(best_idx)

    # Draw focal lesions in Cyan (0, 255, 255)
    for p_idx in matched_p:
        pts = np.array(polys[p_idx]["points"], dtype=np.int32).reshape(-1, 2)
        cv2.polylines(panel2, [pts], True, (255, 255, 0), 2)
    for r in rects:
        rx1, ry1, rx2, ry2 = map(int, r["points"])
        cv2.rectangle(panel2, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

    # Draw diffuse polygons in Magenta (255, 0, 255)
    diffuse_boxes = []
    for p_idx, p in enumerate(polys):
        if p_idx in matched_p:
            continue
        pts = np.array(p["points"], dtype=np.int32).reshape(-1, 2)
        cv2.polylines(panel2, [pts], True, (255, 0, 255), 2)
        cv2.fillPoly(panel2, [pts], (255, 0, 255))
        px1, py1 = pts.min(axis=0)
        px2, py2 = pts.max(axis=0)
        diffuse_boxes.append([px1, py1, px2, py2])

    panel2 = cv2.addWeighted(orig_bgr, 0.4, panel2, 0.6, 0)

    # Panel 3: Generated Faster R-CNN Ignore Boxes (Yellow Dashed/Box) + Focal Boxes (Cyan)
    panel3 = orig_bgr.copy()
    for bx1, by1, bx2, by2 in diffuse_boxes:
        cv2.rectangle(panel3, (bx1, by1), (bx2, by2), (0, 165, 255), 2)  # Orange for ignore box
        # Add label
        cv2.putText(panel3, "TIER 2 IGNORE", (bx1 + 5, by1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    for r in rects:
        rx1, ry1, rx2, ry2 = map(int, r["points"])
        cv2.rectangle(panel3, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)  # Cyan for focal detection box
        cv2.putText(panel3, "TIER 1 FOCAL", (rx1 + 3, ry1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Resize panels to 400x400 for comparison card
    p1_small = cv2.resize(panel1, (400, 400))
    p2_small = cv2.resize(panel2, (400, 400))
    p3_small = cv2.resize(panel3, (400, 400))

    # Add header titles
    def add_title(img, text, color=(255, 255, 255)):
        banner = np.zeros((40, 400, 3), dtype=np.uint8)
        cv2.putText(banner, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        return np.vstack([banner, img])

    c1 = add_title(p1_small, "1. Native Lamina + Leaf Poly", (100, 255, 100))
    c2 = add_title(p2_small, "2. GT: Cyan=Focal, Magenta=Diffuse", (255, 200, 255))
    c3 = add_title(p3_small, "3. Detector: Cyan=Box, Orange=Ignore", (100, 200, 255))

    card = np.hstack([c1, c2, c3])

    # Top banner with metadata
    top_banner = np.zeros((50, 1200, 3), dtype=np.uint8)
    title_text = f"Phase 9.5.3 Readiness Audit: Frame {frame_idx:02d} - {fn} ({cl})"
    cv2.putText(top_banner, title_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    full_card = np.vstack([top_banner, card])

    out_file = OUT_DIR / f"audit_overlay_frame_{frame_idx:02d}_{Path(fn).stem}.png"
    cv2.imwrite(str(out_file), full_card)
    print(f"Generated diagnostic overlay: {out_file.name}")

print("Visual QA generation completed successfully.")
