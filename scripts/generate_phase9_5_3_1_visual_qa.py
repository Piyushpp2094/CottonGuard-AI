"""
Generate Diagnostic Visual Overlays for Phase 9.5.3.1 Outliers & Risk Categories.
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "artifacts" / "visual_qa" / "phase9_5_3_1_preflight"
OUT_DIR.mkdir(parents=True, exist_ok=True)

candidate_json = PROJECT_ROOT / "data" / "annotation_pilot" / "phase9_5" / "candidate_annotations_task6.json"
manifest_path = PROJECT_ROOT / "artifacts" / "manifests" / "consolidated_100_annotation_manifest.csv"
images_dir = PROJECT_ROOT / "data" / "pilot_batch_03" / "images"

with open(candidate_json, "r", encoding="utf-8") as f:
    t6_data = json.load(f)

manifest = pd.read_csv(manifest_path)
t6_manifest = manifest[manifest["cvat_task_id"] == 6].sort_values("global_id").reset_index(drop=True)

# Select frames:
# 5 Whole-Leaf: Frame 43 (LR00003), Frame 37 (CV00006), Frame 6 (CV00102), Frame 5 (CV00001), Frame 7 (CV00211)
# Risk categories: Frame 40 (HGD00280), Frame 22 (LR00286), Frame 16 (LHJ00059), Frame 35 (BBC00008)
target_frames = [43, 37, 6, 5, 7, 40, 22, 16, 35]

for f_idx in target_frames:
    fn = t6_manifest.iloc[f_idx]["filename"]
    cl = t6_manifest.iloc[f_idx]["class_label"]
    img_path = images_dir / fn

    shapes = [s for s in t6_data["shapes"] if s["frame"] == f_idx]
    rects = [s for s in shapes if s["type"] == "rectangle" and s["label_id"] == 6]
    polys = [s for s in shapes if s["type"] == "polygon" and s["label_id"] == 6]
    leaf = [s for s in shapes if s["label_id"] == 5]

    orig_bgr = cv2.imread(str(img_path))
    h, w = orig_bgr.shape[:2]

    # Panel 1: Native image + Leaf polygon
    panel1 = orig_bgr.copy()
    if len(leaf) > 0:
        l_pts = np.array(leaf[0]["points"], dtype=np.int32).reshape(-1, 2)
        cv2.polylines(panel1, [l_pts], True, (0, 255, 0), 2)

    # Panel 2: GT Polygons (Focal in Cyan, Diffuse in Magenta)
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

    panel2 = orig_bgr.copy()
    for p_idx in matched_p:
        pts = np.array(polys[p_idx]["points"], dtype=np.int32).reshape(-1, 2)
        cv2.polylines(panel2, [pts], True, (255, 255, 0), 2)
    for r in rects:
        rx1, ry1, rx2, ry2 = map(int, r["points"])
        cv2.rectangle(panel2, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

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

    # Panel 3: Faster R-CNN Targets: Focal Boxes (Cyan) & Ignore Boxes (Orange)
    panel3 = orig_bgr.copy()
    for bx1, by1, bx2, by2 in diffuse_boxes:
        cv2.rectangle(panel3, (bx1, by1), (bx2, by2), (0, 140, 255), 2)
        cv2.putText(panel3, "TIER 2 IGNORE", (bx1 + 5, by1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)

    for r in rects:
        rx1, ry1, rx2, ry2 = map(int, r["points"])
        cv2.rectangle(panel3, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
        cv2.putText(panel3, "TIER 1 FOCAL", (rx1 + 3, ry1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Combine 3 panels (400x400 each)
    p1 = cv2.resize(panel1, (400, 400))
    p2 = cv2.resize(panel2, (400, 400))
    p3 = cv2.resize(panel3, (400, 400))

    def add_title(img, text, color=(255, 255, 255)):
        b = np.zeros((36, 400, 3), dtype=np.uint8)
        cv2.putText(b, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return np.vstack([b, img])

    c1 = add_title(p1, "1. Native Lamina + Leaf Poly", (100, 255, 100))
    c2 = add_title(p2, "2. GT: Cyan=Focal, Magenta=Diffuse", (255, 200, 255))
    c3 = add_title(p3, "3. Detector: Cyan=Focal, Orange=Ignore", (100, 200, 255))

    card = np.hstack([c1, c2, c3])
    top = np.zeros((45, 1200, 3), dtype=np.uint8)
    cv2.putText(top, f"Phase 9.5.3.1 Preflight: Frame {f_idx:02d} - {fn} ({cl})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    full_card = np.vstack([top, card])

    out_file = OUT_DIR / f"outlier_overlay_frame_{f_idx:02d}_{Path(fn).stem}.png"
    cv2.imwrite(str(out_file), full_card)
    print(f"Rendered diagnostic overlay: {out_file.name}")

print("All preflight diagnostic overlays generated successfully.")
