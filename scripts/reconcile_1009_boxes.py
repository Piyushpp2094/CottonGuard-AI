"""
Reconcile all 1,009 Tier 1 Detection Boxes and 130 Tier 2 Ignore Regions (Phase 9.5.3.1).
Parses raw XML from Tasks 4 and 5 and raw JSON from Task 6.
Calculates class-wise totals and reconciles historical figures.
"""

import sys
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "manifests" / "consolidated_100_annotation_manifest.csv"

manifest = pd.read_csv(MANIFEST_PATH)

# 1. Parse Task 4 raw XML
t4_xml_images = {}
with zipfile.ZipFile(PROJECT_ROOT / "data" / "cvat_exports" / "task4_pilot_all21.zip") as z:
    root = ET.fromstring(z.read("annotations.xml"))
    for img in root.findall("image"):
        t4_xml_images[img.attrib["name"]] = img

# 2. Parse Task 5 raw XML
t5_xml_images = {}
with zipfile.ZipFile(PROJECT_ROOT / "data" / "cvat_exports" / "task5_pilot_batch02.zip") as z:
    root = ET.fromstring(z.read("annotations.xml"))
    for img in root.findall("image"):
        t5_xml_images[img.attrib["name"]] = img

# 3. Parse Task 6 raw JSON
with open(PROJECT_ROOT / "data" / "annotation_pilot" / "phase9_5" / "candidate_annotations_task6.json", "r", encoding="utf-8") as f:
    t6_json = json.load(f)

t6_manifest = manifest[manifest["cvat_task_id"] == 6].sort_values("global_id").reset_index(drop=True)
frame_to_fn = {i: t6_manifest.iloc[i]["filename"] for i in range(50)}

# Class tracking structures
CLASSES = [
    "Bacterial Blight",
    "Curl Virus",
    "Healthy Leaf",
    "Herbicide Growth Damage",
    "Leaf Hopper Jassids",
    "Leaf Redding",
    "Leaf Variegation",
]

class_data = {
    c: {
        "images": 0,
        "focal_polys": 0,
        "explicit_rects": 0,
        "derived_boxes": 0,
        "emitted_targets": 0,
        "diffuse_polys": 0,
        "ignore_boxes": 0,
    }
    for c in CLASSES
}

# Image-level details
image_records = []

for idx, row in manifest.iterrows():
    fn = row["filename"]
    task_id = row["cvat_task_id"]
    cl = row["class_label"]
    global_id = row["global_id"]

    class_data[cl]["images"] += 1

    img_focal_polys = 0
    img_explicit_rects = 0
    img_derived_boxes = 0
    img_emitted_targets = 0
    img_diffuse_polys = 0
    img_ignore_boxes = 0

    if task_id in [4, 5]:
        xml_img = t4_xml_images[fn] if task_id == 4 else t5_xml_images[fn]
        # In Tasks 4 & 5, disease shapes are polygons
        # Each disease polygon has attribute disease_type
        # Bounding boxes were derived from polygon extrema
        for p in xml_img.findall("polygon"):
            lbl = p.attrib.get("label")
            if lbl == "Leaf":
                continue
            # Disease polygon
            img_focal_polys += 1
            img_derived_boxes += 1
            img_emitted_targets += 1
            # In Tasks 4 & 5, diffuse symptoms were not separately marked as Tier 2 ignore polygons
    elif task_id == 6:
        frame_idx = global_id - 50  # Frames 0..49
        shapes = [s for s in t6_json["shapes"] if s["frame"] == frame_idx]
        rects = [s for s in shapes if s["type"] == "rectangle" and s["label_id"] == 6]
        polys = [s for s in shapes if s["type"] == "polygon" and s["label_id"] == 6]

        img_explicit_rects = len(rects)
        img_emitted_targets = len(rects)

        # Match rects to polys to differentiate focal vs diffuse
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
                    best_iou = iou
                    best_idx = p_idx
            if best_idx >= 0 and best_iou > 0.5:
                matched_p.add(best_idx)

        img_focal_polys = len(matched_p)
        img_diffuse_polys = len(polys) - len(matched_p)
        img_ignore_boxes = img_diffuse_polys

    class_data[cl]["focal_polys"] += img_focal_polys
    class_data[cl]["explicit_rects"] += img_explicit_rects
    class_data[cl]["derived_boxes"] += img_derived_boxes
    class_data[cl]["emitted_targets"] += img_emitted_targets
    class_data[cl]["diffuse_polys"] += img_diffuse_polys
    class_data[cl]["ignore_boxes"] += img_ignore_boxes

    image_records.append({
        "global_id": global_id,
        "filename": fn,
        "task_id": task_id,
        "class_label": cl,
        "focal_polys": img_focal_polys,
        "explicit_rects": img_explicit_rects,
        "derived_boxes": img_derived_boxes,
        "emitted_targets": img_emitted_targets,
        "diffuse_polys": img_diffuse_polys,
        "ignore_boxes": img_ignore_boxes,
    })

df_images = pd.DataFrame(image_records)
df_class = pd.DataFrame.from_dict(class_data, orient="index")

print("=" * 80)
print("CLASS-WISE DETECTION TARGET & IGNORE RECONCILIATION TABLE")
print("=" * 80)
print(df_class.to_string())

print("\n" + "=" * 80)
print("TASK-WISE BREAKDOWN TABLE")
print("=" * 80)
task_summary = df_images.groupby("task_id").agg({
    "filename": "count",
    "focal_polys": "sum",
    "explicit_rects": "sum",
    "derived_boxes": "sum",
    "emitted_targets": "sum",
    "diffuse_polys": "sum",
    "ignore_boxes": "sum",
}).rename(columns={"filename": "images"})
print(task_summary.to_string())

print("\n" + "=" * 80)
print("CORPUS TOTALS")
print("=" * 80)
print(f"Total Source Images: {df_images['filename'].count()}")
print(f"Total Tier 1 Focal Polygons: {df_images['focal_polys'].sum()}")
print(f"Total Explicit Rectangles: {df_images['explicit_rects'].sum()}")
print(f"Total Derived Boxes: {df_images['derived_boxes'].sum()}")
print(f"Total Emitted Focal Targets (Tier 1): {df_images['emitted_targets'].sum()}")
print(f"Total Tier 2 Diffuse Polygons: {df_images['diffuse_polys'].sum()}")
print(f"Total Generated Ignore Boxes: {df_images['ignore_boxes'].sum()}")
hl_df = df_images[df_images["class_label"] == "Healthy Leaf"]
print(f"Healthy Controls: {len(hl_df)} images (Boxes: {hl_df['emitted_targets'].sum()}, Ignore: {hl_df['ignore_boxes'].sum()})")
