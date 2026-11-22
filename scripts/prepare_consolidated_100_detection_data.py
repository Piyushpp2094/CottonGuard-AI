"""
Prepare Isolated Consolidated 100-Image Detection Dataset (Phase 9.5.3.2).
Stages data/detection_consolidated_100/ with images, labels, and ignore_labels.
Strictly non-destructive: Preserves data/detection/ and all existing checkpoints.
"""

import sys
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "manifests" / "consolidated_100_annotation_manifest.csv"
CANDIDATE_JSON = PROJECT_ROOT / "data" / "annotation_pilot" / "phase9_5" / "candidate_annotations_task6.json"

T45_IMG_DIR = PROJECT_ROOT / "data" / "detection" / "images"
T45_LBL_DIR = PROJECT_ROOT / "data" / "detection" / "labels"
T6_IMG_DIR = PROJECT_ROOT / "data" / "pilot_batch_03" / "images"

DEST_DIR = PROJECT_ROOT / "data" / "detection_consolidated_100"
DEST_IMG_DIR = DEST_DIR / "images"
DEST_LBL_DIR = DEST_DIR / "labels"
DEST_IGN_DIR = DEST_DIR / "ignore_labels"

CLASS_NAME_TO_YOLO_ID = {
    "Bacterial Blight": 0,
    "Curl Virus": 1,
    "Herbicide Growth Damage": 2,
    "Leaf Hopper Jassids": 3,
    "Leaf Redding": 4,
    "Leaf Variegation": 5,
}

def prepare_staged_dataset():
    print("=" * 80)
    print("COTTONGUARD AI — PREPARING ISOLATED CONSOLIDATED 100-IMAGE DETECTION DATASET")
    print("=" * 80)

    DEST_IMG_DIR.mkdir(parents=True, exist_ok=True)
    DEST_LBL_DIR.mkdir(parents=True, exist_ok=True)
    DEST_IGN_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    t6_manifest = manifest[manifest["cvat_task_id"] == 6].sort_values("global_id").reset_index(drop=True)

    with open(CANDIDATE_JSON, "r", encoding="utf-8") as f:
        t6_data = json.load(f)

    total_focal_boxes = 0
    total_ignore_boxes = 0
    total_images_copied = 0

    for idx, row in manifest.iterrows():
        fn = row["filename"]
        stem = Path(fn).stem
        task_id = row["cvat_task_id"]
        class_label = row["class_label"]
        global_id = row["global_id"]

        dest_img_path = DEST_IMG_DIR / fn
        dest_lbl_path = DEST_LBL_DIR / f"{stem}.txt"
        dest_ign_path = DEST_IGN_DIR / f"{stem}.txt"

        # 1. Copy image file
        if task_id in [4, 5]:
            src_img_path = T45_IMG_DIR / fn
        else:
            src_img_path = T6_IMG_DIR / fn

        if not src_img_path.is_file():
            raise FileNotFoundError(f"Source image missing: {src_img_path}")

        if not dest_img_path.is_file():
            shutil.copy2(src_img_path, dest_img_path)
        total_images_copied += 1

        # 2. Extract or copy labels
        if task_id in [4, 5]:
            # Tasks 4 & 5: Copy existing verified YOLO labels
            src_lbl_path = T45_LBL_DIR / f"{stem}.txt"
            if not src_lbl_path.is_file():
                raise FileNotFoundError(f"Source label missing: {src_lbl_path}")
            lines = src_lbl_path.read_text(encoding="utf-8").strip().splitlines()
            dest_lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            total_focal_boxes += len([l for l in lines if l.strip()])

            # No Tier 2 ignore boxes in Tasks 4 & 5
            dest_ign_path.write_text("", encoding="utf-8")

        elif task_id == 6:
            # Task 6: Derive from candidate_annotations_task6.json
            frame_idx = global_id - 50
            shapes = [s for s in t6_data["shapes"] if s["frame"] == frame_idx]
            rects = [s for s in shapes if s["type"] == "rectangle" and s["label_id"] == 6]
            polys = [s for s in shapes if s["type"] == "polygon" and s["label_id"] == 6]

            # Get image size
            with Image.open(src_img_path) as im:
                w, h = im.size

            # Match rects to polys to separate focal vs diffuse
            matched_poly_indices = set()
            for r in rects:
                rx1, ry1, rx2, ry2 = r["points"]
                best_iou, best_idx = 0.0, -1
                for p_idx, p in enumerate(polys):
                    if p_idx in matched_poly_indices:
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
                    matched_poly_indices.add(best_idx)

            # Write focal YOLO boxes
            yolo_id = CLASS_NAME_TO_YOLO_ID.get(class_label, 0)
            focal_lines = []
            for r in rects:
                rx1, ry1, rx2, ry2 = r["points"]
                bw = (rx2 - rx1) / float(w)
                bh = (ry2 - ry1) / float(h)
                cx = (rx1 + rx2) / (2.0 * float(w))
                cy = (ry1 + ry2) / (2.0 * float(h))
                focal_lines.append(f"{yolo_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            dest_lbl_path.write_text("\n".join(focal_lines) + ("\n" if focal_lines else ""), encoding="utf-8")
            total_focal_boxes += len(focal_lines)

            # Write ignore boxes (diffuse polygons)
            ignore_lines = []
            for p_idx, p in enumerate(polys):
                if p_idx in matched_poly_indices:
                    continue
                pts = np.array(p["points"]).reshape(-1, 2)
                px1, py1 = pts.min(axis=0)
                px2, py2 = pts.max(axis=0)
                ibw = (px2 - px1) / float(w)
                ibh = (py2 - py1) / float(h)
                icx = (px1 + px2) / (2.0 * float(w))
                icy = (py1 + py2) / (2.0 * float(h))
                ignore_lines.append(f"0 {icx:.6f} {icy:.6f} {ibw:.6f} {ibh:.6f}")

            dest_ign_path.write_text("\n".join(ignore_lines) + ("\n" if ignore_lines else ""), encoding="utf-8")
            total_ignore_boxes += len(ignore_lines)

    print(f"Staged images: {total_images_copied}/100")
    print(f"Total Tier 1 Focal Boxes staged: {total_focal_boxes} (Expected: 1,009)")
    print(f"Total Tier 2 Ignore Boxes staged: {total_ignore_boxes} (Expected: 130)")
    assert total_focal_boxes == 1009, f"Mismatch in focal boxes: {total_focal_boxes} != 1009"
    assert total_ignore_boxes == 130, f"Mismatch in ignore boxes: {total_ignore_boxes} != 130"
    print("Consolidated 100-image detection dataset successfully staged.")
    print(f"Location: {DEST_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    prepare_staged_dataset()
