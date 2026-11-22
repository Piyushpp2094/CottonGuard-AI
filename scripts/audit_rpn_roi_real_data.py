"""
Real-Dataset RPN and RoI Assignment Audit Script (Phase 9.5.3).
Inspects anchor assignment and RoI proposal matching using real images and targets from Task 6.
Strictly non-executing: no optimizer, no loss backprop, no model training.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models.detection.image_list import ImageList

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.faster_rcnn import build_faster_rcnn_model
from src.models.ignore_region_faster_rcnn import substantially_overlaps_ignore

CLASS_NAME_TO_ID = {
    "Bacterial Blight": 1,
    "Curl Virus": 2,
    "Healthy Leaf": 0,
    "Herbicide Growth Damage": 3,
    "Leaf Hopper Jassids": 4,
    "Leaf Redding": 5,
    "Leaf Variegation": 6,
}

def extract_task6_targets(candidate_json_path: Path, manifest_path: Path):
    with open(candidate_json_path, "r", encoding="utf-8") as f:
        t6_data = json.load(f)

    manifest = pd.read_csv(manifest_path)
    t6_manifest = manifest[manifest["cvat_task_id"] == 6].sort_values("global_id").reset_index(drop=True)

    targets_by_frame = {}

    for f_idx in range(50):
        shapes = [s for s in t6_data["shapes"] if s["frame"] == f_idx]
        rects = [s for s in shapes if s["type"] == "rectangle"]
        polys = [s for s in shapes if s["type"] == "polygon" and s["label_id"] == 6]
        fn = t6_manifest.iloc[f_idx]["filename"]
        class_name = t6_manifest.iloc[f_idx]["class_label"]
        class_id = CLASS_NAME_TO_ID[class_name]

        # Match rects to polys to identify focal vs diffuse
        matched_poly_indices = set()
        for r in rects:
            rx1, ry1, rx2, ry2 = r["points"]
            best_iou = 0.0
            best_idx = -1
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

        # Focal boxes
        focal_boxes_list = []
        focal_labels_list = []
        for r in rects:
            rx1, ry1, rx2, ry2 = r["points"]
            if rx2 > rx1 and ry2 > ry1:
                focal_boxes_list.append([rx1, ry1, rx2, ry2])
                focal_labels_list.append(class_id)

        # Diffuse ignore boxes
        diffuse_boxes_list = []
        for p_idx, p in enumerate(polys):
            if p_idx in matched_poly_indices:
                continue
            pts = np.array(p["points"]).reshape(-1, 2)
            px1, py1 = pts.min(axis=0)
            px2, py2 = pts.max(axis=0)
            if px2 > px1 and py2 > py1:
                diffuse_boxes_list.append([float(px1), float(py1), float(px2), float(py2)])

        boxes = torch.as_tensor(focal_boxes_list, dtype=torch.float32) if len(focal_boxes_list) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(focal_labels_list, dtype=torch.int64) if len(focal_labels_list) > 0 else torch.zeros((0,), dtype=torch.int64)
        ignore_boxes = torch.as_tensor(diffuse_boxes_list, dtype=torch.float32) if len(diffuse_boxes_list) > 0 else torch.zeros((0, 4), dtype=torch.float32)

        targets_by_frame[f_idx] = {
            "filename": fn,
            "class_name": class_name,
            "target": {
                "boxes": boxes,
                "labels": labels,
                "ignore_boxes": ignore_boxes,
            }
        }

    return targets_by_frame


def run_audit():
    print("=" * 80)
    print("COTTON GUARD AI — REAL-DATASET RPN & ROI PRE-TRAINING READINESS AUDIT")
    print("=" * 80)

    device = torch.device("cpu")
    model = build_faster_rcnn_model(
        num_classes=7,
        pretrained=False,
        trainable_backbone_layers=1,
        device=device,
        enable_ignore_regions=True,
        rpn_ignore_threshold=0.30,
        roi_ignore_threshold=0.30,
    )
    model.eval()

    candidate_json = PROJECT_ROOT / "data" / "annotation_pilot" / "phase9_5" / "candidate_annotations_task6.json"
    manifest_path = PROJECT_ROOT / "artifacts" / "manifests" / "consolidated_100_annotation_manifest.csv"
    targets_by_frame = extract_task6_targets(candidate_json, manifest_path)

    # Audited sample frames covering all 4 categories:
    # Tier 1 only: Frame 1 (BBC00062), Frame 25 (LV00003), Frame 41 (LHJ00002)
    # Tier 2 only: Frame 0 (BBC00001), Frame 16 (LHJ00059), Frame 20 (LR00001), Frame 35 (BBC00008)
    # Both: Frame 5 (CV00001), Frame 21 (LR00140), Frame 44 (LR00573)
    # Neither (Healthy): Frame 30 (HL00002), Frame 48 (HL00099)
    audited_frames = [1, 25, 41, 0, 16, 20, 35, 5, 21, 44, 30, 48]

    # Find image file for each frame
    images_dir = PROJECT_ROOT / "data" / "pilot_batch_03" / "images"

    rpn_results = []
    roi_results = []

    for f_idx in audited_frames:
        info = targets_by_frame[f_idx]
        fn = info["filename"]
        cl = info["class_name"]
        tgt = info["target"]
        img_path = images_dir / fn

        with Image.open(img_path) as pil_img:
            img_tensor = TF.to_tensor(pil_img.convert("RGB")).unsqueeze(0)

        # Feature extraction & anchor generation
        with torch.no_grad():
            features = model.backbone(img_tensor)
            img_list = ImageList(img_tensor, [(800, 800)])
            features_list = list(features.values())
            anchors = model.rpn.anchor_generator(img_list, features_list)[0]

            # 1. RPN Assignment Audit
            # Base assignment without ignore
            gt_boxes = tgt["boxes"]
            ign_boxes = tgt["ignore_boxes"]

            if gt_boxes.numel() == 0:
                base_labels = torch.zeros((anchors.shape[0],), dtype=torch.float32)
            else:
                match_matrix = model.rpn.box_similarity(gt_boxes, anchors)
                matched_idxs = model.rpn.proposal_matcher(match_matrix)
                base_labels = (matched_idxs >= 0).to(dtype=torch.float32)
                bg_inds = matched_idxs == model.rpn.proposal_matcher.BELOW_LOW_THRESHOLD
                base_labels[bg_inds] = 0.0
                discard_inds = matched_idxs == model.rpn.proposal_matcher.BETWEEN_THRESHOLDS
                base_labels[discard_inds] = -1.0

            # Ignore-Aware Assignment
            labels_list, _ = model.rpn.assign_targets_to_anchors([anchors], [tgt])
            final_rpn_labels = labels_list[0]

            total_anchors = anchors.shape[0]
            pos_anchors = int((final_rpn_labels == 1.0).sum())
            neg_anchors = int((final_rpn_labels == 0.0).sum())
            ign_anchors = int((final_rpn_labels == -1.0).sum())

            # Specific ignore-region shifts
            shifted_to_ignore = int(((base_labels == 0.0) & (final_rpn_labels == -1.0)).sum())

            # Positive priority preservation
            if ign_boxes.numel() > 0 and pos_anchors > 0:
                overlaps_ign = substantially_overlaps_ignore(anchors, ign_boxes, threshold=0.30)
                pos_overlapping_ign = int((overlaps_ign & (base_labels == 1.0)).sum())
                pos_preserved = int((overlaps_ign & (final_rpn_labels == 1.0)).sum())
            else:
                pos_overlapping_ign = 0
                pos_preserved = 0

            rpn_results.append({
                "frame": f_idx,
                "filename": fn,
                "class": cl,
                "focal_boxes": tgt["boxes"].shape[0],
                "diffuse_boxes": tgt["ignore_boxes"].shape[0],
                "total_anchors": total_anchors,
                "pos_anchors": pos_anchors,
                "neg_anchors": neg_anchors,
                "ign_anchors": ign_anchors,
                "shifted_to_ignore": shifted_to_ignore,
                "pos_overlapping_ign": pos_overlapping_ign,
                "pos_preserved": pos_preserved,
            })

            # 2. RoI Assignment Audit
            # Generate proposals
            objectness, pred_bbox_deltas = model.rpn.head(features_list)
            num_anchors_per_level = [o[0].shape[0] * o[0].shape[1] * o[0].shape[2] for o in objectness]
            from torchvision.models.detection.rpn import concat_box_prediction_layers
            obj_cat, deltas_cat = concat_box_prediction_layers(objectness, pred_bbox_deltas)
            proposals = model.rpn.box_coder.decode(deltas_cat.detach(), [anchors])
            proposals = proposals.view(1, -1, 4)
            proposals, _ = model.rpn.filter_proposals(proposals, obj_cat, [(800, 800)], num_anchors_per_level)
            prop = proposals[0]

            # RoI Matching
            gt_labels = [tgt["labels"]]
            gt_boxes_list = [tgt["boxes"]]
            props_with_gt = model.roi_heads.add_gt_proposals([prop], gt_boxes_list)
            matched_idxs_roi, raw_roi_labels = model.roi_heads.assign_targets_to_proposals(props_with_gt, gt_boxes_list, gt_labels)
            raw_labels = raw_roi_labels[0].clone()

            # Now select_training_samples
            _, _, final_roi_labels_list, _ = model.roi_heads.select_training_samples([prop], [tgt])
            final_roi_labels = final_roi_labels_list[0]

            # In the proposals before subsampling:
            if ign_boxes.numel() > 0:
                overlaps_roi_ign = substantially_overlaps_ignore(props_with_gt[0], ign_boxes, threshold=0.30)
                proposals_shifted_to_ignore = int((overlaps_roi_ign & (raw_labels <= 0)).sum())
                pos_proposals_overlapping_ign = int((overlaps_roi_ign & (raw_labels > 0)).sum())
            else:
                proposals_shifted_to_ignore = 0
                pos_proposals_overlapping_ign = 0

            roi_results.append({
                "frame": f_idx,
                "filename": fn,
                "class": cl,
                "total_proposals": props_with_gt[0].shape[0],
                "sampled_for_loss": final_roi_labels.numel(),
                "sampled_pos": int((final_roi_labels > 0).sum()),
                "sampled_neg": int((final_roi_labels == 0).sum()),
                "proposals_masked_to_ignore": proposals_shifted_to_ignore,
                "pos_proposals_overlapping_ign": pos_proposals_overlapping_ign,
            })

    df_rpn = pd.DataFrame(rpn_results)
    df_roi = pd.DataFrame(roi_results)

    print("\n--- RPN ASSIGNMENT AUDIT TABLE ---")
    print(df_rpn.to_string())

    print("\n--- ROI ASSIGNMENT AUDIT TABLE ---")
    print(df_roi.to_string())

    # Save summary json
    summary = {
        "rpn_audit": rpn_results,
        "roi_audit": roi_results,
    }
    out_path = PROJECT_ROOT / "reports" / "phase9_5_3_assignment_audit_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAudit summary saved to: {out_path}")

if __name__ == "__main__":
    run_audit()
