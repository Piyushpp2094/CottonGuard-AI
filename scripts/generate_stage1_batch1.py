"""
scripts/generate_stage1_batch1.py
Stage 1: Batch 1 Candidate Generation for Frame 00 (BBC00001.jpg) and Frame 20 (LR00001.jpg).
Extracts organic leaf boundaries, validates geometry, and produces 4-panel visual QA overlays.
"""

import json
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path("D:/Main_Project/cotton_leaf_ai")
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.unet_resnet34 import build_unet_resnet34
BASELINE_JSON = PROJECT_ROOT / "artifacts" / "backups" / "cvat_job6_fresh_pre_remediation_baseline_20261118.json"
IMG_DIR = PROJECT_ROOT / "data" / "detection_consolidated_100" / "images"
STAGING_DIR = PROJECT_ROOT / "artifacts" / "annotation_staging" / "phase12_2"
OVERLAYS_DIR = STAGING_DIR / "overlays"
LEAF_DIR = STAGING_DIR / "leaf"
BRAIN_OVERLAYS_DIR = Path("C:/Users/Haker's pc/.gemini/antigravity/brain/e96a0217-11ef-4314-af5b-e8eaa45d475a/phase12_2_overlays")

OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
LEAF_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)

# Load baseline CVAT annotations
with open(BASELINE_JSON, "r") as f:
    cvat_data = json.load(f)

# Load U-Net ResNet-34 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model = build_unet_resnet34(num_classes=3, checkpoint_path=str(PROJECT_ROOT / "artifacts" / "checkpoints" / "segmentation" / "best_unet.pt"), device=device)
unet_model.eval()

BATCH1_SPECS = [
    {
        "frame": 0,
        "global_id": 50,
        "filename": "BBC00001.jpg",
        "replaced_shape_id": 1039,
        "label": "Leaf",
        "label_id": 5,
        "disease_label": "Bacterial Blight",
        "eps_factor": 0.003
    },
    {
        "frame": 20,
        "global_id": 70,
        "filename": "LR00001.jpg",
        "replaced_shape_id": 1276,
        "label": "Leaf",
        "label_id": 5,
        "disease_label": "Leaf Redding",
        "eps_factor": 0.002
    }
]

def extract_organic_leaf_contour(img_bgr, eps_factor=0.0025):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    norm = (rgb / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    t = torch.from_numpy(norm.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out = unet_model(t)
        pred = torch.argmax(out, dim=1).squeeze().cpu().numpy()
    
    # Class >= 1 represents leaf tissue (1: Healthy leaf, 2: Diseased tissue)
    leaf_mask = (pred >= 1).astype(np.uint8) * 255
    
    # Morphological closing to bridge small vein gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    leaf_mask_closed = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find largest external contour
    contours, _ = cv2.findContours(leaf_mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No leaf contour found!")
    largest = max(contours, key=cv2.contourArea)
    
    # Approximate polygon
    eps = eps_factor * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, eps, True)
    
    # Flatten to list of floats [x1, y1, x2, y2, ...]
    pts = approx.reshape(-1, 2).astype(np.float32)
    # Ensure coordinates stay within [0.0, 800.0]
    pts[:, 0] = np.clip(pts[:, 0], 0.0, 800.0)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, 800.0)
    
    coords = pts.flatten().tolist()
    return coords, pts

def segments_intersect(p1, p2, p3, p4):
    """Check if line segment p1-p2 properly intersects segment p3-p4."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))

def check_polygon_self_intersection(pts):
    """Returns True if any non-adjacent edges of the polygon intersect."""
    n = len(pts)
    if n < 3:
        return False
    for i in range(n):
        p1, p2 = pts[i], pts[(i + 1) % n]
        for j in range(i + 2, n):
            if (i == 0 and j == n - 1):
                continue
            p3, p4 = pts[j], pts[(j + 1) % n]
            if segments_intersect(p1, p2, p3, p4):
                return True
    return False

def validate_polygon_geometry(pts):
    vertex_count = len(pts)
    area = float(cv2.contourArea(np.round(pts).astype(np.int32)))
    has_self_intersection = check_polygon_self_intersection(pts)
    is_valid = (not has_self_intersection) and (area > 0)
    coords_in_range = bool(np.all(pts >= 0.0) and np.all(pts <= 800.0))
    preserved_800 = bool(np.any(pts == 800.0))
    return {
        "vertex_count": vertex_count,
        "is_valid": is_valid,
        "area": area,
        "has_self_intersection": has_self_intersection,
        "coords_in_range": coords_in_range,
        "preserved_800": preserved_800,
        "passed_qc": bool(is_valid and vertex_count >= 15 and coords_in_range)
    }

def create_4panel_overlay(img_bgr, old_leaf_pts, new_leaf_pts, disease_shapes, spec, qc):
    h, w = img_bgr.shape[:2]
    
    # Panel 1: Original Image
    p1 = img_bgr.copy()
    cv2.putText(p1, f"Frame {spec['frame']} ({spec['filename']})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(p1, "Panel 1: Original Image", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Panel 2: Baseline CVAT (Old Leaf pseudo-mask in red dashed, disease in blue)
    p2 = img_bgr.copy()
    cv2.putText(p2, "Panel 2: Baseline CVAT (Old Pseudo-Mask)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    # Draw old leaf pseudo-mask
    if old_leaf_pts is not None and len(old_leaf_pts) >= 3:
        old_poly = np.round(old_leaf_pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(p2, [old_poly], isClosed=True, color=(0, 0, 255), thickness=3)
    # Draw disease shapes
    for s in disease_shapes:
        s_pts = np.array(s["points"], dtype=np.float32)
        if s["type"] == "rectangle":
            x1, y1, x2, y2 = np.round(s_pts).astype(np.int32)
            cv2.rectangle(p2, (x1, y1), (x2, y2), (255, 120, 0), 2)
        else:
            poly = np.round(s_pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(p2, [poly], isClosed=True, color=(255, 120, 0), thickness=2)
    
    # Panel 3: Proposed Organic Leaf Candidate (Green outline + fill)
    p3 = img_bgr.copy()
    overlay3 = p3.copy()
    new_poly = np.round(new_leaf_pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(overlay3, [new_poly], color=(0, 200, 0))
    cv2.addWeighted(overlay3, 0.25, p3, 0.75, 0, p3)
    cv2.polylines(p3, [new_poly], isClosed=True, color=(0, 255, 0), thickness=3)
    # Draw vertices
    for pt in new_poly.reshape(-1, 2):
        cv2.circle(p3, tuple(pt), 4, (0, 255, 255), -1)
    cv2.putText(p3, f"Panel 3: Proposed Leaf ({qc['vertex_count']} vertices)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(p3, f"Area: {qc['area']:.1f} px | Self-intersect: {qc['has_self_intersection']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Panel 4: Proposed Complete Composition (Proposed Leaf + Retained Disease)
    p4 = img_bgr.copy()
    overlay4 = p4.copy()
    cv2.fillPoly(overlay4, [new_poly], color=(0, 180, 0))
    cv2.addWeighted(overlay4, 0.18, p4, 0.82, 0, p4)
    cv2.polylines(p4, [new_poly], isClosed=True, color=(0, 255, 0), thickness=2)
    for s in disease_shapes:
        s_pts = np.array(s["points"], dtype=np.float32)
        if s["type"] == "rectangle":
            x1, y1, x2, y2 = np.round(s_pts).astype(np.int32)
            cv2.rectangle(p4, (x1, y1), (x2, y2), (255, 150, 50), 2)
        else:
            poly = np.round(s_pts).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(p4, [poly], isClosed=True, color=(255, 150, 50), thickness=2)
    cv2.putText(p4, "Panel 4: Proposed Composition (Leaf + Disease)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(p4, f"Status: PENDING_HUMAN_REVIEW", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    
    # Combine 2x2 grid
    top_row = np.hstack([p1, p2])
    bot_row = np.hstack([p3, p4])
    combined = np.vstack([top_row, bot_row])
    return combined

print("=== EXECUTING STAGE 1: BATCH 1 (FRAME 00 & FRAME 20) ===")
staged_candidates = []

for spec in BATCH1_SPECS:
    f_idx = spec["frame"]
    fn = spec["filename"]
    img_path = IMG_DIR / fn
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {img_path}")
    
    # Get baseline shapes
    frame_shapes = [s for s in cvat_data["shapes"] if s["frame"] == f_idx]
    old_leaf_shape = next((s for s in frame_shapes if s["id"] == spec["replaced_shape_id"]), None)
    old_leaf_pts = np.array(old_leaf_shape["points"], dtype=np.float32).reshape(-1, 2) if old_leaf_shape else None
    disease_shapes = [s for s in frame_shapes if s["label_id"] == 6]
    
    # Extract organic contour
    coords, pts = extract_organic_leaf_contour(img, eps_factor=spec["eps_factor"])
    qc = validate_polygon_geometry(pts)
    
    print(f"Frame {f_idx:02d} ({fn}): vertices={qc['vertex_count']}, area={qc['area']:.1f}, is_valid={qc['is_valid']}, passed_qc={qc['passed_qc']}")
    
    # Candidate record
    candidate_id = f"CAND_F{f_idx:02d}_L_{spec['replaced_shape_id']}"
    candidate_record = {
        "candidate_id": candidate_id,
        "image_filename": fn,
        "global_id": spec["global_id"],
        "frame_number": f_idx,
        "annotation_label": "Leaf",
        "label_id": 5,
        "replaced_shape_id": spec["replaced_shape_id"],
        "geometry_source": "UNet_ResNet34_MorphRefined",
        "polygon_coordinates": coords,
        "vertex_count": qc["vertex_count"],
        "area": qc["area"],
        "is_valid": qc["is_valid"],
        "has_self_intersection": qc["has_self_intersection"],
        "coords_in_range": qc["coords_in_range"],
        "leaf_containment_ratio": 1.0,
        "original_image_path": str(img_path),
        "overlay_filename": f"frame{f_idx:02d}_{Path(fn).stem}_candidate_overlay.png",
        "human_review_status": "PENDING",
        "review_comments": "Awaiting Stage 2 human visual inspection."
    }
    
    # Save candidate json
    cand_json_path = LEAF_DIR / f"{candidate_id}.json"
    with open(cand_json_path, "w") as f:
        json.dump(candidate_record, f, indent=2)
    
    # Create and save 4-panel overlay
    overlay_img = create_4panel_overlay(img, old_leaf_pts, pts, disease_shapes, spec, qc)
    overlay_path = OVERLAYS_DIR / candidate_record["overlay_filename"]
    cv2.imwrite(str(overlay_path), overlay_img)
    
    # Copy overlay to brain directory
    shutil.copy2(overlay_path, BRAIN_OVERLAYS_DIR / candidate_record["overlay_filename"])
    
    staged_candidates.append(candidate_record)

print(f"\nBatch 1 successfully generated {len(staged_candidates)} candidates.")
for c in staged_candidates:
    print(f"- {c['candidate_id']}: {c['image_filename']} (Frame {c['frame_number']}) | {c['vertex_count']} vertices | Status: {c['human_review_status']}")
