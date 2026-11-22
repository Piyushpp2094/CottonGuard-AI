"""
scripts/generate_phase9_0b_candidates.py
Phase 9.0B — High-Precision Offline SAM-Assisted Candidate Generation for CottonGuard AI.
Applies refined biological prompt sets:
- CV00211.jpg: puckered cupped margins & enations (~24.4% severity)
- HGD00148.jpg: deformed strap blades, excluding petiole & stem (~72.0% severity)
- LHJ00059.jpg: marginal hopper burn yellow halo & necrotic tips, excluding central green lamina (~25.1% severity)
- LR00573.jpg: interveinal anthocyanin red patches, strictly excluding primary/secondary green veins (~49.4% severity)
- LV00017.jpg: distinct cream/white chimera variegation sectors, excluding green parenchyma (~48.9% severity)

Generates:
1. Exact Leaf polygon and Diseased_Region polygons.
2. Individual candidate JSON files in artifacts/pilot_corrections/phase9_0b/.
3. Unified master JSON in artifacts/pilot_corrections/phase9_0b/candidate_annotations_offline_phase9_0b.json.
4. Binary and composite semantic masks in artifacts/pilot_corrections/phase9_0b/masks/.
5. Comprehensive 4-panel visual QA preview figures in artifacts/visual_qa/phase9_0b/ (and copied to brain artifact directory).
6. Per-image validation metrics and provenance log.
"""

import time
import json
import shutil
import hashlib
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

PROJECT_ROOT = Path("D:/Main_Project/cotton_leaf_ai")
weights_path = PROJECT_ROOT / "artifacts" / "models" / "sam_vit_h_4b8939.pth"
images_dir = PROJECT_ROOT / "data" / "pilot_batch_03" / "images"

out_dir = PROJECT_ROOT / "artifacts" / "pilot_corrections" / "phase9_0b"
masks_dir = out_dir / "masks"
qa_dir = PROJECT_ROOT / "artifacts" / "visual_qa" / "phase9_0b"
brain_dir = Path("C:/Users/Haker's pc/.gemini/antigravity/brain/e96a0217-11ef-4314-af5b-e8eaa45d475a")

out_dir.mkdir(parents=True, exist_ok=True)
masks_dir.mkdir(parents=True, exist_ok=True)
qa_dir.mkdir(parents=True, exist_ok=True)

# Candidate configurations with high-precision prompts
candidates_spec = [
    {
        "frame_idx": 7,
        "filename": "CV00211.jpg",
        "disease_type": "Curl Virus",
        "symptom_subtype": "Deformation",
        "split": "train",
        "group_id": "group_00461",
        "sha256": "2a33d407c3f34424120a81e3ab7ff5e6c7544301538f299ad345db5d59dba100",
        "leaf_prompts": {
            "pos": [[400, 380], [300, 300], [500, 300], [420, 500], [280, 480], [520, 480], [380, 200]],
            "neg": [[30, 30], [770, 30], [30, 770], [770, 770], [400, 30], [770, 400], [30, 400], [400, 770]]
        },
        "disease_clusters": [
            {
                "id": "P01_curled_upper_left_lobe",
                "pos": [[260, 280], [220, 340], [280, 240]],
                "neg": [[380, 380], [350, 420], [100, 100]],
                "provenance": "SAM_GENERATED"
            },
            {
                "id": "P02_curled_upper_right_lobe",
                "pos": [[480, 260], [520, 320], [450, 220]],
                "neg": [[380, 380], [350, 420], [700, 100]],
                "provenance": "SAM_GENERATED"
            },
            {
                "id": "P03_enated_mid_margin_left",
                "pos": [[180, 440], [220, 480]],
                "neg": [[350, 450], [50, 500]],
                "provenance": "SAM_GENERATED"
            },
            {
                "id": "P04_enated_mid_margin_right",
                "pos": [[560, 420], [520, 480]],
                "neg": [[400, 450], [700, 500]],
                "provenance": "SAM_GENERATED"
            },
            {
                "id": "P05_puckered_lower_lobe",
                "pos": [[380, 580], [420, 620]],
                "neg": [[380, 420], [380, 750]],
                "provenance": "SAM_GENERATED"
            }
        ]
    },
    {
        "frame_idx": 12,
        "filename": "HGD00148.jpg",
        "disease_type": "Herbicide Growth Damage",
        "symptom_subtype": "Deformation",
        "split": "train",
        "group_id": "group_01079",
        "sha256": "141efac476b6af9e943c8e8936ca90bb77df0ffe4f62aca48b50b910ed3a15d1",
        "leaf_prompts": {
            "pos": [[420, 400], [350, 280], [500, 300], [250, 420], [550, 460], [400, 580]],
            "neg": [[30, 30], [770, 30], [30, 770], [770, 770], [100, 400], [700, 400], [400, 50], [400, 750]]
        },
        "disease_clusters": [
            {
                "id": "P01_strap_blade_left",
                "pos": [[260, 340], [300, 280]],
                "neg": [[420, 540], [420, 620], [100, 100]],
                "provenance": "SAM_GENERATED"
            },
            {
                "id": "P02_strap_blade_center_apex",
                "pos": [[420, 200], [450, 160], [380, 260]],
                "neg": [[420, 540], [420, 620], [420, 50]],
                "provenance": "SAM_GENERATED"
            },
            {
                "id": "P03_strap_blade_right",
                "pos": [[560, 280], [600, 340]],
                "neg": [[420, 540], [420, 620], [750, 100]],
                "provenance": "SAM_GENERATED"
            },
            {
                "id": "P04_deformed_lateral_strap",
                "pos": [[320, 440], [480, 440]],
                "neg": [[420, 540], [420, 620], [380, 580]],
                "provenance": "SAM_GENERATED"
            }
        ]
    },
    {
        "frame_idx": 16,
        "filename": "LHJ00059.jpg",
        "disease_type": "Leaf Hopper Jassids",
        "symptom_subtype": "Chlorosis",
        "split": "train",
        "group_id": "group_01270",
        "sha256": "a6bd64f3f12d1042fb2fa274f7fcbdd2a9d17842d5f4b516fd52946a72130dc4",
        "leaf_prompts": {
            "pos": [[400, 400], [300, 300], [500, 300], [250, 500], [550, 500], [400, 600]],
            "neg": [[30, 30], [770, 30], [30, 770], [770, 770], [50, 400], [750, 400]]
        },
        "disease_clusters": [
            {
                "id": "P01_marginal_burn_left_lobe",
                "pos": [[120, 360], [140, 420], [130, 480]],
                "neg": [
                    [400, 300], [400, 350], [400, 400], [400, 450], [400, 500],
                    [350, 300], [350, 350], [350, 400], [350, 450], [350, 500],
                    [300, 350], [300, 400], [300, 450], [250, 380], [50, 50]
                ],
                "provenance": "SAM_GENERATED"
            },
            {
                "id": "P02_marginal_burn_apex_lobe",
                "pos": [[380, 120], [420, 120], [400, 150]],
                "neg": [
                    [400, 300], [400, 350], [400, 400], [400, 250], [400, 50],
                    [350, 250], [450, 250]
                ],
                "provenance": "SAM_GENERATED"
            },
            {
                "id": "P03_marginal_burn_right_lobe",
                "pos": [[660, 360], [680, 420], [670, 480]],
                "neg": [
                    [400, 300], [400, 350], [400, 400], [400, 450], [400, 500],
                    [450, 300], [450, 350], [450, 400], [450, 450], [450, 500],
                    [500, 350], [500, 400], [500, 450], [550, 380], [750, 50]
                ],
                "provenance": "SAM_GENERATED"
            },
            {
                "id": "P04_marginal_burn_basal_lobes",
                "pos": [[240, 680], [560, 680], [380, 720], [420, 720]],
                "neg": [
                    [400, 400], [400, 450], [400, 500], [400, 550], [400, 600],
                    [350, 550], [450, 550], [400, 770]
                ],
                "provenance": "SAM_GENERATED"
            }
        ]
    },
    {
        "frame_idx": 44,
        "filename": "LR00573.jpg",
        "disease_type": "Leaf Redding",
        "symptom_subtype": "Chlorosis",
        "split": "validation",
        "group_id": "group_02005",
        "sha256": "c7a05968f3f6cca3caffefea174a68d1133a08144f3a2e812d667c9e2aba42ec",
        "leaf_prompts": {
            "pos": [[400, 400], [280, 320], [520, 320], [320, 520], [480, 520], [400, 250]],
            "neg": [[30, 30], [770, 30], [30, 770], [770, 770], [400, 40], [760, 400], [40, 400], [400, 760]]
        },
        "disease_clusters": [
            {
                "id": "P01_interveinal_upper_left",
                "pos": [[280, 230], [320, 210], [260, 260]],
                "neg": [[400, 300], [400, 400], [320, 280], [360, 320], [200, 200]],
                "provenance": "SAM_GENERATED"
            },
            {
                "id": "P02_interveinal_upper_right",
                "pos": [[500, 210], [540, 230], [460, 240]],
                "neg": [[400, 300], [400, 400], [480, 280], [440, 320], [600, 200]],
                "provenance": "SAM_GENERATED"
            },
            {
                "id": "P03_interveinal_mid_left",
                "pos": [[200, 340], [230, 380], [170, 380]],
                "neg": [[400, 400], [280, 300], [50, 400]],
                "provenance": "SAM_GENERATED"
            },
            {
                "id": "P04_interveinal_mid_right",
                "pos": [[600, 340], [570, 380], [630, 380]],
                "neg": [[400, 400], [520, 300], [750, 400]],
                "provenance": "SAM_GENERATED"
            },
            {
                "id": "P05_interveinal_lower_lobes",
                "pos": [[320, 500], [480, 500], [350, 540], [450, 540]],
                "neg": [[400, 400], [400, 500], [400, 580]],
                "provenance": "SAM_GENERATED"
            }
        ]
    },
    {
        "frame_idx": 45,
        "filename": "LV00017.jpg",
        "disease_type": "Leaf Variegation",
        "symptom_subtype": "Chlorosis",
        "split": "validation",
        "group_id": "group_02027",
        "sha256": "5109993408571b545d8d7ad6ee92d69fab584e6bdbed575a12d82d9cf07a3035",
        "leaf_prompts": {
            "pos": [[400, 400], [300, 300], [520, 300], [300, 520], [520, 520], [400, 220]],
            "neg": [[30, 30], [770, 30], [30, 770], [770, 770], [400, 30], [770, 400], [30, 400], [400, 770]]
        },
        "disease_clusters": [
            {
                "id": "P01_variegated_chimera_upper_left",
                "pos": [[320, 240], [360, 200], [300, 280]],
                "neg": [[350, 480], [350, 520], [250, 550], [250, 200], [50, 50]],
                "provenance": "SAM_GENERATED"
            },
            {
                "id": "P02_variegated_chimera_upper_right",
                "pos": [[480, 220], [520, 260], [450, 260]],
                "neg": [[350, 480], [350, 520], [450, 500], [550, 450], [750, 50]],
                "provenance": "SAM_GENERATED"
            },
            {
                "id": "P03_variegated_chimera_center_blade",
                "pos": [[350, 350], [350, 400], [250, 400]],
                "neg": [[350, 480], [350, 550], [250, 550], [450, 500], [150, 450]],
                "provenance": "SAM_GENERATED"
            },
            {
                "id": "P04_variegated_chimera_lateral_flanks",
                "pos": [[200, 440], [600, 440]],
                "neg": [[350, 500], [350, 550], [400, 100]],
                "provenance": "SAM_GENERATED"
            }
        ]
    }
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading SAM ViT-H predictor on {device}...")
t0 = time.time()
sam = sam_model_registry["vit_h"](checkpoint=str(weights_path))
sam.to(device=device)
predictor = SamPredictor(sam)
print(f"SAM loaded in {time.time()-t0:.2f}s")

master_validation_results = {}
unified_candidate_payload = []

for cand in candidates_spec:
    fname = cand["filename"]
    fidx = cand["frame_idx"]
    cname = cand["disease_type"]
    subtype = cand["symptom_subtype"]
    split = cand["split"]
    expected_sha = cand["sha256"]
    
    print(f"\n==========================================")
    print(f"Processing Frame {fidx:02d}: {fname} ({cname})")
    print(f"==========================================")
    
    img_path = images_dir / fname
    with open(img_path, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()
    assert actual_sha == expected_sha, f"SHA mismatch on {fname}"
    
    img_bgr = cv2.imread(str(img_path))
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    predictor.set_image(img_rgb)
    
    # 1. Segment Leaf
    lp = cand["leaf_prompts"]["pos"]
    ln = cand["leaf_prompts"]["neg"]
    l_coords = np.array(lp + ln, dtype=np.float32)
    l_labels = np.array([1]*len(lp) + [0]*len(ln), dtype=np.int32)
    
    l_masks, l_scores, _ = predictor.predict(point_coords=l_coords, point_labels=l_labels, multimask_output=False)
    raw_leaf_mask = l_masks[0].astype(np.uint8)
    leaf_sam_score = float(l_scores[0])
    
    # Approximate polygon
    cnts_leaf, _ = cv2.findContours(raw_leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_leaf_cnt = max(cnts_leaf, key=cv2.contourArea)
    approx_leaf = cv2.approxPolyDP(main_leaf_cnt, 1.2, True)
    
    # Rasterized Leaf Mask from polygon
    leaf_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(leaf_mask, [approx_leaf], 1)
    leaf_area_px = int(np.count_nonzero(leaf_mask))
    leaf_points_flat = [float(coord) for pt in approx_leaf.reshape(-1, 2) for coord in pt]
    
    # 2. Segment Disease Regions
    combined_disease_mask = np.zeros((h, w), dtype=np.uint8)
    disease_shapes = []
    cluster_records = []
    
    # For Leaf Redding (LR00573), compute green veins exclusion mask
    if fname == "LR00573.jpg":
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        vein_mask = cv2.inRange(hsv, (35, 40, 30), (85, 255, 200))
        vein_mask = cv2.bitwise_and(vein_mask, leaf_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_OPEN, kernel)
    else:
        vein_mask = None
        
    for cluster in cand["disease_clusters"]:
        cid = cluster["id"]
        cp = cluster["pos"]
        cn = cluster["neg"]
        coords_d = np.array(cp + cn, dtype=np.float32)
        labels_d = np.array([1]*len(cp) + [0]*len(cn), dtype=np.int32)
        
        d_masks, d_scores, _ = predictor.predict(point_coords=coords_d, point_labels=labels_d, multimask_output=False)
        d_mask = d_masks[0].astype(np.uint8)
        d_score = float(d_scores[0])
        
        # Constrain strictly to Leaf
        d_mask = cv2.bitwise_and(d_mask, leaf_mask)
        
        # Exclude veins if applicable
        if vein_mask is not None:
            d_mask[vein_mask > 0] = 0
            
        cnts_d, _ = cv2.findContours(d_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        extracted_polys = 0
        for cd in cnts_d:
            area_cnt = cv2.contourArea(cd)
            if area_cnt >= 200: # Biological threshold: omit microscopic noise specks
                app_d = cv2.approxPolyDP(cd, 1.2, True)
                if len(app_d) >= 3:
                    extracted_polys += 1
                    # Rasterize polygon into combined mask
                    poly_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(poly_mask, [app_d], 1)
                    combined_disease_mask = cv2.bitwise_or(combined_disease_mask, poly_mask)
                    
                    pts_d_flat = [float(coord) for pt in app_d.reshape(-1, 2) for coord in pt]
                    shape_entry = {
                        "type": "polygon",
                        "label": "Diseased_Region",
                        "frame": fidx,
                        "occluded": False,
                        "points": pts_d_flat,
                        "z_order": 0,
                        "attributes": [
                            {"name": "disease_type", "value": cname},
                            {"name": "symptom_subtype", "value": subtype}
                        ],
                        "provenance": cluster["provenance"],
                        "cluster_id": cid,
                        "sam_score": round(d_score, 4),
                        "raster_area_px": int(np.count_nonzero(poly_mask))
                    }
                    disease_shapes.append(shape_entry)
                    
        cluster_records.append({
            "cluster_id": cid,
            "pos_prompts": cp,
            "neg_prompts": cn,
            "sam_score": round(d_score, 4),
            "polygons_extracted": extracted_polys
        })
        
    disease_area_px = int(np.count_nonzero(combined_disease_mask))
    severity_pct = (disease_area_px / leaf_area_px) * 100.0 if leaf_area_px > 0 else 0.0
    
    print(f"Leaf Area: {leaf_area_px:,} px | Disease Area: {disease_area_px:,} px | Severity: {severity_pct:.4f}%")
    print(f"Disease Polygons: {len(disease_shapes)} | Leaf Vertices: {len(approx_leaf)}")
    
    # 3. Save Binary & Composite Semantic Masks
    cv2.imwrite(str(masks_dir / f"mask_leaf_{fname.replace('.jpg', '.png')}"), leaf_mask * 255)
    cv2.imwrite(str(masks_dir / f"mask_disease_{fname.replace('.jpg', '.png')}"), combined_disease_mask * 255)
    
    # Composite: 0=background, 1=healthy leaf lamina, 2=diseased region
    composite_mask = np.zeros((h, w), dtype=np.uint8)
    composite_mask[leaf_mask > 0] = 1
    composite_mask[combined_disease_mask > 0] = 2
    cv2.imwrite(str(masks_dir / f"mask_composite_{fname.replace('.jpg', '.png')}"), composite_mask)
    
    # 4. Construct CVAT-Compatible Shapes Array
    leaf_shape_entry = {
        "type": "polygon",
        "label": "Leaf",
        "frame": fidx,
        "occluded": False,
        "points": leaf_points_flat,
        "z_order": 0,
        "attributes": [],
        "provenance": "SAM_GENERATED",
        "sam_score": round(leaf_sam_score, 4),
        "raster_area_px": leaf_area_px
    }
    
    frame_payload = {
        "frame_idx": fidx,
        "filename": fname,
        "disease_type": cname,
        "symptom_subtype": subtype,
        "split": split,
        "group_id": cand["group_id"],
        "sha256": expected_sha,
        "dimensions": [w, h],
        "leaf_area_px": leaf_area_px,
        "disease_area_px": disease_area_px,
        "severity_pct": round(severity_pct, 4),
        "leaf_polygon": leaf_shape_entry,
        "disease_polygons": disease_shapes,
        "polygon_count": {
            "leaf": 1,
            "diseased_region": len(disease_shapes),
            "total": 1 + len(disease_shapes)
        },
        "clusters": cluster_records,
        "uncertain_regions": [
            {
                "description": f"Boundary transition between {subtype} and healthy lamina",
                "recommended_action": "Verify in CVAT 400% zoom before final acceptance"
            }
        ]
    }
    
    # Save per-image JSON
    per_img_json_path = out_dir / f"{fname.replace('.jpg', '_candidate_annotations.json')}"
    with open(per_img_json_path, "w") as f:
        json.dump(frame_payload, f, indent=2)
        
    unified_candidate_payload.append(frame_payload)
    master_validation_results[fname] = frame_payload
    
    # 5. Generate 4-Panel Visual QA Preview Figure
    # Panel 1: Original RGB Image
    p1 = img_bgr.copy()
    cv2.putText(p1, f"Frame {fidx}: {fname} ({cname})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(p1, f"Split: {split} | {w}x{h}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Panel 2: Anatomical Leaf Boundary Overlay
    p2 = img_bgr.copy()
    cv2.drawContours(p2, [approx_leaf], -1, (0, 255, 0), 2)
    for pt in lp:
        cv2.circle(p2, (int(pt[0]), int(pt[1])), 6, (0, 255, 0), -1)
        cv2.circle(p2, (int(pt[0]), int(pt[1])), 6, (0, 0, 0), 1)
    for pt in ln:
        cv2.circle(p2, (int(pt[0]), int(pt[1])), 6, (0, 0, 255), -1)
        cv2.circle(p2, (int(pt[0]), int(pt[1])), 6, (255, 255, 255), 1)
    cv2.putText(p2, f"Leaf Polygon (Vertices: {len(approx_leaf)})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(p2, f"Leaf Area: {leaf_area_px:,} px (SAM score: {leaf_sam_score:.4f})", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    # Panel 3: Diseased_Region Overlay
    p3 = img_bgr.copy()
    red_layer = np.zeros_like(p3)
    red_layer[combined_disease_mask > 0] = [0, 0, 255]
    p3 = cv2.addWeighted(p3, 0.75, red_layer, 0.25, 0)
    for shape in disease_shapes:
        pts = np.array(shape["points"]).reshape(-1, 2).astype(np.int32)
        cv2.polylines(p3, [pts], True, (0, 0, 255), 2)
    for cluster in cand["disease_clusters"]:
        for pt in cluster["pos"]:
            cv2.circle(p3, (int(pt[0]), int(pt[1])), 5, (0, 255, 255), -1)
            cv2.circle(p3, (int(pt[0]), int(pt[1])), 5, (0, 0, 0), 1)
        for pt in cluster["neg"]:
            cv2.circle(p3, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
            cv2.circle(p3, (int(pt[0]), int(pt[1])), 5, (255, 255, 255), 1)
    cv2.putText(p3, f"Diseased_Region ({cname} - {subtype})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(p3, f"Polygons: {len(disease_shapes)} | Disease Area: {disease_area_px:,} px", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    # Panel 4: Composite Multi-Class Segmentation & Severity Breakdown
    p4 = np.zeros_like(img_bgr)
    p4[leaf_mask > 0] = [34, 139, 34]       # Forest green for healthy lamina
    p4[combined_disease_mask > 0] = [0, 0, 220] # Crimson red for diseased region
    p4 = cv2.addWeighted(img_bgr, 0.4, p4, 0.6, 0)
    cv2.polylines(p4, [approx_leaf], True, (0, 255, 0), 2)
    for shape in disease_shapes:
        pts = np.array(shape["points"]).reshape(-1, 2).astype(np.int32)
        cv2.polylines(p4, [pts], True, (0, 255, 255), 1)
    cv2.putText(p4, f"Composite Segmentation (Severity: {severity_pct:.4f}%)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(p4, f"Green=Healthy Lamina | Red=Pathological Tissue", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    
    # Assemble 2x2 grid (1600 x 1600)
    top_grid = np.hstack([p1, p2])
    bot_grid = np.hstack([p3, p4])
    qa_grid = np.vstack([top_grid, bot_grid])
    
    qa_fig_path = qa_dir / f"qa_phase9_0b_{fname.replace('.jpg', '.png')}"
    cv2.imwrite(str(qa_fig_path), qa_grid)
    
    # Also copy to brain artifact directory
    brain_fig_path = brain_dir / f"qa_phase9_0b_{fname.replace('.jpg', '.png')}"
    shutil.copyfile(qa_fig_path, brain_fig_path)
    print(f"Visual QA figure saved: {qa_fig_path} and copied to brain")

# Save unified candidate payload JSON
unified_json_path = out_dir / "candidate_annotations_offline_phase9_0b.json"
with open(unified_json_path, "w") as f:
    json.dump(unified_candidate_payload, f, indent=2)

print(f"\nUnified candidate payload saved: {unified_json_path}")

# Print summary table
print("\n" + "="*80)
print(f"{'Frame':<6} {'Filename':<14} {'Class':<24} {'Leaf px':<10} {'Dis px':<10} {'Severity':<10} {'Polys':<6}")
print("="*80)
for cand in unified_candidate_payload:
    fidx = cand["frame_idx"]
    fname = cand["filename"]
    cname = cand["disease_type"]
    lpx = cand["leaf_area_px"]
    dpx = cand["disease_area_px"]
    sev = cand["severity_pct"]
    pcount = cand["polygon_count"]["diseased_region"]
    print(f"{fidx:<6} {fname:<14} {cname:<24} {lpx:<10,} {dpx:<10,} {sev:<9.4f}% {pcount:<6}")
print("="*80)
