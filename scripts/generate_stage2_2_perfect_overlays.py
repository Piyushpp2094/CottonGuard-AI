"""
scripts/generate_stage2_2_perfect_overlays.py
Generates authoritative 6-panel visual review packages (2400x1600) for:
- Frame 15: LHJ00001.jpg
- Frame 21: LR00140.jpg
- Frame 22: LR00286.jpg
- Frame 24: LR00578.jpg
Fully adheres to all Phase 12.2 Stage 2.2 specifications.
"""

import cv2, json, shutil
import numpy as np
import pandas as pd
from pathlib import Path

STAGE_DIR = Path("artifacts/annotation_staging/phase12_2")
OUTPUT_DIR = STAGE_DIR / "stage2_2_corrected_overlays"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BRAIN_DIR = Path(r"C:\Users\Haker's pc\.gemini\antigravity\brain\e96a0217-11ef-4314-af5b-e8eaa45d475a\phase12_2_review")
BRAIN_DIR.mkdir(parents=True, exist_ok=True)

# Load master candidates and baseline snapshot
with open(STAGE_DIR / "candidates_stage2_2_master.json") as f:
    master_cands = json.load(f)

with open("artifacts/backups/cvat_job6_stage0_pre_remediation_snapshot_20261118_205937.json") as f:
    cvat_snapshot = json.load(f)

adj_df = pd.read_csv(STAGE_DIR / "candidates_adjudication_master.csv")

# Frame metadata
frames_meta = {
    15: {"fn": "LHJ00001.jpg", "cls": "Leaf Hopper Jassids", "gid": 65},
    21: {"fn": "LR00140.jpg", "cls": "Leaf Redding", "gid": 71},
    22: {"fn": "LR00286.jpg", "cls": "Leaf Redding", "gid": 72},
    24: {"fn": "LR00578.jpg", "cls": "Leaf Redding", "gid": 74}
}

for f_idx, meta in frames_meta.items():
    fn = meta["fn"]
    img_p = Path("data/detection_consolidated_100/images") / fn
    orig = cv2.imread(str(img_p))
    h, w = orig.shape[:2]

    # -------------------------------------------------------------------------
    # Panel 1: Original Image
    # -------------------------------------------------------------------------
    p1 = orig.copy()
    cv2.rectangle(p1, (0, 0), (800, 60), (30, 30, 30), -1)
    cv2.putText(p1, f"Frame {f_idx:02d} | Global ID: {meta['gid']} | {fn}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(p1, f"Panel 1: Raw Original Image ({meta['cls']})", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 1)

    # -------------------------------------------------------------------------
    # Panel 2: Existing CVAT Annotations (Baseline Snapshot)
    # -------------------------------------------------------------------------
    p2 = orig.copy()
    cv2.rectangle(p2, (0, 0), (800, 60), (30, 30, 30), -1)
    cv2.putText(p2, f"Panel 2: Baseline CVAT Job #6 Annotations", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    f_shapes = [s for s in cvat_snapshot["shapes"] if s["frame"] == f_idx]
    rect_c = sum(1 for s in f_shapes if s["type"] == "rectangle")
    poly_c = sum(1 for s in f_shapes if s["type"] == "polygon")
    cv2.putText(p2, f"Total Shapes: {len(f_shapes)} (Rectangles: {rect_c}, Polygons: {poly_c})", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    for s in f_shapes:
        pts = np.array(s["points"]).reshape(-1, 2).astype(np.int32)
        if s["type"] == "rectangle":
            x1, y1, x2, y2 = pts[0][0], pts[0][1], pts[1][0], pts[1][1]
            cv2.rectangle(p2, (x1, y1), (x2, y2), (0, 165, 255), 1)
        elif s["label_id"] == 5: # Leaf
            cv2.polylines(p2, [pts], True, (0, 255, 0), 2)
        else: # Lesion polygon
            cv2.polylines(p2, [pts], True, (0, 0, 255), 1)

    # -------------------------------------------------------------------------
    # Panel 3: Final Proposed Leaf Polygons
    # -------------------------------------------------------------------------
    p3 = orig.copy()
    cv2.rectangle(p3, (0, 0), (800, 60), (30, 30, 30), -1)
    cv2.putText(p3, "Panel 3: Final Proposed Leaf Polygons", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    f_leaves = [c for c in master_cands if c["frame_number"] == f_idx and c["annotation_label"] == "Leaf"]
    if f_idx == 22:
        # Frame 22 uses retained CVAT leaf shape 1345
        s1345 = next(s for s in cvat_snapshot["shapes"] if s["id"] == 1345)
        pts = np.array(s1345["points"]).reshape(-1, 2).astype(np.int32)
        cv2.polylines(p3, [pts], True, (0, 255, 0), 2)
        cv2.putText(p3, f"Shape 1345: Retained Canopy Polygon (Verts: {len(pts)})", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
    else:
        leaf_desc = []
        for l_idx, lf in enumerate(f_leaves):
            pts = np.array(lf["polygon_coordinates"]).reshape(-1, 2).astype(np.int32)
            color = (0, 255, 0) if l_idx == 0 else (255, 255, 0) # Green for primary, Cyan for secondary
            cv2.polylines(p3, [pts], True, color, 2)
            c_pt = np.mean(pts, axis=0).astype(np.int32)
            cv2.putText(p3, lf["candidate_id"], (c_pt[0] - 80, c_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(p3, lf["candidate_id"], (c_pt[0] - 80, c_pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            leaf_desc.append(f"{lf['candidate_id']} ({len(pts)}v)")
        cv2.putText(p3, " | ".join(leaf_desc), (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # -------------------------------------------------------------------------
    # Panel 4: Final Proposed Lesion Polygons (Accepted vs Rejected with IDs)
    # -------------------------------------------------------------------------
    p4 = orig.copy()
    cv2.rectangle(p4, (0, 0), (800, 60), (30, 30, 30), -1)
    cv2.putText(p4, "Panel 4: Proposed Lesions & Adjudication Status", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

    f_lesions = [c for c in master_cands if c["frame_number"] == f_idx and c["annotation_label"] == "Diseased_Region"]
    app_n = sum(1 for c in f_lesions if c["adjudication_decision"] == "ACCEPT")
    rej_n = sum(1 for c in f_lesions if c["adjudication_decision"] == "REJECT")
    cv2.putText(p4, f"ACCEPTED: {app_n} (Cyan) | REJECTED: {rej_n} (Red)", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    for c in f_lesions:
        pts = np.array(c["polygon_coordinates"]).reshape(-1, 2).astype(np.int32)
        dec = c["adjudication_decision"]
        color = (255, 255, 0) if dec == "ACCEPT" else (0, 0, 255) # Cyan for Accept, Red for Reject
        cv2.polylines(p4, [pts], True, color, 2 if dec == "ACCEPT" else 1)
        
        # Draw ID at centroid
        c_pt = np.mean(pts, axis=0).astype(np.int32)
        txt = str(c["replaced_shape_id"])
        cv2.putText(p4, txt, (c_pt[0] - 10, c_pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2)
        cv2.putText(p4, txt, (c_pt[0] - 10, c_pt[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # -------------------------------------------------------------------------
    # Panel 5: Combined Final Composition (Ground Truth Segmentation Preview)
    # -------------------------------------------------------------------------
    p5 = orig.copy()
    overlay_mask = np.zeros_like(orig)
    
    # Fill leaves with green tint
    if f_idx == 22:
        s1345 = next(s for s in cvat_snapshot["shapes"] if s["id"] == 1345)
        pts = np.array(s1345["points"]).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(overlay_mask, [pts], (0, 180, 0))
    else:
        for lf in f_leaves:
            pts = np.array(lf["polygon_coordinates"]).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(overlay_mask, [pts], (0, 180, 0))

    # Fill approved lesions with red/magenta tint
    for c in f_lesions:
        if c["adjudication_decision"] == "ACCEPT":
            pts = np.array(c["polygon_coordinates"]).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(overlay_mask, [pts], (0, 0, 255))

    # Blend translucent overlay
    cv2.addWeighted(overlay_mask, 0.35, p5, 0.65, 0, p5)
    cv2.rectangle(p5, (0, 0), (800, 60), (30, 30, 30), -1)
    cv2.putText(p5, "Panel 5: Combined Ground-Truth Segmentation Preview", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 255), 2)
    cv2.putText(p5, "Leaf (Class 1, Green Tint) | Diseased_Region (Class 2, Red Tint)", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # -------------------------------------------------------------------------
    # Panel 6: Adjudication Register & Status Table (HUD)
    # -------------------------------------------------------------------------
    p6 = np.zeros((800, 800, 3), dtype=np.uint8)
    p6[:] = (20, 24, 28) # Dark elegant slate background
    
    cv2.rectangle(p6, (0, 0), (800, 50), (35, 42, 50), -1)
    cv2.putText(p6, f"Panel 6: Candidate Adjudication Register ({fn})", (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

    # Table Header
    y_off = 75
    cv2.putText(p6, "Shape ID", (15, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cv2.putText(p6, "Host Leaf", (90, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cv2.putText(p6, "Cont %", (260, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cv2.putText(p6, "Decision", (330, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cv2.putText(p6, "Reason / Evidence", (415, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cv2.line(p6, (10, y_off + 8), (790, y_off + 8), (70, 80, 90), 1)

    f_adj = adj_df[adj_df["frame"] == f_idx].sort_values("original_shape_id")
    row_y = y_off + 26
    
    # Render table rows (up to 35 rows)
    step_y = 20 if len(f_adj) > 20 else 24
    for _, r in f_adj.iterrows():
        sid = str(r["original_shape_id"])
        hl = str(r["host_leaf_id"])
        if len(hl) > 18:
            hl = hl[:17] + ".."
        cont = f"{r['containment_percentage']:.1f}%"
        dec = str(r["final_decision"])
        reason = str(r["reason"])
        if len(reason) > 42:
            reason = reason[:40] + ".."
            
        color = (0, 255, 200) if dec == "ACCEPT" else (100, 100, 255)
        
        cv2.putText(p6, sid, (15, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(p6, hl, (90, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)
        cv2.putText(p6, cont, (260, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(p6, dec, (330, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        cv2.putText(p6, reason, (415, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1)
        row_y += step_y
        if row_y > 760:
            break

    # Summary footer
    cv2.rectangle(p6, (0, 765), (800, 800), (35, 42, 50), -1)
    cv2.putText(p6, f"Frame {f_idx:02d} Summary: Approved Lesions = {app_n} | Rejected = {rej_n} | Human Status = PENDING_APPROVAL", (15, 788), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # -------------------------------------------------------------------------
    # Assemble 3x2 Master Grid (2400x1600)
    # -------------------------------------------------------------------------
    row1 = np.hstack([p1, p2, p3])
    row2 = np.hstack([p4, p5, p6])
    grid_6panel = np.vstack([row1, row2])

    out_fn = f"frame{f_idx:02d}_{Path(fn).stem}_stage2_2_authoritative_overlay.png"
    out_p = OUTPUT_DIR / out_fn
    cv2.imwrite(str(out_p), grid_6panel)
    shutil.copy(str(out_p), str(BRAIN_DIR / out_fn))
    print(f"Generated 6-panel authoritative overlay: {out_fn} (2400x1600)")

print("\nAll authoritative 6-panel overlays generated successfully.")
