import json, cv2, numpy as np
import pandas as pd
from pathlib import Path

# Load candidates
with open('artifacts/annotation_staging/phase12_2/candidates_153.json') as f:
    cands = json.load(f)

# Load baseline backup to get existing leaf polygons for frames where leaf wasn't replaced (18, 19, 22, 23)
with open('artifacts/backups/cvat_job6_stage0_pre_remediation_snapshot_20261118_205937.json') as f:
    cvat = json.load(f)

# Map frame -> leaf mask
frame_leaf_masks = {}

# Replaced leaves
for c in cands:
    if c['annotation_label'] == 'Leaf':
        pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
        m = np.zeros((800, 800), dtype=np.uint8)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        frame_leaf_masks[c['frame_number']] = m

# Non-replaced leaves from baseline
non_replaced_leaf_frames = {
    18: 1208,
    19: 1245,
    22: 1345,
    23: 1419
}

for f_idx, shape_id in non_replaced_leaf_frames.items():
    s = next(x for x in cvat['shapes'] if x['id'] == shape_id)
    pts = np.array(s['points']).reshape(-1, 2)
    m = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    frame_leaf_masks[f_idx] = m

print(f"Loaded leaf masks for {len(frame_leaf_masks)} frames.")

# Recompute exact raster intersection containment for all 143 lesion candidates
lesion_cands = [c for c in cands if c['annotation_label'] == 'Diseased_Region']
results = []

for c in lesion_cands:
    f_idx = c['frame_number']
    leaf_mask = frame_leaf_masks.get(f_idx)
    assert leaf_mask is not None, f"Missing leaf mask for frame {f_idx}"
    
    pts = np.array(c['polygon_coordinates']).reshape(-1, 2)
    l_mask = np.zeros((800, 800), dtype=np.uint8)
    cv2.fillPoly(l_mask, [np.round(pts).astype(np.int32)], 1)
    
    lesion_px = int(np.sum(l_mask == 1))
    overlap_px = int(np.sum((leaf_mask == 1) & (l_mask == 1)))
    ratio = overlap_px / lesion_px if lesion_px > 0 else 0.0
    
    results.append({
        "candidate_id": c['candidate_id'],
        "frame_number": f_idx,
        "replaced_shape_id": c['replaced_shape_id'],
        "lesion_pixels": lesion_px,
        "overlap_pixels": overlap_px,
        "containment_ratio": ratio,
        "containment_percent": round(ratio * 100, 2)
    })

df = pd.DataFrame(results)
df.to_csv('artifacts/annotation_staging/phase12_2/exact_containment_audit.csv', index=False)
print(f"Computed exact containment for {len(df)} lesion candidates.")
print("\nSummary of containment across all 143 lesions against primary leaf:")
print(f"  100% containment: {np.sum(df['containment_percent'] == 100.0)}")
print(f"  90-99.9% containment: {np.sum((df['containment_percent'] >= 90.0) & (df['containment_percent'] < 100.0))}")
print(f"  50-89.9% containment: {np.sum((df['containment_percent'] >= 50.0) & (df['containment_percent'] < 90.0))}")
print(f"  1-49.9% containment: {np.sum((df['containment_percent'] > 0.0) & (df['containment_percent'] < 50.0))}")
print(f"  0% containment: {np.sum(df['containment_percent'] == 0.0)}")

# Detailed print for Frame 21
f21_df = df[df['frame_number'] == 21]
print("\n--- Frame 21 Exact Containment Table ---")
for _, r in f21_df.iterrows():
    print(f"  {r['candidate_id']} (rep {r['replaced_shape_id']}): {r['overlap_pixels']}/{r['lesion_pixels']} px -> {r['containment_percent']:.2f}%")
