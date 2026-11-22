import os, json, csv, shutil

os.chdir('cotton_leaf_ai')

with open('data/annotation_pilot/phase9_5/candidate_annotations_task6.json', 'r') as f:
    cvat_data = json.load(f)

with open('artifacts/audit_phase11/phase12_0_5_annotation_inventory.csv', 'r') as f:
    inv = list(csv.DictReader(f))

frames = {}
for shape in cvat_data.get('shapes', []):
    frame = shape.get('frame')
    frames.setdefault(frame, []).append(shape)

# Lesion invalid images (8 images identified in Phase 12.0.6 having 4-vertex bbox lesions without organic contours)
lesion_invalid_images = {
    'LHJ00001.jpg', 'LHJ00116.jpg', 'LHJ00172.jpg', 'LHJ00225.jpg',
    'LR00140.jpg', 'LR00286.jpg', 'LR00440.jpg', 'LR00578.jpg'
}

# Weak supervision only (images with 0 organic disease polygons)
weak_supervision_images = {'LHJ00001.jpg', 'LR00440.jpg'}

rows = []
for r in inv:
    gid = int(r['global_id'])
    if gid >= 85:
        continue # Strictly dev candidates
    
    filename = r['filename']
    category = r['disease_category']
    is_healthy = (r['healthy_control_flag'] == 'True')
    status_12_0_5 = r['final_readiness_status']
    
    leaf_mask_valid = (status_12_0_5 == 'READY_FOR_DUAL_MASK')
    leaf_correction_required = not leaf_mask_valid
    
    lesion_mask_valid = (filename not in lesion_invalid_images)
    lesion_contour_required = not lesion_mask_valid
    
    weak_supervision_only = (filename in weak_supervision_images)
    holdout_status = 'DEVELOPMENT_CANDIDATE'
    final_training_eligibility = (leaf_mask_valid and lesion_mask_valid)
    
    rows.append({
        'global_id': gid,
        'batch': 'Batch 03',
        'image_id': filename,
        'disease_label': category,
        'leaf_mask_valid': leaf_mask_valid,
        'lesion_mask_valid': lesion_mask_valid,
        'leaf_correction_required': leaf_correction_required,
        'lesion_contour_required': lesion_contour_required,
        'weak_supervision_only': weak_supervision_only,
        'holdout_status': holdout_status,
        'final_training_eligibility': final_training_eligibility
    })

fieldnames = [
    'global_id', 'batch', 'image_id', 'disease_label',
    'leaf_mask_valid', 'lesion_mask_valid',
    'leaf_correction_required', 'lesion_contour_required',
    'weak_supervision_only', 'holdout_status',
    'final_training_eligibility'
]

out_path = 'artifacts/audit_phase11/phase12_0_6_1_readiness_crosstab.csv'
with open(out_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Successfully generated {out_path} with {len(rows)} rows.")

# Also mirror to brain
brain_dir = r"C:\Users\Haker's pc\.gemini\antigravity\brain\e96a0217-11ef-4314-af5b-e8eaa45d475a"
brain_path = os.path.join(brain_dir, "phase12_0_6_1_readiness_crosstab.csv")
shutil.copyfile(out_path, brain_path)
print(f"Successfully mirrored to {brain_path}")
