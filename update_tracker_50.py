import os, shutil

os.chdir('cotton_leaf_ai')

tracker_path = 'reports/MASTER_PROJECT_TASK_TRACKER.md'
with open(tracker_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Insert into table
target_table_line = '| `TASK-PHASE12-0-6-1-INVENTORY-RECONCILIATION` | PHASE 12.0.6.1 | Disease Annotation Inventory Count and Readiness Reconciliation | P0 | `TASK-PHASE12-0-6-DISEASE-ANNOTATION-FITNESS-AUDIT` | **`COMPLETED`** | `artifacts/audit_phase11/phase12_0_6_1_count_reconciliation.md, artifacts/audit_phase11/phase12_0_6_1_readiness_crosstab.csv` |'

new_table_line = '| `TASK-PHASE12-0-6-2-FINAL-READINESS-CORRECTION` | PHASE 12.0.6.2 | Final Readiness Count, Containment and Holdout Provenance Correction | P0 | `TASK-PHASE12-0-6-1-INVENTORY-RECONCILIATION` | **`COMPLETED`** | `artifacts/audit_phase11/phase12_0_6_2_final_readiness_correction.md, artifacts/audit_phase11/phase12_0_6_1_readiness_crosstab.csv` |'

if target_table_line in content:
    content = content.replace(target_table_line, target_table_line + '\n' + new_table_line)
    print('Table line inserted successfully.')
else:
    print('ERROR: Target table line not found!')

# 2. Insert into detailed specs
target_next_spec = '### TASK-FE-MULTI-MODAL-CARDS: Frontend Multi-Modal Results Display & Experimental Badging'

new_spec = """### TASK-PHASE12-0-6-2-FINAL-READINESS-CORRECTION: Final Readiness Count, Containment and Holdout Provenance Correction
- **Phase & Priority:** PHASE 12.0.6.2 | Priority P0
- **Dependencies:** `TASK-PHASE12-0-6-1-INVENTORY-RECONCILIATION`
- **Status:** `COMPLETED`
- **Files Involved:** `artifacts/audit_phase11/phase12_0_6_2_final_readiness_correction.md`, `artifacts/audit_phase11/phase12_0_6_1_readiness_crosstab.csv`, `reports/MASTER_PROJECT_TASK_TRACKER.md`
- **Verification Method:** Comprehensive, strictly read-only documentation and methodological correction of counts, containment interpretation, holdout provenance, and checkpoint identities. Authoritatively resolved lesion correction count to exactly 8 development candidate images (`LHJ00001`, `LHJ00116`, `LHJ00172`, `LHJ00225`, `LR00140`, `LR00286`, `LR00440`, `LR00578`), correcting narrative references to seven and re-affirming the exact intersection (6 leaf-only, 4 lesion-only, 4 both, 14 distinct remediations, 21 eligible training candidates); invalidated the degenerate 99.13% aggregate containment metric due to canvas-boundary inflation across 10 leaf-invalid images (76 polygons), retaining 98.70% as the sole verified mean containment across the 155 organic polygons on the 25 valid leaf images (150/155 with >= 90% containment, isolating 5 margin serration clippings); disclosed Case B JSON shape-type metadata parsing for holdout counting while guaranteeing zero image content access, zero coordinate inspection, zero holdout edits, and zero model evaluation; corrected holdout readiness terminology to `LEAF_MASK_VERIFIED / LESION_MASK_UNVERIFIED` (9 samples) and `LEAF_MASK_CONFLICT / LESION_MASK_UNVERIFIED` (6 samples), prohibiting claims of dual-mask ready holdout data; reconciled production checkpoints to their true paths, architectures, and hashes, correcting the narrative YOLOv8 misnomer to Faster R-CNN ResNet-50 FPN (`best_faster_rcnn.pt`, 345 MB, hash `01d82b32...`); confirmed zero state mutations across model checkpoints, database (48 predictions, 5 users), and test split (322 lines); preserved Gate 4 locked in `PENDING_APPROVAL`.
- **Completion Evidence:** Audit Deliverables: `artifacts/audit_phase11/phase12_0_6_2_final_readiness_correction.md`, `artifacts/audit_phase11/phase12_0_6_1_readiness_crosstab.csv`; Verdict: `LESION COUNTS RECTIFIED (8 IMAGES), CONTAINMENT RESTRICTED TO VALID LEAF SUBSET (98.70%), HOLDOUT METADATA ACCESS DISCLOSED, FASTER R-CNN IDENTITY RESTORED`; Gate 4 preserved as `PENDING_APPROVAL`.

"""

if target_next_spec in content:
    content = content.replace(target_next_spec, new_spec + target_next_spec)
    print('Spec section inserted successfully.')
else:
    print('ERROR: Target next spec not found!')

with open(tracker_path, 'w', encoding='utf-8') as f:
    f.write(content)

# Mirror to brain
brain_path = r"C:\Users\Haker's pc\.gemini\antigravity\brain\e96a0217-11ef-4314-af5b-e8eaa45d475a\MASTER_PROJECT_TASK_TRACKER.md"
shutil.copyfile(tracker_path, brain_path)
print('Tracker updated and mirrored to brain.')
