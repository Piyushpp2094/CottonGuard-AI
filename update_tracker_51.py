import os, shutil

os.chdir('cotton_leaf_ai')

tracker_path = 'reports/MASTER_PROJECT_TASK_TRACKER.md'
with open(tracker_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Insert into table
target_table_line = '| `TASK-PHASE12-0-6-2-FINAL-READINESS-CORRECTION` | PHASE 12.0.6.2 | Final Readiness Count, Containment and Holdout Provenance Correction | P0 | `TASK-PHASE12-0-6-1-INVENTORY-RECONCILIATION` | **`COMPLETED`** | `artifacts/audit_phase11/phase12_0_6_2_final_readiness_correction.md, artifacts/audit_phase11/phase12_0_6_1_readiness_crosstab.csv` |'

new_table_line = '| `TASK-PHASE12-0-6-3-CLASSIFIER-CONTAINMENT-VERIFICATION` | PHASE 12.0.6.3 | Final Classifier Architecture Identity and Containment Wording Verification | P0 | `TASK-PHASE12-0-6-2-FINAL-READINESS-CORRECTION` | **`COMPLETED`** | `artifacts/audit_phase11/phase12_0_6_3_classifier_and_containment_verification.md` |'

if target_table_line in content:
    content = content.replace(target_table_line, target_table_line + '\n' + new_table_line)
    print('Table line inserted successfully.')
else:
    print('ERROR: Target table line not found!')

# 2. Insert into detailed specs
target_next_spec = '### TASK-FE-MULTI-MODAL-CARDS: Frontend Multi-Modal Results Display & Experimental Badging'

new_spec = """### TASK-PHASE12-0-6-3-CLASSIFIER-CONTAINMENT-VERIFICATION: Final Classifier Architecture Identity and Containment Wording Verification
- **Phase & Priority:** PHASE 12.0.6.3 | Priority P0
- **Dependencies:** `TASK-PHASE12-0-6-2-FINAL-READINESS-CORRECTION`
- **Status:** `COMPLETED`
- **Files Involved:** `artifacts/audit_phase11/phase12_0_6_3_classifier_and_containment_verification.md`, `reports/MASTER_PROJECT_TASK_TRACKER.md`
- **Verification Method:** Read-only state-dictionary tensor inspection, source code analysis, model registry verification, and provenance audit. Conclusively verified production classifier checkpoint (`artifacts/checkpoints/classification/best_model.pt`, SHA-256 `552baba8...`) as EfficientNet-B0: state_dict inspection revealed classifier head shape `torch.Size([7, 1280])`, stem conv `torch.Size([32, 3, 3, 3])`, final conv `torch.Size([1280, 320, 1, 1])`, and total parameters 4,058,580 (~4.06M), confirming historical EfficientNet-B0 identity and refuting the Phase 12.0.6.2 EfficientNet-B4 editorial misnomer (Cin = 1792, 19.3M params); corrected narrative descriptions of the five sub-90% containment polygons (`HGD00002`: 88.35%, `LHJ00059`: 69.75% & 32.13%, `LHJ00172`: 78.92%, `LHJ00225`: 85.51%), retracting unverified "margin serration clipping" hypotheses and reclassifying all five as `REQUIRING_MANUAL_ANNOTATION_ADJUDICATION`; formalized holdout access tier distinctions (metadata accessed via Case B; pixels not accessed, coordinates not accessed, data not modified, models not evaluated); confirmed zero state mutations across model checkpoints, database (48 predictions, 5 users), and test split (322 lines); preserved Gate 4 locked in `PENDING_APPROVAL`.
- **Completion Evidence:** Audit Deliverables: `artifacts/audit_phase11/phase12_0_6_3_classifier_and_containment_verification.md`; Verdict: `CLASSIFIER IDENTITY CONCLUDED (EFFICIENTNET-B0), CONTAINMENT WORDING CORRECTED (MANUAL ADJUDICATION REQUIRED), HOLDOUT ACCESS TIERS FORMALIZED`; Gate 4 preserved as `PENDING_APPROVAL`.

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
