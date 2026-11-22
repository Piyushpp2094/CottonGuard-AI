import os, shutil

os.chdir('cotton_leaf_ai')

tracker_path = 'reports/MASTER_PROJECT_TASK_TRACKER.md'
with open(tracker_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Insert into table
target_table_line = '| `TASK-PHASE12-0-6-3-CLASSIFIER-CONTAINMENT-VERIFICATION` | PHASE 12.0.6.3 | Final Classifier Architecture Identity and Containment Wording Verification | P0 | `TASK-PHASE12-0-6-2-FINAL-READINESS-CORRECTION` | **`COMPLETED`** | `artifacts/audit_phase11/phase12_0_6_3_classifier_and_containment_verification.md` |'

new_table_line = '| `TASK-PHASE12-1-REMEDIATION-PREPARATION` | PHASE 12.1 | Development Annotation Remediation Preparation | P0 | `TASK-PHASE12-0-6-3-CLASSIFIER-CONTAINMENT-VERIFICATION` | **`COMPLETED`** | `artifacts/audit_phase11/phase12_1_remediation_preparation.md` |'

if target_table_line in content:
    content = content.replace(target_table_line, target_table_line + '\n' + new_table_line)
    print('Table line inserted successfully.')
else:
    print('ERROR: Target table line not found!')

# 2. Insert into detailed specs
target_next_spec = '### TASK-FE-MULTI-MODAL-CARDS: Frontend Multi-Modal Results Display & Experimental Badging'

new_spec = """### TASK-PHASE12-1-REMEDIATION-PREPARATION: Development Annotation Remediation Preparation
- **Phase & Priority:** PHASE 12.1 | Priority P0
- **Dependencies:** `TASK-PHASE12-0-6-3-CLASSIFIER-CONTAINMENT-VERIFICATION`
- **Status:** `COMPLETED`
- **Files Involved:** `artifacts/audit_phase11/phase12_1_remediation_preparation.md`, `reports/MASTER_PROJECT_TASK_TRACKER.md`
- **Verification Method:** Comprehensive, strictly read-only execution planning for remediating the 14 non-conforming development candidate images (`global_id 50–84`, frames 00–34 of CVAT Task #6) and adjudicating the 5 sub-90% containment cases (`HGD00002`: 88.35%, `LHJ00059`: 69.75% & 32.13%, `LHJ00172`: 78.92%, `LHJ00225`: 85.51%). Established verified mapping for 6 leaf-only corrections (`BBC00001`, `CV00431`, `HL00070`, `LR00001`, `LV00028`, `LV00087`), 4 lesion-only corrections (`LHJ00172`, `LHJ00225`, `LR00286`, `LR00440`), and 4 dual corrections (`LHJ00001`, `LHJ00116`, `LR00140`, `LR00578`); documented exact CVAT Task #6 frame indices, pre-remediation backup protocol with SHA-256 baseline hashing, emergency rollback script procedure, and holdout slice isolation (frames 35–49 strictly sealed without access); drafted step-by-step annotation guidelines enforcing organic contours (leaf >= 20 pts, lesion >= 5 pts) and zero bounding boxes; formulated automated 9-point pre-acceptance validation test suite (`validate_remediated_annotations.py`); defined target 70-training / 15-dev-validation / 15-holdout dataset architecture resolving the 14-image deficit without holdout contamination; confirmed zero state mutations across model checkpoints, database (48 predictions, 5 users; read-only inspected / not modified), and test split (322 lines); preserved Gate 4 locked in `PENDING_APPROVAL`.
- **Completion Evidence:** Audit Deliverables: `artifacts/audit_phase11/phase12_1_remediation_preparation.md`; Verdict: `14-IMAGE REMEDIATION PLAN ESTABLISHED, ADJUDICATION PROTOCOL FORMALIZED, CVAT SAFEGUARDS CONFIGURED, 70-IMAGE TARGET CORPUS ARCHITECTED`; Gate 4 preserved as `PENDING_APPROVAL`.

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
