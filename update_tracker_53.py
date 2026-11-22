import os, shutil

os.chdir('cotton_leaf_ai')

tracker_path = 'reports/MASTER_PROJECT_TASK_TRACKER.md'
with open(tracker_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Insert into table
target_table_line = '| `TASK-PHASE12-1-REMEDIATION-PREPARATION` | PHASE 12.1 | Development Annotation Remediation Preparation | P0 | `TASK-PHASE12-0-6-3-CLASSIFIER-CONTAINMENT-VERIFICATION` | **`COMPLETED`** | `artifacts/audit_phase11/phase12_1_remediation_preparation.md` |'

new_table_line = '| `TASK-PHASE12-1-1-GOVERNANCE-CORRECTION` | PHASE 12.1.1 | Remediation Protocol Consistency and Governance Correction | P0 | `TASK-PHASE12-1-REMEDIATION-PREPARATION` | **`COMPLETED`** | `artifacts/audit_phase11/phase12_1_1_remediation_governance_correction.md, artifacts/audit_phase11/phase12_1_remediation_preparation.md` |'

if target_table_line in content:
    content = content.replace(target_table_line, target_table_line + '\n' + new_table_line)
    print('Table line inserted successfully.')
else:
    print('ERROR: Target table line not found!')

# 2. Insert into detailed specs
target_next_spec = '### TASK-FE-MULTI-MODAL-CARDS: Frontend Multi-Modal Results Display & Experimental Badging'

new_spec = """### TASK-PHASE12-1-1-GOVERNANCE-CORRECTION: Remediation Protocol Consistency and Governance Correction
- **Phase & Priority:** PHASE 12.1.1 | Priority P0
- **Dependencies:** `TASK-PHASE12-1-REMEDIATION-PREPARATION`
- **Status:** `COMPLETED`
- **Files Involved:** `artifacts/audit_phase11/phase12_1_1_remediation_governance_correction.md`, `artifacts/audit_phase11/phase12_1_remediation_preparation.md`, `reports/MASTER_PROJECT_TASK_TRACKER.md`
- **Verification Method:** Documentation and governance correction addendum resolving execution inconsistencies in the Phase 12.1 preparation report. Reconciled leaf vertex standards to a unified hard minimum of >= 15 vertices (rejecting < 15, recommending >= 20, prioritizing true organic curvature over raw count); replaced unconditional pseudo-mask area rejections with intelligent warning indicators (`HIGH_AREA_EDGE_TOUCHING_WARNING`), allowing genuine close-up/macro leaves to be approved upon visual inspection; architected a Group-Aware Stratified K-Fold Cross-Validation protocol on the 70-image training corpus where multi-image duplicate groups from `duplicate_groups.csv` remain atomic within single folds, formally accepting variable fold sizes (13-15 images) to prevent train/val leakage; clarified holdout access history (high-level JSON metadata accessed in Phase 12.0.6.1; pixels/geometry not accessed, data not modified, models not evaluated); established a unified baseline backup protocol with programmatic SHA-256 fingerprinting over frames 35-49 without slicing or exposing holdout coordinates; formalized the 90% containment threshold into a 4-tier review gate (Auto Pass, Review Trigger, Redraw, and Documented Human-Approved Biological Exception); confirmed technical preparation is COMPLETE while execution authorization remains PENDING_APPROVAL and CVAT edits are NOT AUTHORIZED; confirmed zero state mutations across checkpoints, database (48 predictions, 5 users; read-only inspected / not modified), and test split (322 lines); preserved Gate 4 locked in `PENDING_APPROVAL`.
- **Completion Evidence:** Audit Deliverables: `artifacts/audit_phase11/phase12_1_1_remediation_governance_correction.md`, `artifacts/audit_phase11/phase12_1_remediation_preparation.md`; Verdict: `VERTEX POLICY UNIFIED (HARD MIN 15), PSEUDO-MASK WARNING TRIGGERS ESTABLISHED, GROUP-AWARE CV PROTOCOL ARCHITECTED, 4-TIER CONTAINMENT GATE FORMALIZED`; Gate 4 preserved as `PENDING_APPROVAL`.

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
