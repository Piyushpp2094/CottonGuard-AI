"""
scripts/inspect_governing_documents.py
Inspects mentions of deletion manifest across the governing documents.
"""

import re
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")

docs = [
    "artifacts/audit_phase11/phase12_2_stage2_6_final_inventory_and_export_integrity.md",
    "artifacts/audit_phase11/phase12_2_stage2_7_final_provenance_and_execution_boundary_correction.md",
    "artifacts/audit_phase11/phase12_2_stage3_controlled_addition_report.md",
    "artifacts/audit_phase11/phase12_2_stage3_readonly_id_reconciliation.md",
    "reports/MASTER_PROJECT_TASK_TRACKER.md"
]

for doc_rel in docs:
    p = WORKSPACE / doc_rel
    print("=" * 80)
    print(f"DOCUMENT: {doc_rel} (Exists: {p.exists()})")
    print("=" * 80)
    if not p.exists():
        continue
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()
    
    matches = re.findall(r"(?:deletion manifest|deletions manifest|stage 4 deletion|manifest).*?(?:\.json|\.csv|\.py)?", text, re.IGNORECASE)
    for m in set(matches[:15]):
        print(f"  - {m.strip()}")
