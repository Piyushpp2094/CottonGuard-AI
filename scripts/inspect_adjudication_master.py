"""
scripts/inspect_adjudication_master.py
Inspects candidates_adjudication_master.csv and candidates_manifest.csv.
"""

import pandas as pd
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")

p_adj = WORKSPACE / "artifacts/annotation_staging/phase12_2/candidates_adjudication_master.csv"
if p_adj.exists():
    df_adj = pd.read_csv(p_adj)
    print(f"candidates_adjudication_master.csv: {len(df_adj)} rows")
    print("Columns:", list(df_adj.columns))
    print(df_adj.head(3))

p_man = WORKSPACE / "artifacts/annotation_staging/phase12_2/candidates_manifest.csv"
if p_man.exists():
    df_man = pd.read_csv(p_man)
    print(f"\ncandidates_manifest.csv: {len(df_man)} rows")
    print("Columns:", list(df_man.columns))
    print(df_man.head(3))
