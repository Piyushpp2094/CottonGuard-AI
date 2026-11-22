"""
scripts/print_f21_adjudication_csv.py
"""
import pandas as pd

df = pd.read_csv("artifacts/annotation_staging/phase12_2/candidates_adjudication_master.csv")
f21 = df[df["frame"] == 21].sort_values("original_shape_id")

print(f"Total Frame 21 rows in CSV: {len(f21)}")
for idx, r in f21.iterrows():
    print(f"Shape {r['original_shape_id']} | CID: {r['candidate_id']} | Host: {r['host_leaf_id']} | Cont: {r['containment_percentage']}% | Decision: {r['final_decision']} | Reason: {r['reason'][:40]}")
