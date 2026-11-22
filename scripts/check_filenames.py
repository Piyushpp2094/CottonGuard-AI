import pandas as pd
df = pd.read_csv('artifacts/manifests/consolidated_100_annotation_manifest.csv')
t6 = df[df['cvat_task_id'] == 6].sort_values('global_id')
for f in [26, 28]:
    r = t6[t6['global_id'] == 50 + f]
    print(f"Frame {f}: global_id={50+f}, filename={r.iloc[0]['filename']}, class={r.iloc[0]['class_label']}")
