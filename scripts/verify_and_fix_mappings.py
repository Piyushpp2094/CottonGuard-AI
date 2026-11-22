import pandas as pd
import json

# Check CSV
csv_p = 'artifacts/annotation_staging/phase12_2/candidates_adjudication_master.csv'
df = pd.read_csv(csv_p)
for f in [18, 19, 26, 28]:
    f_df = df[df['frame'] == f]
    print(f"Frame {f}: count={len(f_df)}, unique filenames={f_df['filename'].unique().tolist()}")

# Check JSON
json_p = 'artifacts/annotation_staging/phase12_2/candidates_stage2_2_master.json'
with open(json_p) as f:
    data = json.load(f)

for f in [18, 19, 26, 28]:
    f_cands = [c for c in data if c['frame_number'] == f]
    fns = set(c.get('image_filename') for c in f_cands)
    print(f"JSON Frame {f}: count={len(f_cands)}, filenames={fns}")
