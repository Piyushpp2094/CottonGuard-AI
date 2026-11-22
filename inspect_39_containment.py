import json
import pandas as pd
from pathlib import Path

csv_path = Path("artifacts/annotation_staging/phase12_2/candidates_manifest.csv")
df = pd.read_csv(csv_path)

low = df[df["leaf_containment_ratio"] < 0.90].copy()
print(f"Total low-containment candidates (< 0.90): {len(low)}")

# Group by frame
print("\nBreakdown by frame:")
print(low["frame_number"].value_counts().sort_index())

print("\nDetailed list of all 39 candidates:")
for idx, row in low.iterrows():
    print(f"CandID: {row['candidate_id']:<18} | Frame: {row['frame_number']:02d} | Img: {row['image_filename']:<14} | RepID: {row['replaced_shape_id']:<5} | Verts: {row['vertex_count']:<2} | Area: {row['area']:<7.1f} | Containment: {row['leaf_containment_ratio']*100:5.1f}%")
