import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path("d:/Main_Project/cotton_leaf_ai")
p1_path = PROJECT_ROOT / "artifacts" / "manifests" / "pilot_batch_manifest.csv"
p2_path = PROJECT_ROOT / "artifacts" / "manifests" / "batch_02_manifest.csv"
combined_path = PROJECT_ROOT / "artifacts" / "manifests" / "combined_pilot_manifest.csv"

df1 = pd.read_csv(p1_path)
df2 = pd.read_csv(p2_path)
df_combined = pd.concat([df1, df2], ignore_index=True)
df_combined.to_csv(combined_path, index=False)

print(f"Generated {combined_path}")
print(f"Total images: {len(df_combined)}")
print("Class breakdown:\n", df_combined["class_label"].value_counts())
print("Split breakdown:\n", df_combined["split"].value_counts())
assert len(df_combined) == 50, f"Expected 50 images, got {len(df_combined)}"
assert "test" not in df_combined["split"].values, "Test split found in combined manifest!"
