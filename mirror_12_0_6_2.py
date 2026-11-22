import shutil, os

src = 'cotton_leaf_ai/artifacts/audit_phase11/phase12_0_6_2_final_readiness_correction.md'
brain_dir = r"C:\Users\Haker's pc\.gemini\antigravity\brain\e96a0217-11ef-4314-af5b-e8eaa45d475a"
dst = os.path.join(brain_dir, "phase12_0_6_2_final_readiness_correction.md")
shutil.copyfile(src, dst)
print("Successfully mirrored to brain.")
