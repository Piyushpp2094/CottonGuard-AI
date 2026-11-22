import shutil, os

src = 'cotton_leaf_ai/reports/MASTER_PROJECT_TASK_TRACKER.md'
brain_dir = r"C:\Users\Haker's pc\.gemini\antigravity\brain\e96a0217-11ef-4314-af5b-e8eaa45d475a"
dst = os.path.join(brain_dir, "MASTER_PROJECT_TASK_TRACKER.md")
shutil.copyfile(src, dst)
print("Successfully mirrored tracker to brain.")
