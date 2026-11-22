import glob

brain_files = glob.glob(r"C:\Users\Haker's pc\.gemini\antigravity\brain\e96a0217-11ef-4314-af5b-e8eaa45d475a\*.md")
for p in brain_files:
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    orig = content
    content = content.replace('LHJ00145.jpg', 'LHJ00172.jpg')
    content = content.replace('LHJ00145', 'LHJ00172')
    content = content.replace('LHJ00344.jpg', 'LHJ00225.jpg')
    content = content.replace('LHJ00344', 'LHJ00225')
    content = content.replace('LV00001.jpg', 'LV00028.jpg')
    content = content.replace('LV00120.jpg', 'LV00087.jpg')
    if content != orig:
        with open(p, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated filenames in brain file: {p}")
