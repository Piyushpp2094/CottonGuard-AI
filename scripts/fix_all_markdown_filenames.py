import glob

md_files = glob.glob('artifacts/**/*.md', recursive=True) + glob.glob('reports/**/*.md', recursive=True)
for p in md_files:
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
        print(f"Updated filenames in: {p}")
