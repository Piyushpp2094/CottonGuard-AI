"""
scripts/inspect_schema_paths.py
Finds annotations paths in schema.
"""

import json
import urllib.request

schema_url = "http://localhost:8080/api/schema/"
req = urllib.request.Request(schema_url, headers={"Accept": "application/json"})
with urllib.request.urlopen(req, timeout=10) as resp:
    schema = json.loads(resp.read().decode("utf-8"))

for p in schema.get("paths", {}).keys():
    if "annotations" in p:
        print(p, list(schema["paths"][p].keys()))
        for method, spec in schema["paths"][p].items():
            print(f"  {method.upper()}: {spec.get('summary')} | params: {[x.get('name') for x in spec.get('parameters', [])]}")
