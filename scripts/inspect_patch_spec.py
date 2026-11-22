"""
scripts/inspect_patch_spec.py
Prints PATCH spec.
"""

import json
import urllib.request

schema_url = "http://localhost:8080/api/schema/"
req = urllib.request.Request(schema_url, headers={"Accept": "application/json"})
with urllib.request.urlopen(req, timeout=10) as resp:
    schema = json.loads(resp.read().decode("utf-8"))

spec = schema["paths"]["/api/jobs/{id}/annotations/"]["patch"]
print("Summary:", spec.get("summary"))
print("Description:", spec.get("description"))
print("Parameters:", json.dumps(spec.get("parameters"), indent=2))
print("RequestBody:", json.dumps(spec.get("requestBody"), indent=2))
