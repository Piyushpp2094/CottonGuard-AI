import json
import urllib.request

schema_url = "http://localhost:8080/api/schema/"
req = urllib.request.Request(schema_url, headers={"Accept": "application/json"})
with urllib.request.urlopen(req, timeout=10) as resp:
    data = resp.read()
    schema = json.loads(data.decode("utf-8"))
    paths = schema.get("paths", {})
    patch_resp = paths.get("/api/jobs/{id}/annotations/", {}).get("patch", {}).get("responses", {}).get("200", {})
    print("PATCH 200 response:", json.dumps(patch_resp, indent=2))
