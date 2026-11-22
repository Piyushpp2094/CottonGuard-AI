"""
scripts/inspect_cvat_delete_schema.py
Inspects CVAT OpenAPI schema for DELETE and PATCH ?action=delete on /api/jobs/{id}/annotations/.
"""

import json
import urllib.request

schema_url = "http://localhost:8080/api/schema/"
req = urllib.request.Request(schema_url, headers={"Accept": "application/json"})
with urllib.request.urlopen(req, timeout=10) as resp:
    data = resp.read()
    schema = json.loads(data.decode("utf-8"))
    paths = schema.get("paths", {})
    job_ann = paths.get("/api/jobs/{id}/annotations/", {})
    print("Methods on /api/jobs/{id}/annotations/:", list(job_ann.keys()))
    
    # 1. DELETE method
    del_spec = job_ann.get("delete", {})
    print("\n--- DELETE method ---")
    print("Summary:", del_spec.get("summary"))
    print("Description:", del_spec.get("description"))
    print("Parameters:", [p.get("name") for p in del_spec.get("parameters", [])])
    
    # 2. PATCH method
    patch_spec = job_ann.get("patch", {})
    print("\n--- PATCH method ---")
    print("Summary:", patch_spec.get("summary"))
    print("Description:", patch_spec.get("description"))
    print("Action enum:", patch_spec.get("parameters", []))
    schemas = schema.get("components", {}).get("schemas", {})
    pldr = schemas.get("PatchedLabeledDataRequest", {})
    print("PatchedLabeledDataRequest:", json.dumps(pldr, indent=2))
