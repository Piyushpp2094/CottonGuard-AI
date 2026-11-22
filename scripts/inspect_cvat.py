"""
CVAT Inspection Utility for Cotton Leaf AI (Phase 6).

Safely queries the local CVAT Community Edition REST API at http://localhost:8080
to verify:
1. Whether CVAT service is running and reachable.
2. Existing projects and tasks (specifically 'Cotton_Leaf_Segmentation_Pilot').
3. Whether task-level label inheritance includes 'Leaf' and 'Diseased_Region'
   with mutable attribute 'disease_type' (6 disease options).

Does not modify CVAT's database or containers.
"""

import argparse
import getpass
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

EXPECTED_DISEASE_TYPES = {
    "Bacterial Blight",
    "Curl Virus",
    "Herbicide Growth Damage",
    "Leaf Hopper Jassids",
    "Leaf Redding",
    "Leaf Variegation",
}


def inspect_cvat_server(
    base_url: str = "http://localhost:8080",
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> Dict[str, Any]:
    url = base_url.rstrip("/")
    report: Dict[str, Any] = {
        "cvat_url": url,
        "reachable": False,
        "projects": [],
        "pilot_task_found": False,
        "pilot_task_details": None,
        "label_verification": {},
    }

    try:
        resp = httpx.get(f"{url}/api/server/about", timeout=4.0)
        if resp.status_code >= 500:
            report["error"] = f"CVAT returned HTTP {resp.status_code}"
            return report
        report["reachable"] = True
        report["server_info"] = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    except Exception as exc:
        report["error"] = f"Could not reach CVAT at {url}: {exc}. Ensure Docker Desktop and CVAT containers are started."
        return report

    if not username or not password:
        report["auth_note"] = "No CVAT credentials provided (--username / --password). Public reachability verified only."
        return report

    with httpx.Client(base_url=url, timeout=10.0) as client:
        login_res = client.post("/api/auth/login", json={"username": username, "password": password})
        if login_res.status_code not in (200, 201):
            report["error"] = f"CVAT authentication failed ({login_res.status_code}): {login_res.text[:200]}"
            return report
        key = login_res.json().get("key")
        if key:
            client.headers.update({"Authorization": f"Token {key}"})

        # Inspect projects
        proj_res = client.get("/api/projects")
        if proj_res.status_code == 200:
            projects = proj_res.json().get("results", [])
            report["projects"] = [{"id": p["id"], "name": p["name"], "tasks": p.get("tasks", {}).get("count")} for p in projects]

        # Inspect tasks
        tasks_res = client.get("/api/tasks", params={"search": "Cotton_Leaf"})
        tasks = tasks_res.json().get("results", []) if tasks_res.status_code == 200 else []
        pilot = next((t for t in tasks if "Cotton_Leaf_Segmentation_Pilot" in t.get("name", "")), tasks[0] if tasks else None)

        if pilot:
            report["pilot_task_found"] = True
            task_id = pilot["id"]
            labels_res = client.get("/api/labels", params={"task_id": task_id})
            labels = labels_res.json().get("results", []) if labels_res.status_code == 200 else []

            label_names = {lbl["name"]: lbl for lbl in labels}
            has_leaf = "Leaf" in label_names
            has_diseased = "Diseased_Region" in label_names
            attr_ok = False
            found_types = set()

            if has_diseased:
                for attr in label_names["Diseased_Region"].get("attributes", []):
                    if attr.get("name") == "disease_type":
                        attr_ok = bool(attr.get("mutable", False))
                        found_types = set(attr.get("values", []))

            report["pilot_task_details"] = {
                "id": task_id,
                "name": pilot["name"],
                "size": pilot.get("size"),
                "status": pilot.get("status"),
                "project_id": pilot.get("project_id"),
            }
            report["label_verification"] = {
                "inherited_labels": list(label_names.keys()),
                "has_Leaf_label": has_leaf,
                "has_Diseased_Region_label": has_diseased,
                "disease_type_mutable": attr_ok,
                "found_disease_types": sorted(found_types),
                "missing_disease_types": sorted(EXPECTED_DISEASE_TYPES - found_types),
                "is_schema_valid": has_leaf and has_diseased and attr_ok and EXPECTED_DISEASE_TYPES.issubset(found_types),
            }

    return report


def main():
    parser = argparse.ArgumentParser(description="Inspect local CVAT Community Edition server and pilot task schema")
    parser.add_argument("--url", type=str, default="http://localhost:8080")
    parser.add_argument("--username", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--prompt-password", action="store_true")
    args = parser.parse_args()

    password = args.password
    if args.username and args.prompt_password and not password:
        password = getpass.getpass("CVAT Password: ")

    report = inspect_cvat_server(args.url, args.username, password)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
