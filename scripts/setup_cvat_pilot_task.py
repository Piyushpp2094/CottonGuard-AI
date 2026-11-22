"""
CVAT Pilot Project, Task & Export Compatibility Provisioner (Phase 6).

Creates (idempotently) a split-clean CVAT project and annotation task for the
21-image representative pilot (`data/pilot_batch_01/images/`) using
`artifacts/manifests/cvat_task_label_schema.json`:
  - Preserves existing Project #2 & Task #2 untouched (zero mixing with test-split images).
  - Configures `Leaf` + `Diseased_Region` with mutable `disease_type` (6 disease classes).
  - Verifies SAM ViT-H interactor availability (`/api/lambda/functions`).
  - Verifies `CVAT for images 1.1` XML dry-export compatibility before human annotation.

Never fabricates polygons, bounding boxes, or segmentation masks.
"""

import argparse
import csv
import getpass
import json
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_DISEASE_TYPES = {
    "Bacterial Blight",
    "Curl Virus",
    "Herbicide Growth Damage",
    "Leaf Hopper Jassids",
    "Leaf Redding",
    "Leaf Variegation",
}


def setup_and_verify_pilot_task(
    base_url: str,
    username: str,
    password: str,
    project_name: str = "CottonGuard_AI_Annotation_v2",
    task_name: str = "CottonGuard_Pilot_Batch_01_ShapesOnly",
    schema_path: Path = PROJECT_ROOT / "artifacts" / "manifests" / "cvat_task_label_schema.json",
    pilot_images_dir: Path = PROJECT_ROOT / "data" / "pilot_batch_01" / "images",
    manifest_csv_path: Optional[Path] = PROJECT_ROOT / "artifacts" / "manifests" / "pilot_batch_manifest.csv",
    dry_export_path: Path = PROJECT_ROOT / "artifacts" / "manifests" / "cvat_pilot_task4_dry_export_check.zip",
    report_path: Path = PROJECT_ROOT / "artifacts" / "manifests" / "cvat_pilot_task4_setup.json",
) -> Dict[str, Any]:
    url = base_url.rstrip("/")
    labels_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if manifest_csv_path and manifest_csv_path.is_file():
        with open(manifest_csv_path, "r", encoding="utf-8") as f:
            ordered_names = [row["filename"].strip() for row in csv.DictReader(f) if row.get("filename")]
        image_files = [pilot_images_dir / name for name in ordered_names if (pilot_images_dir / name).is_file()]
        sorting_method = "predefined"
    else:
        image_files = sorted(
            p for p in pilot_images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        sorting_method = "lexicographical"
    if not image_files:
        raise FileNotFoundError(f"No pilot images found in {pilot_images_dir}")

    with httpx.Client(base_url=url, timeout=30.0) as client:
        login_res = client.post("/api/auth/login", json={"username": username, "password": password})
        if login_res.status_code not in (200, 201):
            raise RuntimeError(f"CVAT login failed ({login_res.status_code}): {login_res.text[:250]}")
        token = login_res.json().get("key")
        if token:
            client.headers.update({"Authorization": f"Token {token}"})

        # 1. Check SAM ViT-H interactor in serverless endpoint
        sam_info: Dict[str, Any] = {"available": False, "functions": []}
        fn_res = client.get("/api/lambda/functions")
        if fn_res.status_code == 200:
            fn_json = fn_res.json()
            fn_list = fn_json if isinstance(fn_json, list) else fn_json.get("results", [])
            sam_info["functions"] = [
                {
                    "id": f.get("id"),
                    "name": f.get("name"),
                    "kind": f.get("kind"),
                    "state": f.get("status") or f.get("state"),
                }
                for f in fn_list
            ]
            sam_info["available"] = any("sam" in str(f.get("id", "")).lower() for f in fn_list)
        else:
            sam_info["http_status"] = fn_res.status_code

        # 2. Ensure clean project exists (idempotent lookup by name)
        proj_res = client.get("/api/projects", params={"search": project_name})
        proj_res.raise_for_status()
        projects = proj_res.json().get("results", [])
        project = next((p for p in projects if p.get("name") == project_name), None)
        if not project:
            create_proj = client.post(
                "/api/projects",
                json={"name": project_name, "labels": labels_schema},
            )
            if create_proj.status_code not in (200, 201):
                raise RuntimeError(f"Failed to create CVAT project: {create_proj.status_code} {create_proj.text}")
            project = create_proj.json()

        project_id = int(project["id"])

        # Verify / attach labels on project if created blank earlier
        proj_labels_res = client.get("/api/labels", params={"project_id": project_id})
        proj_labels_res.raise_for_status()
        proj_labels = proj_labels_res.json().get("results", [])
        if not proj_labels:
            patch_proj = client.patch(f"/api/projects/{project_id}", json={"labels": labels_schema})
            patch_proj.raise_for_status()

        # 3. Ensure task exists under project_id
        tasks_res = client.get("/api/tasks", params={"search": task_name})
        tasks_res.raise_for_status()
        tasks = tasks_res.json().get("results", [])
        task = next((t for t in tasks if t.get("name") == task_name and t.get("project_id") == project_id), None)

        if not task:
            create_task = client.post(
                "/api/tasks",
                json={
                    "name": task_name,
                    "project_id": project_id,
                    "subset": "pilot_batch_01",
                },
            )
            if create_task.status_code not in (200, 201):
                raise RuntimeError(f"Failed to create CVAT task: {create_task.status_code} {create_task.text}")
            task = create_task.json()

        task_id = int(task["id"])

        # 4. Upload images if task has size == 0
        task_detail = client.get(f"/api/tasks/{task_id}").json()
        if int(task_detail.get("size") or 0) == 0:
            files_payload = [
                ("client_files[{}]".format(idx), (img_p.name, img_p.read_bytes(), "image/jpeg"))
                for idx, img_p in enumerate(image_files)
            ]
            data_res = client.post(
                f"/api/tasks/{task_id}/data",
                data={
                    "image_quality": "100",
                    "use_zip_chunks": "true",
                    "use_cache": "true",
                    "sorting_method": sorting_method,
                },
                files=files_payload,
            )
            if data_res.status_code not in (201, 202):
                raise RuntimeError(f"Failed to upload pilot images: {data_res.status_code} {data_res.text}")
            rq_id = data_res.json().get("rq_id")
            if rq_id:
                for _ in range(45):
                    req_stat = client.get(f"/api/requests/{rq_id}")
                    if req_stat.status_code == 200:
                        st = req_stat.json().get("status")
                        if st == "finished":
                            break
                        if st == "failed":
                            raise RuntimeError(f"CVAT image import failed: {req_stat.text}")
                    time.sleep(1.0)
            else:
                for _ in range(30):
                    td = client.get(f"/api/tasks/{task_id}").json()
                    if int(td.get("size") or 0) == len(image_files):
                        break
                    time.sleep(1.0)
            task_detail = client.get(f"/api/tasks/{task_id}").json()

        # 5. Fetch task jobs & label inheritance verification
        jobs_res = client.get("/api/jobs", params={"task_id": task_id})
        jobs_res.raise_for_status()
        jobs = jobs_res.json().get("results", [])

        labels_res = client.get("/api/labels", params={"task_id": task_id})
        labels_res.raise_for_status()
        labels_list = labels_res.json().get("results", [])
        label_map = {lbl["name"]: lbl for lbl in labels_list}

        has_leaf = "Leaf" in label_map
        has_diseased = "Diseased_Region" in label_map
        disease_attr = None
        if has_diseased:
            for attr in label_map["Diseased_Region"].get("attributes", []):
                if attr.get("name") == "disease_type":
                    disease_attr = attr

        found_values = set(disease_attr.get("values", [])) if disease_attr else set()
        schema_valid = (
            has_leaf
            and has_diseased
            and disease_attr is not None
            and bool(disease_attr.get("mutable", False))
            and EXPECTED_DISEASE_TYPES.issubset(found_values)
        )

        # 6. Verify 'CVAT for images 1.1' export compatibility via modern CVAT v2 async export API
        export_ok = False
        export_xml_images = 0
        export_xml_labels: List[str] = []
        export_xml_disease_attrs: List[str] = []
        dry_export_path.parent.mkdir(parents=True, exist_ok=True)
        init_exp = client.post(
            f"/api/tasks/{task_id}/dataset/export",
            params={"format": "CVAT for images 1.1", "save_images": "false"},
        )
        if init_exp.status_code in (201, 202):
            exp_rq_id = init_exp.json().get("rq_id")
            result_url = None
            for _ in range(30):
                rq_stat = client.get(f"/api/requests/{exp_rq_id}")
                if rq_stat.status_code == 200:
                    st_data = rq_stat.json()
                    if st_data.get("status") == "finished":
                        result_url = st_data.get("result_url")
                        break
                    if st_data.get("status") == "failed":
                        break
                time.sleep(1.0)
            if result_url:
                dl_res = client.get(result_url)
                if dl_res.status_code == 200 and dl_res.content[:2] == b"PK":
                    dry_export_path.write_bytes(dl_res.content)
                    with zipfile.ZipFile(dry_export_path, "r") as zf:
                        if "annotations.xml" in zf.namelist():
                            xml_root = ET.fromstring(zf.read("annotations.xml"))
                            export_xml_images = len(xml_root.findall(".//image"))
                            export_xml_labels = [el.text or "" for el in xml_root.findall(".//labels/label/name") if el.text]
                            attr_vals_el = xml_root.find(".//labels/label[name='Diseased_Region']/attributes/attribute[name='disease_type']/values")
                            if attr_vals_el is not None and attr_vals_el.text:
                                export_xml_disease_attrs = [v.strip() for v in attr_vals_el.text.splitlines() if v.strip()]
                            export_ok = (
                                export_xml_images == int(task_detail.get("size") or 0)
                                and "Leaf" in export_xml_labels
                                and "Diseased_Region" in export_xml_labels
                                and EXPECTED_DISEASE_TYPES.issubset(set(export_xml_disease_attrs))
                            )

        report = {
            "cvat_url": url,
            "project": {"id": project_id, "name": project_name},
            "task": {
                "id": task_id,
                "name": task_name,
                "size": task_detail.get("size"),
                "status": task_detail.get("status"),
                "mode": task_detail.get("mode"),
                "job_ids": [j["id"] for j in jobs],
                "ui_url": f"{url}/tasks/{task_id}",
                "job_ui_url": f"{url}/tasks/{task_id}/jobs/{jobs[0]['id']}" if jobs else None,
            },
            "label_verification": {
                "inherited_labels": list(label_map.keys()),
                "has_Leaf_label": has_leaf,
                "has_Diseased_Region_label": has_diseased,
                "disease_type_mutable": bool(disease_attr.get("mutable", False)) if disease_attr else False,
                "found_disease_types": sorted(found_values),
                "missing_disease_types": sorted(EXPECTED_DISEASE_TYPES - found_values),
                "is_schema_valid": schema_valid,
            },
            "sam_vith_interactor": sam_info,
            "export_compatibility": {
                "format": "CVAT for images 1.1",
                "dry_export_zip": str(dry_export_path),
                "export_succeeded": export_ok,
                "xml_image_entries": export_xml_images,
                "xml_declared_labels": export_xml_labels,
                "xml_disease_type_values": export_xml_disease_attrs,
            },
            "expected_human_export_ingest_path": str(PROJECT_ROOT / "data" / "cvat_exports" / "pilot_batch_01_export.zip"),
            "human_annotations_completed": False,
        }

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Create & verify CottonGuard 21-image pilot CVAT project/task")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", default=None)
    parser.add_argument("--task-name", default="CottonGuard_Pilot_Batch_01_ShapesOnly")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts" / "manifests" / "pilot_batch_manifest.csv")
    args = parser.parse_args()

    password = args.password or getpass.getpass("CVAT Password: ")
    report = setup_and_verify_pilot_task(
        base_url=args.url,
        username=args.username,
        password=password,
        task_name=args.task_name,
        manifest_csv_path=args.manifest if args.manifest.is_file() else None,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
