"""
Read-Only CVAT Installation, Project/Task Schema, Annotation Count & Local Dataset Audit (Phase 6).

Inspects:
1. Local Docker containers (`cvat_server`, `cvat_ui`, `cvat_db`, `traefik`, `nuclio`) if accessible.
2. Live PostgreSQL `cvat` database tables (`engine_project`, `engine_task`, `engine_label`,
   `engine_attributespec`, `engine_image`, `engine_labeledshape`, `engine_labeledimage`).
3. Split-leakage check of any images currently loaded inside CVAT tasks against `test.csv`.
4. Local exported files in `data/annotations/`, `data/detection/`, and `datasets/yolo/`.

Never mutates CVAT containers, CVAT DB rows, or dataset splits.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_cmd(cmd: List[str], timeout: int = 12) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except Exception as exc:
        return -1, "", str(exc)


def _load_split_lookup() -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    splits_dir = PROJECT_ROOT / "artifacts" / "manifests" / "splits"
    for split_name in ("train", "validation", "test"):
        csv_path = splits_dir / f"{split_name}.csv"
        if not csv_path.is_file():
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            name = Path(str(row.get("relative_path", ""))).name
            if name:
                lookup[name] = {
                    "split": split_name,
                    "group_id": str(row.get("group_id", "")),
                    "class_label": str(row.get("class_folder_label", "")),
                }
    return lookup


def audit_cvat_and_local_annotations() -> Dict[str, Any]:
    split_lookup = _load_split_lookup()
    audit: Dict[str, Any] = {
        "cvat_compose_dir": r"D:\Main_Project\cvat_setup\cvat",
        "docker_accessible": False,
        "containers": {},
        "cvat_db_reachable": False,
        "projects": [],
        "tasks": [],
        "labels": [],
        "attributes": [],
        "annotation_counts": {
            "labeled_shapes": 0,
            "labeled_images": 0,
            "labeled_tracks": 0,
            "uploaded_images": 0,
        },
        "pilot_split_audit": {},
        "local_files": {},
        "issues_detected": [],
    }

    # 1. Inspect containers
    rc, out, err = _run_cmd(
        ["docker", "ps", "-a", "--format", "{{.Names}}||{{.Status}}||{{.Ports}}"]
    )
    if rc == 0:
        audit["docker_accessible"] = True
        for line in out.splitlines():
            parts = line.split("||")
            if len(parts) >= 2 and any(k in parts[0] for k in ("cvat", "traefik", "nuclio")):
                audit["containers"][parts[0]] = {
                    "status": parts[1],
                    "ports": parts[2] if len(parts) > 2 else "",
                }
        for key in ("traefik", "nuclio"):
            st = audit["containers"].get(key, {}).get("status", "")
            if st and not st.lower().startswith("up"):
                audit["issues_detected"].append(f"Container '{key}' is not running ({st}).")
    else:
        audit["docker_error"] = err or "Docker CLI unreachable"

    # 2. Query cvat_db PostgreSQL directly
    sql_queries = {
        "projects": "SELECT json_agg(t) FROM (SELECT id, name FROM engine_project ORDER BY id) t;",
        "tasks": "SELECT json_agg(t) FROM (SELECT id, name, project_id, status, mode FROM engine_task ORDER BY id) t;",
        "labels": "SELECT json_agg(t) FROM (SELECT id, name, project_id, task_id FROM engine_label ORDER BY id) t;",
        "attributes": "SELECT json_agg(t) FROM (SELECT id, name, mutable, input_type, values, label_id FROM engine_attributespec ORDER BY id) t;",
        "images": "SELECT json_agg(t) FROM (SELECT tk.id AS task_id, i.frame, i.path, i.width, i.height FROM engine_image i LEFT JOIN engine_task tk ON i.data_id = tk.data_id ORDER BY i.frame) t;",
        "counts": (
            "SELECT json_build_object("
            "'labeled_shapes', (SELECT COUNT(*) FROM engine_labeledshape),"
            "'labeled_images', (SELECT COUNT(*) FROM engine_labeledimage),"
            "'labeled_tracks', (SELECT COUNT(*) FROM engine_labeledtrack),"
            "'uploaded_images', (SELECT COUNT(*) FROM engine_image)"
            ");"
        ),
    }

    db_ok = False
    for key, query in sql_queries.items():
        q_rc, q_out, _ = _run_cmd(
            ["docker", "exec", "cvat_db", "psql", "-U", "root", "-d", "cvat", "-t", "-A", "-c", query]
        )
        if q_rc == 0:
            db_ok = True
            try:
                parsed = json.loads(q_out) if q_out and q_out != "null" else ([] if key != "counts" else {})
            except json.JSONDecodeError:
                parsed = []
            if key == "counts":
                audit["annotation_counts"] = parsed or audit["annotation_counts"]
            elif key == "images":
                task_images = parsed or []
                split_summary: Dict[str, List[str]] = {"train": [], "validation": [], "test": [], "unmatched": []}
                for rec in task_images:
                    fname = Path(str(rec.get("path", ""))).name
                    info = split_lookup.get(fname)
                    sp = info["split"] if info else "unmatched"
                    split_summary.setdefault(sp, []).append(fname)
                audit["pilot_split_audit"] = {
                    "cvat_uploaded_images": task_images,
                    "split_breakdown": {k: len(v) for k, v in split_summary.items()},
                    "test_split_overlap_files": split_summary.get("test", []),
                }
                if split_summary.get("test"):
                    audit["issues_detected"].append(
                        f"Existing CVAT task contains holdout test split files: {split_summary['test']}. "
                        "Do not include test.csv files in training annotations."
                    )
            else:
                audit[key] = parsed or []

    audit["cvat_db_reachable"] = db_ok
    if db_ok:
        if audit["annotation_counts"].get("labeled_shapes", 0) == 0:
            audit["issues_detected"].append("CVAT has 0 labeled polygons/boxes saved (engine_labeledshape = 0).")
        has_disease_attr = any(a.get("name") == "disease_type" for a in audit["attributes"])
        if not has_disease_attr:
            audit["issues_detected"].append(
                "CVAT label 'Diseased_Region' lacks mutable 'disease_type' attribute (engine_attributespec = 0)."
            )

    # 3. Inspect local folders
    ann_imgs = list((PROJECT_ROOT / "data" / "annotations" / "images").glob("*")) if (PROJECT_ROOT / "data" / "annotations" / "images").is_dir() else []
    ann_masks = list((PROJECT_ROOT / "data" / "annotations" / "masks").glob("*")) if (PROJECT_ROOT / "data" / "annotations" / "masks").is_dir() else []
    det_imgs = list((PROJECT_ROOT / "data" / "detection" / "images").glob("*")) if (PROJECT_ROOT / "data" / "detection" / "images").is_dir() else []
    det_lbls = list((PROJECT_ROOT / "data" / "detection" / "labels").glob("*")) if (PROJECT_ROOT / "data" / "detection" / "labels").is_dir() else []

    audit["local_files"] = {
        "segmentation_images_count": len([p for p in ann_imgs if p.is_file()]),
        "segmentation_masks_count": len([p for p in ann_masks if p.is_file()]),
        "detection_images_count": len([p for p in det_imgs if p.is_file()]),
        "detection_labels_count": len([p for p in det_lbls if p.is_file()]),
        "yolo_data_yaml_exists": (PROJECT_ROOT / "datasets" / "yolo" / "data.yaml").is_file(),
    }

    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit CVAT setup, DB tasks, attributes, and local annotations")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "manifests" / "cvat_status_audit.json",
    )
    args = parser.parse_args()

    audit = audit_cvat_and_local_annotations()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
