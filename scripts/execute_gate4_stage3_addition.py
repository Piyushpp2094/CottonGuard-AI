"""
scripts/execute_gate4_stage3_addition.py
CottonGuard AI — Phase 12.2 Stage 3: Controlled Addition of 106 Replacement Polygons.

Performs controlled addition of exactly 106 precision polygons (12 Leaf, 94 Diseased_Region)
to live CVAT Task #6 / Job #6 using PATCH /api/jobs/6/annotations/?action=create.

STRICT SAFETY CONSTRAINTS:
1. Zero deletions (Stage 4 remains BLOCKED).
2. Exactly 106 additions (12 Leaf [id=5], 94 Diseased_Region [id=6]).
3. Intermediate live inventory must become exactly 1,276 shapes (495 rectangles, 781 polygons).
4. All 37 DEC-01 rectangles (Frames 26 & 28) and holdout frames (35–49) must remain intact.
5. All 303 scheduled deletion IDs must remain present in live CVAT.
6. Cryptographic verification of baseline backup and Stage 0 snapshot hashes.
7. Fresh pre- and post-execution live snapshots saved and hashed.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")
sys.path.insert(0, str(WORKSPACE))

from scripts.execute_gate4_stage0_verification import DELETIONS_BY_FRAME

# Authoritative paths
CANONICAL_BACKUP_PATH = WORKSPACE / "artifacts/backups/cvat_job6_fresh_pre_remediation_baseline_20261118.json"
CANONICAL_BACKUP_ALT = WORKSPACE / "artifacts/backups/cvat_job6_post_phase9_5_backup_20261116_223741.json"
CANONICAL_BACKUP_SHA256 = "b516c238c330ee48b3a3f8492c694fac0165fc7e37c5ffafcedd89e8fff2e145"

STAGE0_SNAPSHOT_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage0_pre_remediation_snapshot_20261118_205937.json"
STAGE0_SNAPSHOT_SHA256 = "79ffce8ab7c004c789e32d3c2f4d9968ff541e56ddd01938a0d4c78e34d36939"

ADDITIONS_MANIFEST_PATH = WORKSPACE / "artifacts/annotation_staging/phase12_2/precision_additions_manifest.json"

PRE_STAGE3_SNAPSHOT_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage3_pre_addition_snapshot.json"
POST_STAGE3_SNAPSHOT_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage3_post_addition_snapshot.json"
MAPPING_OUTPUT_PATH = WORKSPACE / "artifacts/annotation_staging/phase12_2/stage3_added_shapes_mapping.json"

# CVAT connection defaults
DEFAULT_CVAT_URL = os.environ.get("CVAT_URL", "http://localhost:8080")
DEFAULT_CVAT_USER = os.environ.get("CVAT_USERNAME", "cottonguard")
DEFAULT_CVAT_PASS = os.environ.get("CVAT_PASSWORD", "CottonGuardCVAT2026!")


def calc_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


class CVATClient:
    """HTTP client for CVAT 2.82.1 REST API."""

    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.token = None
        self._login()

    def _login(self):
        login_url = f"{self.base_url}/api/auth/login"
        payload = json.dumps({"username": self.username, "password": self.password}).encode("utf-8")
        req = urllib.request.Request(
            login_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self.token = data.get("key")
                if not self.token:
                    raise RuntimeError("Login successful but token key missing in response.")
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"CVAT login failed (HTTP {e.code}): {err}")

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Token {self.token}",
            "Content-Type": "application/json",
        }

    def get_server_about(self) -> Dict[str, Any]:
        url = f"{self.base_url}/api/server/about"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_job(self, job_id: int) -> Dict[str, Any]:
        url = f"{self.base_url}/api/jobs/{job_id}"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_annotations(self, job_id: int) -> Dict[str, Any]:
        url = f"{self.base_url}/api/jobs/{job_id}/annotations/"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def patch_create_shapes(self, job_id: int, shapes: List[Dict[str, Any]]) -> Dict[str, Any]:
        url = f"{self.base_url}/api/jobs/{job_id}/annotations/?action=create"
        payload = {
            "version": 0,
            "shapes": shapes,
            "tracks": [],
            "tags": []
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=self._get_headers(), method="PATCH")
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()
                if data:
                    try:
                        return json.loads(data.decode("utf-8"))
                    except Exception:
                        return {"raw": data.decode("utf-8", errors="replace")}
                return {"status": "ok"}
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"CVAT PATCH annotations failed (HTTP {e.code}): {err}")


def get_303_deletion_ids() -> Set[int]:
    all_del_ids = set()
    for f_idx, d_dict in DELETIONS_BY_FRAME.items():
        all_del_ids.update(d_dict["leaf"])
        all_del_ids.update(d_dict["rect"])
        all_del_ids.update(d_dict["poly_4pt"])
    return all_del_ids


def run_stage3(cvat_url: str, cvat_user: str, cvat_pass: str, job_id: int = 6, dry_run: bool = False) -> Dict[str, Any]:
    print("================================================================================")
    print(" COTTONGUARD AI — PHASE 12.2 STAGE 3 CONTROLLED ADDITION EXECUTION")
    print(f" Execution Mode: {'DRY RUN (READ-ONLY)' if dry_run else 'LIVE MUTATION AUTHORIZED'}")
    print(f" Target Job ID: #{job_id}")
    print(f" CVAT Host: {cvat_url}")
    print("================================================================================\n")

    # -------------------------------------------------------------------------
    # PART 1: PRE-EXECUTION VERIFICATION
    # -------------------------------------------------------------------------
    print("--- [PART 1: PRE-EXECUTION VERIFICATION] ---")

    # 1. Verify Hashes
    canon_path = CANONICAL_BACKUP_PATH if CANONICAL_BACKUP_PATH.exists() else CANONICAL_BACKUP_ALT
    assert canon_path.exists(), f"Canonical backup not found: {canon_path}"
    canon_hash = calc_sha256(canon_path)
    print(f"1. Canonical Baseline Backup: {canon_path.name}")
    print(f"   SHA-256: {canon_hash}")
    assert canon_hash == CANONICAL_BACKUP_SHA256, f"Canonical hash mismatch! Expected {CANONICAL_BACKUP_SHA256}"
    print("   -> PASS: Canonical hash verified.")

    assert STAGE0_SNAPSHOT_PATH.exists(), f"Stage 0 snapshot not found: {STAGE0_SNAPSHOT_PATH}"
    stage0_hash = calc_sha256(STAGE0_SNAPSHOT_PATH)
    print(f"2. Stage 0 Recovery Snapshot: {STAGE0_SNAPSHOT_PATH.name}")
    print(f"   SHA-256: {stage0_hash}")
    assert stage0_hash == STAGE0_SNAPSHOT_SHA256, f"Stage 0 hash mismatch! Expected {STAGE0_SNAPSHOT_SHA256}"
    print("   -> PASS: Stage 0 hash verified.")

    # 2. Verify Manifest
    assert ADDITIONS_MANIFEST_PATH.exists(), f"Additions manifest not found: {ADDITIONS_MANIFEST_PATH}"
    with open(ADDITIONS_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_items: List[Dict[str, Any]] = json.load(f)
    print(f"3. Authoritative Additions Manifest: {ADDITIONS_MANIFEST_PATH.name}")
    print(f"   Total Addition Entries: {len(manifest_items)} (Expected: 106)")
    assert len(manifest_items) == 106, f"Expected 106 manifest entries, got {len(manifest_items)}"

    leaf_manifest = [x for x in manifest_items if x.get("label_id") == 5]
    lesion_manifest = [x for x in manifest_items if x.get("label_id") == 6]
    print(f"   - Leaf Additions (label_id=5): {len(leaf_manifest)} (Expected: 12)")
    print(f"   - Diseased_Region Additions (label_id=6): {len(lesion_manifest)} (Expected: 94)")
    assert len(leaf_manifest) == 12, f"Expected 12 leaf additions, got {len(leaf_manifest)}"
    assert len(lesion_manifest) == 94, f"Expected 94 lesion additions, got {len(lesion_manifest)}"
    assert all(x.get("type") == "polygon" for x in manifest_items), "All manifest shapes must be polygons!"
    print("   -> PASS: Manifest composition verified (12 Leaf + 94 Diseased_Region).")

    # 3. Verify Candidate 1337
    c1337 = [x for x in manifest_items if "1337" in str(x.get("candidate_id")) or x.get("replaced_shape_id") == 1337]
    assert len(c1337) == 1, f"Expected 1 Candidate 1337, found {len(c1337)}"
    cand_1337 = c1337[0]
    print(f"4. Candidate 1337 Verification:")
    print(f"   Candidate ID: {cand_1337.get('candidate_id')}, Replaced ID: {cand_1337.get('replaced_shape_id')}")
    print(f"   Type: {cand_1337.get('type')}, Vertices: {cand_1337.get('vertex_count')}, Area: {cand_1337.get('area')}")
    assert cand_1337.get("type") == "polygon", "Candidate 1337 must be a polygon!"
    assert cand_1337.get("vertex_count") == 20, f"Candidate 1337 must have 20 vertices, got {cand_1337.get('vertex_count')}"
    assert len(cand_1337.get("polygon_coordinates")) == 40, "Candidate 1337 must have 40 float coordinates!"
    print("   -> PASS: Candidate 1337 verified as 20-vertex polygon replacement.")

    # 4. Verify 303 Scheduled Deletions
    del_ids_303 = get_303_deletion_ids()
    print(f"5. Scheduled Deletion Inventory: {len(del_ids_303)} IDs (Expected: 303)")
    assert len(del_ids_303) == 303, f"Expected 303 deletion IDs, got {len(del_ids_303)}"

    # 5. Connect to CVAT
    print(f"6. Connecting to CVAT at {cvat_url}...")
    client = CVATClient(cvat_url, cvat_user, cvat_pass)
    about = client.get_server_about()
    server_version = about.get("version", "unknown")
    print(f"   CVAT Connected. Version: {server_version}")

    job = client.get_job(job_id)
    assert job.get("task_id") == 6, f"Job #{job_id} belongs to Task #{job.get('task_id')}, expected 6"
    print(f"   Target Confirmed: Job #{job_id} on Task #{job.get('task_id')} ('{job.get('stage')}')")

    # 6. Fetch Live Pre-Execution Annotations
    live_pre = client.get_annotations(job_id)
    live_pre_shapes = live_pre.get("shapes", [])
    total_pre_shapes = len(live_pre_shapes)
    print(f"7. Live Pre-Execution Shape Count: {total_pre_shapes} (Expected: 1,170)")
    assert total_pre_shapes == 1170, f"Expected 1,170 shapes before Stage 3, found {total_pre_shapes}"

    pre_type_counts = Counter(s.get("type") for s in live_pre_shapes)
    pre_label_counts = Counter(s.get("label_id") for s in live_pre_shapes)
    print(f"   Pre Geometry Breakdown: {dict(pre_type_counts)} (Expected: 495 rect, 675 poly)")
    print(f"   Pre Label Breakdown: {dict(pre_label_counts)} (Expected: 50 Leaf [id=5], 1120 Diseased [id=6])")
    assert pre_type_counts.get("rectangle") == 495
    assert pre_type_counts.get("polygon") == 675
    assert pre_label_counts.get(5) == 50
    assert pre_label_counts.get(6) == 1120

    # Verify 37 DEC-01 rectangles
    f26_rects_pre = [s for s in live_pre_shapes if s.get("frame") == 26 and s.get("type") == "rectangle"]
    f28_rects_pre = [s for s in live_pre_shapes if s.get("frame") == 28 and s.get("type") == "rectangle"]
    print(f"   Pre DEC-01 Rectangles: Frame 26={len(f26_rects_pre)}, Frame 28={len(f28_rects_pre)}, Total={len(f26_rects_pre) + len(f28_rects_pre)}")
    assert len(f26_rects_pre) == 14, f"Expected 14 rects in Frame 26, got {len(f26_rects_pre)}"
    assert len(f28_rects_pre) == 23, f"Expected 23 rects in Frame 28, got {len(f28_rects_pre)}"

    # Verify holdouts
    holdout_shapes_pre = [s for s in live_pre_shapes if 35 <= s.get("frame", -1) <= 49]
    print(f"   Pre Holdout Shapes (Frames 35-49): {len(holdout_shapes_pre)} (Expected: 449)")
    assert len(holdout_shapes_pre) == 449

    # Verify all 303 deletion IDs present
    pre_shape_ids = {s.get("id") for s in live_pre_shapes}
    missing_del = del_ids_303 - pre_shape_ids
    assert len(missing_del) == 0, f"Missing deletion IDs in live baseline: {missing_del}"
    print(f"   Pre 303 Deletions Check: All 303 target deletion shapes present in live job.")

    # Save Pre-Execution Snapshot
    PRE_STAGE3_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PRE_STAGE3_SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(live_pre, f, indent=2)
    pre_snapshot_hash = calc_sha256(PRE_STAGE3_SNAPSHOT_PATH)
    print(f"8. Pre-Stage 3 Snapshot Saved: {PRE_STAGE3_SNAPSHOT_PATH.name}")
    print(f"   SHA-256: {pre_snapshot_hash}")
    print("   -> PASS: All pre-execution checks strictly verified.\n")

    # -------------------------------------------------------------------------
    # PART 2: PREPARATION OF 106 ADDITION PAYLOAD
    # -------------------------------------------------------------------------
    print("--- [PART 2: PREPARING ADDITION PAYLOAD] ---")
    cvat_shapes_payload: List[Dict[str, Any]] = []
    for item in manifest_items:
        coords = [float(p) for p in item["polygon_coordinates"]]
        shape_payload = {
            "type": "polygon",
            "frame": int(item["frame_number"]),
            "label_id": int(item["label_id"]),
            "points": coords,
            "occluded": False,
            "outside": False,
            "z_order": 0,
            "rotation": 0.0,
            "attributes": [],
            "group": 0,
            "source": "manual"
        }
        cvat_shapes_payload.append(shape_payload)

    print(f"Prepared {len(cvat_shapes_payload)} polygon shape payloads for CVAT.")
    assert len(cvat_shapes_payload) == 106

    if dry_run:
        print("\n[DRY RUN COMPLETE] Preflight passed 100%. No mutations sent to CVAT.")
        return {"status": "dry_run_success", "pre_shape_count": 1170}

    # -------------------------------------------------------------------------
    # PART 3: EXECUTE CONTROLLED ADDITIONS
    # -------------------------------------------------------------------------
    print("\n--- [PART 3: EXECUTING LIVE CVAT MUTATION (PATCH ?action=create)] ---")
    print(f"Sending atomic PATCH request with {len(cvat_shapes_payload)} polygons to Job #{job_id}...")
    patch_result = client.patch_create_shapes(job_id, cvat_shapes_payload)
    print(f"PATCH request completed successfully. Response: {patch_result}")

    # -------------------------------------------------------------------------
    # PART 4: POST-EXECUTION VERIFICATION
    # -------------------------------------------------------------------------
    print("\n--- [PART 4: POST-EXECUTION VERIFICATION & AUDIT] ---")
    live_post = client.get_annotations(job_id)
    live_post_shapes = live_post.get("shapes", [])
    total_post_shapes = len(live_post_shapes)
    print(f"1. Live Post-Execution Shape Count: {total_post_shapes} (Expected: 1,276)")
    assert total_post_shapes == 1276, f"CRITICAL: Expected 1,276 shapes, got {total_post_shapes}!"
    print("   -> PASS: Total shape count is EXACTLY 1,276 (1,170 + 106).")

    # Geometry Breakdown
    post_type_counts = Counter(s.get("type") for s in live_post_shapes)
    post_label_counts = Counter(s.get("label_id") for s in live_post_shapes)
    print(f"2. Post Geometry Breakdown: {dict(post_type_counts)}")
    print(f"   - Rectangles: {post_type_counts.get('rectangle')} (Expected: 495, delta: 0)")
    print(f"   - Polygons: {post_type_counts.get('polygon')} (Expected: 781, delta: +106)")
    assert post_type_counts.get("rectangle") == 495, f"Expected 495 rects, got {post_type_counts.get('rectangle')}"
    assert post_type_counts.get("polygon") == 781, f"Expected 781 polys, got {post_type_counts.get('polygon')}"
    print("   -> PASS: Geometry counts match expected (495 rect, 781 poly).")

    # Label Breakdown
    print(f"3. Post Label Breakdown: {dict(post_label_counts)}")
    print(f"   - Leaf (id 5): {post_label_counts.get(5)} (Expected: 62, delta: +12)")
    print(f"   - Diseased_Region (id 6): {post_label_counts.get(6)} (Expected: 1,214, delta: +94)")
    assert post_label_counts.get(5) == 62, f"Expected 62 Leaf, got {post_label_counts.get(5)}"
    assert post_label_counts.get(6) == 1214, f"Expected 1214 Diseased, got {post_label_counts.get(6)}"
    print("   -> PASS: Label counts match expected (62 Leaf, 1,214 Diseased_Region).")

    # Verify Baseline Preservation (All 1,170 original shape IDs exist)
    post_shape_ids = {s.get("id") for s in live_post_shapes}
    missing_baseline_ids = pre_shape_ids - post_shape_ids
    print(f"4. Baseline Shape Preservation Check: {len(pre_shape_ids - missing_baseline_ids)}/1170 preserved")
    assert len(missing_baseline_ids) == 0, f"CRITICAL: Baseline shapes were lost: {missing_baseline_ids}"
    print("   -> PASS: All 1,170 baseline shapes remain intact in live CVAT.")

    # Verify 303 Scheduled Deletions Preservation
    missing_del_post = del_ids_303 - post_shape_ids
    print(f"5. Scheduled 303 Deletion Preservation Check: {len(del_ids_303 - missing_del_post)}/303 present")
    assert len(missing_del_post) == 0, f"CRITICAL: Deletion shapes were mutated: {missing_del_post}"
    print("   -> PASS: All 303 scheduled deletion shapes remain in live CVAT (0 deletions executed).")

    # Verify 37 DEC-01 Rectangles Intact
    f26_rects_post = [s for s in live_post_shapes if s.get("frame") == 26 and s.get("type") == "rectangle"]
    f28_rects_post = [s for s in live_post_shapes if s.get("frame") == 28 and s.get("type") == "rectangle"]
    print(f"6. DEC-01 Rectangles Check: Frame 26={len(f26_rects_post)}, Frame 28={len(f28_rects_post)}, Total={len(f26_rects_post) + len(f28_rects_post)}")
    assert len(f26_rects_post) == 14
    assert len(f28_rects_post) == 23
    print("   -> PASS: All 37 DEC-01 rectangles preserved untouched.")

    # Verify Holdouts Intact
    holdout_shapes_post = [s for s in live_post_shapes if 35 <= s.get("frame", -1) <= 49]
    print(f"7. Holdout Frames 35-49 Check: {len(holdout_shapes_post)} shapes (Expected: 449)")
    assert len(holdout_shapes_post) == 449
    print("   -> PASS: Holdout frames 35-49 completely untouched.")

    # Identify Newly Created Shapes
    new_shapes = [s for s in live_post_shapes if s.get("id") not in pre_shape_ids]
    print(f"8. Identified Newly Created Shapes in CVAT: {len(new_shapes)} (Expected: 106)")
    assert len(new_shapes) == 106, f"Expected exactly 106 new shapes, got {len(new_shapes)}"

    # Map candidate_id to new CVAT shape ID
    # Index new shapes by (frame, label_id, rounded points tuple)
    new_shape_index = {}
    for ns in new_shapes:
        pts_key = tuple(round(p, 1) for p in ns.get("points", []))
        new_shape_index[(ns.get("frame"), ns.get("label_id"), pts_key)] = ns

    mapping_records = []
    unmatched_candidates = []
    for item in manifest_items:
        f_num = int(item["frame_number"])
        l_id = int(item["label_id"])
        pts_key = tuple(round(float(p), 1) for p in item["polygon_coordinates"])
        matched_shape = new_shape_index.get((f_num, l_id, pts_key))
        if matched_shape:
            new_id = matched_shape.get("id")
            rec = {
                "candidate_id": item["candidate_id"],
                "new_cvat_shape_id": new_id,
                "frame": f_num,
                "global_id": item.get("global_id"),
                "image_filename": item.get("image_filename"),
                "label": item.get("annotation_label"),
                "label_id": l_id,
                "type": "polygon",
                "replaced_shape_id": item.get("replaced_shape_id"),
                "vertex_count": len(pts_key) // 2,
                "points": list(pts_key)
            }
            mapping_records.append(rec)
        else:
            unmatched_candidates.append(item["candidate_id"])

    print(f"9. Candidate-to-CVAT Mapping: {len(mapping_records)}/106 mapped successfully.")
    assert len(unmatched_candidates) == 0, f"Unmatched candidates: {unmatched_candidates}"

    # Verify Candidate 1337 in mapping
    cand_1337_rec = [r for r in mapping_records if "1337" in r["candidate_id"] or r["replaced_shape_id"] == 1337][0]
    print(f"10. Candidate 1337 Post-Upload Verification:")
    print(f"    Candidate ID: {cand_1337_rec['candidate_id']}")
    print(f"    New CVAT Shape ID: {cand_1337_rec['new_cvat_shape_id']}")
    print(f"    Replaced Shape ID: {cand_1337_rec['replaced_shape_id']}")
    print(f"    Vertex Count: {cand_1337_rec['vertex_count']}")
    assert cand_1337_rec["vertex_count"] == 20
    print("    -> PASS: Candidate 1337 successfully added as 20-vertex polygon.")

    # Save Mapping Artifact
    MAPPING_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPING_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping_records, f, indent=2)
    print(f"11. Saved candidate-to-shape mapping: {MAPPING_OUTPUT_PATH.name}")

    # Save Post-Execution Snapshot
    POST_STAGE3_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(POST_STAGE3_SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(live_post, f, indent=2)
    post_snapshot_hash = calc_sha256(POST_STAGE3_SNAPSHOT_PATH)
    print(f"12. Post-Stage 3 Snapshot Saved: {POST_STAGE3_SNAPSHOT_PATH.name}")
    print(f"    SHA-256: {post_snapshot_hash}")

    print("\n================================================================================")
    print(" STAGE 3 CONTROLLED ADDITION COMPLETED SUCCESSFULLY!")
    print(f" Intermediate Live Shape Count: 1,276 shapes")
    print(f" Polygons: 781 | Rectangles: 495")
    print(f" Leaves: 62 | Diseased_Region: 1,214")
    print(" STAGE 4 DELETIONS REMAIN STRICTLY BLOCKED PENDING USER AUTHORIZATION.")
    print("================================================================================")

    return {
        "status": "success",
        "pre_count": total_pre_shapes,
        "post_count": total_post_shapes,
        "additions_count": len(cvat_shapes_payload),
        "post_snapshot_path": str(POST_STAGE3_SNAPSHOT_PATH),
        "post_snapshot_sha256": post_snapshot_hash,
        "mapping_path": str(MAPPING_OUTPUT_PATH),
        "new_shape_ids_min": min(s["id"] for s in new_shapes),
        "new_shape_ids_max": max(s["id"] for s in new_shapes),
        "candidate_1337_new_id": cand_1337_rec["new_cvat_shape_id"],
    }


def main():
    parser = argparse.ArgumentParser(description="Execute Stage 3 controlled additions to CVAT Job #6.")
    parser.add_argument("--confirm", action="store_true", help="Authorize live mutation in CVAT.")
    parser.add_argument("--dry-run", action="store_true", help="Perform pre-flight verification only.")
    parser.add_argument("--job-id", type=int, default=6, help="Target CVAT job ID (default 6).")
    args = parser.parse_args()

    if not args.confirm and not args.dry_run:
        print("ERROR: Must specify either --confirm (for execution) or --dry-run (for read-only check).")
        sys.exit(1)

    run_stage3(
        cvat_url=DEFAULT_CVAT_URL,
        cvat_user=DEFAULT_CVAT_USER,
        cvat_pass=DEFAULT_CVAT_PASS,
        job_id=args.job_id,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
