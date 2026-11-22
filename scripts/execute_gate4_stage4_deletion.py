"""
scripts/execute_gate4_stage4_deletion.py
CottonGuard AI — Phase 12.2 Stage 4: Controlled Deletion of 303 Obsolete Shapes.

Executes controlled deletion of exactly 303 obsolete shapes (10 Leaf pseudo-masks, 150 lesion rectangles,
143 synthetic 4-point polygons) from live CVAT Task #6 / Job #6 using PATCH /api/jobs/6/annotations/?action=delete.

STRICT SAFETY CONSTRAINTS:
1. Requires explicit `--confirm` flag to execute mutating requests. Defaults to read-only `--dry-run`.
2. Reads the frozen deletion registry (artifacts/audit_phase11/stage4_frozen_deletion_registry.json)
   and verifies its SHA-256 digest before touching CVAT.
3. Live pre-deletion shape inventory MUST be exactly 1,276 shapes.
4. All 303 deletion IDs must exist in the live job prior to deletion.
5. Zero protected resources may be deleted:
   - 37 DEC-01 rectangles (Frames 26 & 28)
   - 449 holdout shapes (Frames 35-49)
   - 106 Stage 3 additions (IDs 2224-2329)
6. Pre- and post-deletion live snapshots saved to artifacts/backups/.
7. Live post-deletion inventory MUST be exactly 973 shapes (345 rectangles, 628 polygons; 52 Leaf, 921 Diseased_Region).
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

# Authoritative Paths
FROZEN_REGISTRY_PATH = WORKSPACE / "artifacts/audit_phase11/stage4_frozen_deletion_registry.json"
EXPECTED_REGISTRY_SHA256 = "b970526a1e7b3aeddc84de367cdf0a415e31e1c71dd908116e9bd00d20e6dbb7"

STAGE3_POST_SNAPSHOT_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage3_post_addition_snapshot.json"
STAGE3_POST_SHA256 = "0aaee0a017b6de2247a6a1c1b8041f3041ff6b659ea6d88c1343a5eb5ae9b936"

PRE_STAGE4_SNAPSHOT_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage4_pre_deletion_snapshot.json"
POST_STAGE4_SNAPSHOT_PATH = WORKSPACE / "artifacts/backups/cvat_job6_stage4_post_deletion_snapshot.json"
STAGE3_MAPPING_PATH = WORKSPACE / "artifacts/annotation_staging/phase12_2/stage3_added_shapes_mapping.json"

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

    def patch_delete_shapes(self, job_id: int, shapes: List[Dict[str, Any]]) -> Dict[str, Any]:
        url = f"{self.base_url}/api/jobs/{job_id}/annotations/?action=delete"
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
            raise RuntimeError(f"CVAT PATCH delete annotations failed (HTTP {e.code}): {err}")


def run_stage4(cvat_url: str, cvat_user: str, cvat_pass: str, job_id: int = 6, dry_run: bool = False) -> Dict[str, Any]:
    print("================================================================================")
    print(" COTTONGUARD AI — PHASE 12.2 STAGE 4 CONTROLLED DELETION EXECUTOR")
    print(f" Execution Mode: {'DRY RUN (READ-ONLY)' if dry_run else 'LIVE MUTATION AUTHORIZED'}")
    print(f" Target Job ID: #{job_id}")
    print(f" CVAT Host: {cvat_url}")
    print("================================================================================\n")

    # -------------------------------------------------------------------------
    # PART 1: PRE-EXECUTION VERIFICATION & REGISTRY INTEGRITY
    # -------------------------------------------------------------------------
    print("--- [PART 1: PRE-EXECUTION VERIFICATION & REGISTRY INTEGRITY] ---")

    # 1. Verify Frozen Registry
    assert FROZEN_REGISTRY_PATH.exists(), f"Frozen deletion registry missing: {FROZEN_REGISTRY_PATH}"
    reg_hash = calc_sha256(FROZEN_REGISTRY_PATH)
    print(f"1. Frozen Deletion Registry: {FROZEN_REGISTRY_PATH.name}")
    print(f"   SHA-256: {reg_hash}")
    assert reg_hash == EXPECTED_REGISTRY_SHA256, f"Registry hash mismatch! Expected {EXPECTED_REGISTRY_SHA256}, got {reg_hash}"
    print("   -> PASS: Frozen deletion registry SHA-256 verified.")

    with open(FROZEN_REGISTRY_PATH, "r", encoding="utf-8") as f:
        deletion_records: List[Dict[str, Any]] = json.load(f)
    print(f"2. Deletion Registry Entry Count: {len(deletion_records)} (Expected: 303)")
    assert len(deletion_records) == 303, f"Expected 303 deletion entries, got {len(deletion_records)}"

    target_del_ids = {r["cvat_shape_id"] for r in deletion_records}
    assert len(target_del_ids) == 303, "Duplicate shape IDs in deletion registry!"

    # Registry Breakdown
    reg_labels = Counter(r["label_id"] for r in deletion_records)
    reg_types = Counter(r["geometry_type"] for r in deletion_records)
    print(f"   - Leaves: {reg_labels[5]} (Expected: 10)")
    print(f"   - Diseased_Region: {reg_labels[6]} (Expected: 293)")
    print(f"   - Rectangles: {reg_types['rectangle']} (Expected: 150)")
    print(f"   - Polygons: {reg_types['polygon']} (Expected: 153)")
    assert reg_labels[5] == 10 and reg_labels[6] == 293
    assert reg_types["rectangle"] == 150 and reg_types["polygon"] == 153
    print("   -> PASS: Registry counts match exact ground truth (10 Leaf, 293 Diseased; 150 Rect, 153 Poly).")

    # 2. Connect to CVAT & Validate Target Job
    print(f"\n3. Connecting to CVAT at {cvat_url}...")
    client = CVATClient(cvat_url, cvat_user, cvat_pass)
    job = client.get_job(job_id)
    assert job.get("task_id") == 6, f"Job #{job_id} belongs to Task #{job.get('task_id')}, expected 6"
    print(f"   Target Confirmed: Job #{job_id} on Task #{job.get('task_id')} ('{job.get('stage')}')")

    # 3. Fetch Live Annotations & Verify Inventory
    live_ann = client.get_annotations(job_id)
    live_shapes = live_ann.get("shapes", [])
    total_live = len(live_shapes)
    print(f"4. Live CVAT Pre-Deletion Shape Count: {total_live} (Expected: 1,276)")
    assert total_live == 1276, f"CRITICAL: Expected 1,276 live shapes before Stage 4, found {total_live}!"

    live_types = Counter(s.get("type") for s in live_shapes)
    live_labels = Counter(s.get("label_id") for s in live_shapes)
    print(f"   Live Types: {dict(live_types)} (Expected: 495 rect, 781 poly)")
    print(f"   Live Labels: {dict(live_labels)} (Expected: 62 Leaf, 1214 Diseased_Region)")
    assert live_types.get("rectangle") == 495
    assert live_types.get("polygon") == 781
    assert live_labels.get(5) == 62
    assert live_labels.get(6) == 1214
    print("   -> PASS: Live baseline inventory verified at exactly 1,276 shapes.")

    # 4. Check Presence of All 303 Deletion IDs
    live_shapes_by_id = {s["id"]: s for s in live_shapes}
    missing_del_ids = target_del_ids - set(live_shapes_by_id.keys())
    print(f"5. Target Deletion IDs Presence Check: {len(target_del_ids - missing_del_ids)}/303 present")
    assert len(missing_del_ids) == 0, f"CRITICAL: Missing deletion IDs in live CVAT: {missing_del_ids}"
    print("   -> PASS: All 303 scheduled deletion shapes are live in CVAT.")

    # 5. Check Protected Resources Exclusion
    # A. 37 DEC-01 Rectangles
    f26_rects = {s["id"] for s in live_shapes if s.get("frame") == 26 and s.get("type") == "rectangle"}
    f28_rects = {s["id"] for s in live_shapes if s.get("frame") == 28 and s.get("type") == "rectangle"}
    dec01_ids = f26_rects | f28_rects
    assert len(f26_rects) == 14 and len(f28_rects) == 23 and len(dec01_ids) == 37
    assert len(target_del_ids & dec01_ids) == 0, "CRITICAL: Deletion roster contains DEC-01 rectangles!"

    # B. Holdout Shapes (Frames 35-49)
    holdout_ids = {s["id"] for s in live_shapes if 35 <= s.get("frame", -1) <= 49}
    assert len(holdout_ids) == 449
    assert len(target_del_ids & holdout_ids) == 0, "CRITICAL: Deletion roster contains holdout shapes!"

    # C. Stage 3 Additions (IDs 2224-2329)
    stage3_ids = set(range(2224, 2330))
    assert len(target_del_ids & stage3_ids) == 0, "CRITICAL: Deletion roster contains Stage 3 additions!"
    print("6. Protected Resources Isolation: 100% EXCLUDED (0 DEC-01, 0 Holdout, 0 Stage 3 Additions).")

    # 6. Save Pre-Deletion Snapshot
    PRE_STAGE4_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PRE_STAGE4_SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(live_ann, f, indent=2)
    pre_snap_hash = calc_sha256(PRE_STAGE4_SNAPSHOT_PATH)
    print(f"7. Pre-Stage 4 Snapshot Saved: {PRE_STAGE4_SNAPSHOT_PATH.name} (SHA-256: {pre_snap_hash})")
    print("   -> PASS: All pre-execution assertions strictly satisfied.\n")

    # -------------------------------------------------------------------------
    # PART 2: PREPARE DELETION PAYLOAD
    # -------------------------------------------------------------------------
    print("--- [PART 2: PREPARING DELETION PAYLOAD] ---")
    # In CVAT 2.82.1 PATCH ?action=delete requires shape objects with id, frame, label_id, type
    deletion_payload_shapes = []
    for sid in sorted(target_del_ids):
        s = live_shapes_by_id[sid]
        # Construct clean shape descriptor for deletion
        del_item = {
            "id": s["id"],
            "frame": s["frame"],
            "label_id": s["label_id"],
            "type": s["type"],
            "points": s.get("points", []),
            "occluded": s.get("occluded", False),
            "outside": s.get("outside", False),
            "z_order": s.get("z_order", 0),
            "rotation": s.get("rotation", 0.0),
            "attributes": s.get("attributes", [])
        }
        deletion_payload_shapes.append(del_item)

    print(f"Prepared {len(deletion_payload_shapes)} deletion shape payloads.")
    assert len(deletion_payload_shapes) == 303

    if dry_run:
        print("\n[DRY RUN COMPLETE] 100% pre-deletion assertions passed. Zero mutations sent to CVAT.")
        return {"status": "dry_run_success", "pre_shape_count": 1276, "deletion_count": 303, "expected_final": 973}

    # -------------------------------------------------------------------------
    # PART 3: EXECUTE CONTROLLED DELETION
    # -------------------------------------------------------------------------
    print("\n--- [PART 3: EXECUTING LIVE CVAT MUTATION (PATCH ?action=delete)] ---")
    print(f"Sending atomic PATCH ?action=delete request with {len(deletion_payload_shapes)} shapes to Job #{job_id}...")
    patch_result = client.patch_delete_shapes(job_id, deletion_payload_shapes)
    print(f"PATCH delete request completed successfully. Response: {patch_result}")

    # -------------------------------------------------------------------------
    # PART 4: POST-DELETION VERIFICATION
    # -------------------------------------------------------------------------
    print("\n--- [PART 4: POST-DELETION VERIFICATION & AUDIT] ---")
    live_post = client.get_annotations(job_id)
    post_shapes = live_post.get("shapes", [])
    total_post = len(post_shapes)
    print(f"1. Live Post-Deletion Shape Count: {total_post} (Expected: 973)")
    assert total_post == 973, f"CRITICAL: Expected exactly 973 shapes after Stage 4, found {total_post}!"
    print("   -> PASS: Total shape count is EXACTLY 973 (1,276 - 303).")

    post_types = Counter(s.get("type") for s in post_shapes)
    post_labels = Counter(s.get("label_id") for s in post_shapes)
    print(f"2. Post Geometry Breakdown: {dict(post_types)}")
    print(f"   - Rectangles: {post_types.get('rectangle')} (Expected: 345)")
    print(f"   - Polygons:   {post_types.get('polygon')} (Expected: 628)")
    assert post_types.get("rectangle") == 345, f"Expected 345 rects, got {post_types.get('rectangle')}"
    assert post_types.get("polygon") == 628, f"Expected 628 polys, got {post_types.get('polygon')}"
    print("   -> PASS: Geometry counts match expected (345 rect, 628 poly).")

    print(f"3. Post Label Breakdown: {dict(post_labels)}")
    print(f"   - Leaf (id 5):            {post_labels.get(5)} (Expected: 52)")
    print(f"   - Diseased_Region (id 6): {post_labels.get(6)} (Expected: 921)")
    assert post_labels.get(5) == 52, f"Expected 52 Leaf, got {post_labels.get(5)}"
    assert post_labels.get(6) == 921, f"Expected 921 Diseased, got {post_labels.get(6)}"
    print("   -> PASS: Label counts match expected (52 Leaf, 921 Diseased_Region).")

    # Verify Deletion Completeness (None of the 303 deletion IDs exist)
    post_shape_ids = {s.get("id") for s in post_shapes}
    lingering_deletions = target_del_ids & post_shape_ids
    assert len(lingering_deletions) == 0, f"CRITICAL: Deletion shapes still present in CVAT: {lingering_deletions}"
    print("4. Deletion Completeness Check: 303/303 target shapes completely deleted from CVAT.")

    # Verify Retention of All 106 Stage 3 Additions
    missing_stage3 = stage3_ids - post_shape_ids
    assert len(missing_stage3) == 0, f"CRITICAL: Stage 3 additions were accidentally deleted: {missing_stage3}"
    print("5. Stage 3 Additions Preservation: 106/106 additions (2224-2329) fully intact.")

    # Verify Retention of 37 DEC-01 Rectangles
    missing_dec01 = dec01_ids - post_shape_ids
    assert len(missing_dec01) == 0, f"CRITICAL: DEC-01 rectangles were deleted: {missing_dec01}"
    print("6. DEC-01 Preservation: 37/37 detection rectangles intact.")

    # Verify Retention of 449 Holdout Shapes
    missing_holdouts = holdout_ids - post_shape_ids
    assert len(missing_holdouts) == 0, f"CRITICAL: Holdout shapes were mutated: {missing_holdouts}"
    print("7. Holdout Preservation: 449/449 holdout shapes completely untouched.")

    # Save Post-Deletion Snapshot
    POST_STAGE4_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(POST_STAGE4_SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(live_post, f, indent=2)
    post_snap_hash = calc_sha256(POST_STAGE4_SNAPSHOT_PATH)
    print(f"8. Post-Stage 4 Snapshot Saved: {POST_STAGE4_SNAPSHOT_PATH.name} (SHA-256: {post_snap_hash})")

    print("\n================================================================================")
    print(" STAGE 4 CONTROLLED DELETION COMPLETED SUCCESSFULLY!")
    print(f" Final Live Shape Count: 973 shapes")
    print(f" Polygons: 628 | Rectangles: 345")
    print(f" Leaves: 52 | Diseased_Region: 921")
    print("================================================================================")

    return {
        "status": "success",
        "pre_count": total_live,
        "post_count": total_post,
        "deleted_count": len(target_del_ids),
        "post_snapshot_path": str(POST_STAGE4_SNAPSHOT_PATH),
        "post_snapshot_sha256": post_snap_hash
    }


def main():
    parser = argparse.ArgumentParser(description="Execute Stage 4 controlled deletion on CVAT Job #6.")
    parser.add_argument("--confirm", action="store_true", help="Authorize live mutation in CVAT.")
    parser.add_argument("--dry-run", action="store_true", help="Perform pre-flight verification only.")
    parser.add_argument("--job-id", type=int, default=6, help="Target CVAT job ID (default 6).")
    args = parser.parse_args()

    if not args.confirm and not args.dry_run:
        print("ERROR: Must specify either --confirm (for execution) or --dry-run (for read-only check).")
        sys.exit(1)

    run_stage4(
        cvat_url=DEFAULT_CVAT_URL,
        cvat_user=DEFAULT_CVAT_USER,
        cvat_pass=DEFAULT_CVAT_PASS,
        job_id=args.job_id,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
