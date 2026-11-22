"""
scripts/restore_cvat_task6_baseline.py
CottonGuard AI — Phase 12.1.4: Safe CVAT Task #6 Annotation Rollback Utility.

Safely restores CVAT Task #6 (or a specified test job) from a verified baseline backup JSON.

Safety Guarantees & Constraints:
1. Explicit Target Job ID Required: `--job-id` has NO default value; caller must explicitly pass `--job-id <int>`.
2. Explicit Confirmation Required: Mutating action requires `--confirm` flag.
3. Cryptographic Backup Hash Verification: Calculates SHA-256 of backup file and asserts match with expected hash.
4. Target Job Pre-Validation: Verifies target job exists in CVAT and has expected frame count before touching annotations.
5. Payload Sanitization: Strips server-assigned 'id' fields from shapes/tracks/tags as mandated by CVAT 2.82.1 PUT API.
6. Post-Upload Parity Verification: Verifies restored shape counts, frame-level distributions, and label/geometry breakdown against baseline.
7. Read-Only Dry-Run Mode: `--dry-run` performs full verification without sending PUT.

Usage Examples:
  # Dry-run validation of Job 6 against baseline
  python scripts/restore_cvat_task6_baseline.py --job-id 6 --dry-run

  # Full rollback of Job 6 to verified baseline
  python scripts/restore_cvat_task6_baseline.py --job-id 6 --confirm

  # Rollback on a disposable test job
  python scripts/restore_cvat_task6_baseline.py --job-id 99 --backup-file path/to/test_backup.json --expected-sha256 <hash> --expected-shape-count 5 --confirm
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Default canonical baseline
DEFAULT_BASELINE_BACKUP = Path(r"D:\Main_Project\cotton_leaf_ai\artifacts\backups\cvat_job6_post_phase9_5_backup_20261116_223741.json")
DEFAULT_BASELINE_SHA256 = "b516c238c330ee48b3a3f8492c694fac0165fc7e37c5ffafcedd89e8fff2e145"
DEFAULT_BASELINE_SHAPES = 1170
DEFAULT_BASELINE_FRAMES = 50

# Default CVAT connection
DEFAULT_CVAT_URL = os.environ.get("CVAT_URL", "http://localhost:8080")
DEFAULT_CVAT_USER = os.environ.get("CVAT_USERNAME", "cottonguard")
DEFAULT_CVAT_PASS = os.environ.get("CVAT_PASSWORD", "CottonGuardCVAT2026!")


def calc_sha256(path: Path) -> str:
    """Computes SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


class CVATClient:
    """Lightweight HTTP client for CVAT 2.82.1 REST API using standard urllib."""

    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.token: Optional[str] = None
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
                    raise RuntimeError("Login successful but token key not found in response.")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"CVAT login failed with HTTP {e.code}: {err_body}")

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Token {self.token}",
            "Content-Type": "application/json",
        }

    def get_job(self, job_id: int) -> Dict[str, Any]:
        """Fetch job details via /api/jobs/{job_id}."""
        url = f"{self.base_url}/api/jobs/{job_id}"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Failed to fetch Job #{job_id} (HTTP {e.code}): {err_body}")

    def get_annotations(self, job_id: int) -> Dict[str, Any]:
        """Fetch annotations via GET /api/jobs/{job_id}/annotations/."""
        url = f"{self.base_url}/api/jobs/{job_id}/annotations/"
        req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Failed to fetch annotations for Job #{job_id} (HTTP {e.code}): {err_body}")

    def put_annotations(self, job_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Replace all job annotations via PUT /api/jobs/{job_id}/annotations/."""
        url = f"{self.base_url}/api/jobs/{job_id}/annotations/"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=self._get_headers(), method="PUT")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Failed to upload annotations to Job #{job_id} (HTTP {e.code}): {err_body}")


def sanitize_payload_for_cvat_put(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes raw backup JSON for CVAT PUT /api/jobs/{id}/annotations/.
    CVAT 2.82.1 rejects shapes with {"shapes":{"0":{"id":["must be absent"]}}}.
    Strips 'id' from shapes, tracks, and tags.
    """
    clean_shapes: List[Dict[str, Any]] = []
    for s in raw_data.get("shapes", []):
        s_clean = {k: v for k, v in s.items() if k != "id"}
        clean_shapes.append(s_clean)

    clean_tracks: List[Dict[str, Any]] = []
    for t in raw_data.get("tracks", []):
        t_clean = {k: v for k, v in t.items() if k != "id"}
        clean_tracks.append(t_clean)

    clean_tags: List[Dict[str, Any]] = []
    for tg in raw_data.get("tags", []):
        tg_clean = {k: v for k, v in tg.items() if k != "id"}
        clean_tags.append(tg_clean)

    clean_intervals: List[Dict[str, Any]] = []
    for it in raw_data.get("intervals", []):
        it_clean = {k: v for k, v in it.items() if k != "id"}
        clean_intervals.append(it_clean)

    return {
        "version": raw_data.get("version", 0),
        "shapes": clean_shapes,
        "tracks": clean_tracks,
        "tags": clean_tags,
        "intervals": clean_intervals,
    }


def analyze_annotations(data: Dict[str, Any]) -> Dict[str, Any]:
    """Summarizes shapes by type, label, and frame distribution."""
    shapes = data.get("shapes", [])
    by_type = Counter(s.get("type") for s in shapes)
    by_label = Counter(s.get("label_id") for s in shapes)
    by_frame = Counter(s.get("frame") for s in shapes)
    return {
        "total_shapes": len(shapes),
        "total_tracks": len(data.get("tracks", [])),
        "total_tags": len(data.get("tags", [])),
        "by_type": dict(by_type),
        "by_label": dict(by_label),
        "frame_count_with_shapes": len(by_frame),
        "min_frame": min(by_frame.keys()) if by_frame else None,
        "max_frame": max(by_frame.keys()) if by_frame else None,
    }


def verify_job_parity(baseline_data: Dict[str, Any], live_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Compares baseline vs live annotations deeply."""
    issues = []
    b_shapes = baseline_data.get("shapes", [])
    l_shapes = live_data.get("shapes", [])

    if len(b_shapes) != len(l_shapes):
        issues.append(f"Shape count mismatch: baseline has {len(b_shapes)}, live has {len(l_shapes)}")

    b_type_counts = Counter(s.get("type") for s in b_shapes)
    l_type_counts = Counter(s.get("type") for s in l_shapes)
    if b_type_counts != l_type_counts:
        issues.append(f"Shape type breakdown mismatch: baseline {dict(b_type_counts)} vs live {dict(l_type_counts)}")

    b_frames = Counter(s.get("frame") for s in b_shapes)
    l_frames = Counter(s.get("frame") for s in l_shapes)
    if b_frames != l_frames:
        mismatched_frames = [f for f in set(b_frames.keys()) | set(l_frames.keys()) if b_frames[f] != l_frames[f]]
        issues.append(f"Frame shape distribution mismatch across {len(mismatched_frames)} frames: {mismatched_frames[:5]}...")

    b_labels = Counter(s.get("label_id") for s in b_shapes)
    l_labels = Counter(s.get("label_id") for s in l_shapes)
    if b_labels != l_labels:
        issues.append(f"Label ID distribution mismatch: baseline {dict(b_labels)} vs live {dict(l_labels)}")

    # Full category verification (shapes, tracks, tags, intervals)
    b_tracks = baseline_data.get("tracks", [])
    l_tracks = live_data.get("tracks", [])
    if len(b_tracks) != len(l_tracks):
        issues.append(f"Track count mismatch: baseline has {len(b_tracks)}, live has {len(l_tracks)}")

    b_tags = baseline_data.get("tags", [])
    l_tags = live_data.get("tags", [])
    if len(b_tags) != len(l_tags):
        issues.append(f"Tag count mismatch: baseline has {len(b_tags)}, live has {len(l_tags)}")

    b_intervals = baseline_data.get("intervals", [])
    l_intervals = live_data.get("intervals", [])
    if len(b_intervals) != len(l_intervals):
        issues.append(f"Interval count mismatch: baseline has {len(b_intervals)}, live has {len(l_intervals)}")

    return len(issues) == 0, issues


def run_rollback(
    job_id: int,
    backup_file: Path,
    expected_sha256: str,
    expected_shape_count: Optional[int],
    expected_frame_count: Optional[int],
    cvat_url: str,
    username: str,
    password: str,
    confirm: bool,
    dry_run: bool,
) -> bool:
    print("=" * 80)
    print("COTTONGUARD AI — CVAT TASK/JOB ANNOTATION ROLLBACK UTILITY")
    print("=" * 80)
    print(f"Target Job ID:          {job_id}")
    print(f"Backup File Path:       {backup_file}")
    print(f"Expected SHA-256:       {expected_sha256.lower()}")
    print(f"Dry-Run Mode:           {dry_run}")
    print(f"Confirmed Execution:    {confirm}")
    print("-" * 80)

    # 1. Enforce confirmation
    if not confirm and not dry_run:
        print("[ERROR] Safety abort: neither --confirm nor --dry-run was specified!")
        print("        To inspect without modifying, use --dry-run.")
        print("        To execute the rollback, pass --confirm explicitly.")
        return False

    # 2. Verify backup file exists and matches expected SHA-256
    if not backup_file.is_file():
        print(f"[ERROR] Backup file does not exist: {backup_file}")
        return False

    print("[STEP 1/5] Verifying backup file integrity...")
    actual_sha256 = calc_sha256(backup_file)
    print(f"  Backup File Size:     {backup_file.stat().st_size:,} bytes")
    print(f"  Actual SHA-256:       {actual_sha256}")
    if actual_sha256.lower() != expected_sha256.lower():
        print(f"[ERROR] SHA-256 MISMATCH!")
        print(f"  Expected: {expected_sha256.lower()}")
        print(f"  Actual:   {actual_sha256.lower()}")
        return False
    print("  -> Cryptographic SHA-256 verification PASS.")

    # 3. Parse backup JSON and summarize
    print("[STEP 2/5] Parsing and validating baseline payload structure...")
    try:
        with open(backup_file, "r", encoding="utf-8") as f:
            backup_raw = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to parse backup JSON: {e}")
        return False

    backup_summary = analyze_annotations(backup_raw)
    print(f"  Total Shapes:         {backup_summary['total_shapes']}")
    print(f"  Shape Types:          {backup_summary['by_type']}")
    print(f"  Label Distribution:   {backup_summary['by_label']}")
    print(f"  Frames with shapes:   {backup_summary['frame_count_with_shapes']}")

    if expected_shape_count is not None and backup_summary["total_shapes"] != expected_shape_count:
        print(f"[ERROR] Backup shape count {backup_summary['total_shapes']} != expected {expected_shape_count}!")
        return False

    # 4. Connect to CVAT and validate target job
    print("[STEP 3/5] Connecting to CVAT and inspecting target job...")
    try:
        client = CVATClient(cvat_url, username, password)
        print("  -> CVAT authentication successful.")
    except Exception as e:
        print(f"[ERROR] CVAT authentication failure: {e}")
        return False

    try:
        job_meta = client.get_job(job_id)
        job_stage = job_meta.get("stage")
        job_state = job_meta.get("state")
        start_frame = job_meta.get("start_frame", 0)
        stop_frame = job_meta.get("stop_frame", 0)
        frame_count = stop_frame - start_frame + 1
        print(f"  Job #{job_id} found:")
        print(f"    Task ID:            {job_meta.get('task_id')}")
        print(f"    Stage:              {job_stage}")
        print(f"    State:              {job_state}")
        print(f"    Frames:             {start_frame}..{stop_frame} (Total: {frame_count} frames)")
    except Exception as e:
        print(f"[ERROR] Target Job #{job_id} validation failed: {e}")
        return False

    if expected_frame_count is not None and frame_count != expected_frame_count:
        print(f"[ERROR] Target job frame count {frame_count} != expected {expected_frame_count}!")
        return False

    # Enforce 50 frames if targeting Job 6
    if job_id == 6 and frame_count != 50:
        print(f"[ERROR] Job 6 frame count {frame_count} != 50!")
        return False

    # Check current live annotations
    pre_live = client.get_annotations(job_id)
    pre_summary = analyze_annotations(pre_live)
    print(f"  Current Live Annotations on Job #{job_id}:")
    print(f"    Total Shapes:       {pre_summary['total_shapes']}")
    print(f"    Shape Types:        {pre_summary['by_type']}")

    # 5. Sanitize payload
    clean_payload = sanitize_payload_for_cvat_put(backup_raw)
    print(f"  Sanitized payload prepared ({len(clean_payload['shapes'])} shapes, 'id' stripped).")

    if dry_run:
        print("\n[DRY RUN COMPLETE] Target job and backup verified. No changes made.")
        return True

    # 6. Upload sanitized payload to target job
    print("\n[STEP 4/5] Executing PUT /api/jobs/{job_id}/annotations/...")
    try:
        put_result = client.put_annotations(job_id, clean_payload)
        put_shapes = put_result.get("shapes", [])
        print(f"  PUT response returned {len(put_shapes)} shapes.")
    except Exception as e:
        print(f"[ERROR] Failed to execute PUT annotations: {e}")
        return False

    # 7. Post-upload verification
    print("\n[STEP 5/5] Performing post-rollback parity verification...")
    try:
        post_live = client.get_annotations(job_id)
        post_summary = analyze_annotations(post_live)
        print(f"  Post-rollback Live Annotations on Job #{job_id}:")
        print(f"    Total Shapes:       {post_summary['total_shapes']}")
        print(f"    Shape Types:        {post_summary['by_type']}")
        print(f"    Label Distribution: {post_summary['by_label']}")
        print(f"    Frames with shapes: {post_summary['frame_count_with_shapes']}")

        is_parity, issues = verify_job_parity(backup_raw, post_live)
        if not is_parity:
            print("[ERROR] Post-rollback parity verification FAILED!")
            for iss in issues:
                print(f"  - {iss}")
            return False

        print("  -> Exact shape count parity PASS.")
        print("  -> Exact shape type breakdown parity PASS.")
        print("  -> Exact frame distribution parity PASS.")
        print("  -> Exact label ID distribution parity PASS.")
        print("\n[SUCCESS] Job #{} successfully rolled back and verified against baseline.".format(job_id))
        return True
    except Exception as e:
        print(f"[ERROR] Post-rollback verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="CottonGuard AI — Safe CVAT Annotation Rollback Utility (Phase 12.1.4)"
    )
    parser.add_argument(
        "--job-id",
        type=int,
        required=True,
        help="CVAT Job ID to restore (REQUIRED: no default allowed for safety).",
    )
    parser.add_argument(
        "--backup-file",
        type=Path,
        default=DEFAULT_BASELINE_BACKUP,
        help=f"Path to backup JSON file (default: {DEFAULT_BASELINE_BACKUP.name}).",
    )
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=DEFAULT_BASELINE_SHA256,
        help=f"Expected SHA-256 of backup file (default: {DEFAULT_BASELINE_SHA256}).",
    )
    parser.add_argument(
        "--expected-shape-count",
        type=int,
        default=DEFAULT_BASELINE_SHAPES,
        help=f"Expected shape count (default: {DEFAULT_BASELINE_SHAPES}).",
    )
    parser.add_argument(
        "--expected-frame-count",
        type=int,
        default=DEFAULT_BASELINE_FRAMES,
        help=f"Expected job frame count (default: {DEFAULT_BASELINE_FRAMES}).",
    )
    parser.add_argument(
        "--cvat-url",
        type=str,
        default=DEFAULT_CVAT_URL,
        help=f"CVAT server base URL (default: {DEFAULT_CVAT_URL}).",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=DEFAULT_CVAT_USER,
        help="CVAT username.",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=DEFAULT_CVAT_PASS,
        help="CVAT password.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Explicit confirmation required to execute the mutating rollback.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform all validation steps without modifying CVAT annotations.",
    )

    args = parser.parse_args()

    success = run_rollback(
        job_id=args.job_id,
        backup_file=args.backup_file,
        expected_sha256=args.expected_sha256,
        expected_shape_count=args.expected_shape_count,
        expected_frame_count=args.expected_frame_count,
        cvat_url=args.cvat_url,
        username=args.username,
        password=args.password,
        confirm=args.confirm,
        dry_run=args.dry_run,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
