"""
scripts/test_cvat_delete_mechanics.py
Tests CVAT 2.82.1 PATCH ?action=delete mechanics on disposable Job #2 (0 shapes).
Ensures zero impact on Task #6.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

WORKSPACE = Path(r"D:\Main_Project\cotton_leaf_ai")
sys.path.insert(0, str(WORKSPACE))

from scripts.restore_cvat_task6_baseline import CVATClient

CVAT_URL = os.environ.get("CVAT_URL", "http://localhost:8080")
CVAT_USER = os.environ.get("CVAT_USERNAME", "cottonguard")
CVAT_PASS = os.environ.get("CVAT_PASSWORD", "CottonGuardCVAT2026!")

TEST_JOB_ID = 2

def main():
    print(f"=== TESTING CVAT DELETION MECHANICS ON DISPOSABLE JOB #{TEST_JOB_ID} ===")
    client = CVATClient(CVAT_URL, CVAT_USER, CVAT_PASS)

    # 1. Verify Job 2 initial state and clean up if needed
    ann = client.get_annotations(TEST_JOB_ID)
    init_shapes = ann.get("shapes", [])
    print(f"Job #{TEST_JOB_ID} initial shapes: {len(init_shapes)}")
    if len(init_shapes) > 0:
        print(f"Cleaning existing {len(init_shapes)} shapes from Job #{TEST_JOB_ID}...")
        del_url = f"{CVAT_URL}/api/jobs/{TEST_JOB_ID}/annotations/?action=delete"
        req_clean = urllib.request.Request(
            del_url,
            data=json.dumps({"version": 0, "shapes": init_shapes, "tracks": [], "tags": []}).encode("utf-8"),
            headers=client._get_headers(),
            method="PATCH"
        )
        with urllib.request.urlopen(req_clean, timeout=15) as resp:
            print("Initial cleanup status:", resp.status)
        ann = client.get_annotations(TEST_JOB_ID)
        init_shapes = ann.get("shapes", [])
        print(f"Job #{TEST_JOB_ID} shapes after initial cleanup: {len(init_shapes)}")
    assert len(init_shapes) == 0

    # 2. Create 2 test shapes on Job 2
    test_shapes_payload = [
        {
            "type": "rectangle",
            "frame": 0,
            "label_id": 3,
            "points": [10.0, 10.0, 50.0, 50.0],
            "occluded": False,
            "outside": False,
            "z_order": 0,
            "rotation": 0.0,
            "attributes": []
        },
        {
            "type": "rectangle",
            "frame": 0,
            "label_id": 3,
            "points": [60.0, 60.0, 100.0, 100.0],
            "occluded": False,
            "outside": False,
            "z_order": 0,
            "rotation": 0.0,
            "attributes": []
        }
    ]

    print("Uploading 2 test shapes to Job #2 via PATCH ?action=create...")
    create_url = f"{CVAT_URL}/api/jobs/{TEST_JOB_ID}/annotations/?action=create"
    req = urllib.request.Request(
        create_url,
        data=json.dumps({"version": 0, "shapes": test_shapes_payload, "tracks": [], "tags": []}).encode("utf-8"),
        headers=client._get_headers(),
        method="PATCH"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        print(f"Create response status: {resp.status}")

    # Verify both exist
    ann_after_create = client.get_annotations(TEST_JOB_ID)
    created_shapes = ann_after_create.get("shapes", [])
    print(f"Job #{TEST_JOB_ID} shapes after create: {len(created_shapes)}")
    assert len(created_shapes) == 2
    id_to_delete = created_shapes[0]["id"]
    id_to_keep = created_shapes[1]["id"]
    print(f"  Target ID to delete: {id_to_delete}")
    print(f"  Target ID to keep:   {id_to_keep}")

    # 3. Test PATCH ?action=delete with shape object
    print(f"\nDeleting shape {id_to_delete} via PATCH ?action=delete...")
    del_url = f"{CVAT_URL}/api/jobs/{TEST_JOB_ID}/annotations/?action=delete"
    # Pass the shape object from created_shapes[0]
    shape_to_delete = created_shapes[0]
    del_payload = {
        "version": 0,
        "shapes": [shape_to_delete],
        "tracks": [],
        "tags": []
    }
    req_del = urllib.request.Request(
        del_url,
        data=json.dumps(del_payload).encode("utf-8"),
        headers=client._get_headers(),
        method="PATCH"
    )
    with urllib.request.urlopen(req_del, timeout=15) as resp:
        print(f"Delete response status: {resp.status}")

    # 4. Verify post-deletion state
    ann_after_del = client.get_annotations(TEST_JOB_ID)
    remaining_shapes = ann_after_del.get("shapes", [])
    print(f"\nJob #{TEST_JOB_ID} shapes after delete: {len(remaining_shapes)}")
    assert len(remaining_shapes) == 1
    assert remaining_shapes[0]["id"] == id_to_keep
    print("  -> PASS: Shape {id_to_delete} was deleted; Shape {id_to_keep} was preserved!")

    # 5. Clean up Job 2 completely back to 0 shapes
    print("\nCleaning up Job #2...")
    shape_to_keep = remaining_shapes[0]
    req_cleanup = urllib.request.Request(
        del_url,
        data=json.dumps({"version": 0, "shapes": [shape_to_keep], "tracks": [], "tags": []}).encode("utf-8"),
        headers=client._get_headers(),
        method="PATCH"
    )
    with urllib.request.urlopen(req_cleanup, timeout=15) as resp:
        pass

    final_ann = client.get_annotations(TEST_JOB_ID)
    final_shapes = final_ann.get("shapes", [])
    print(f"Job #{TEST_JOB_ID} final shapes after cleanup: {len(final_shapes)}")
    assert len(final_shapes) == 0
    print("=== DISPOSABLE TEST JOB #2 CLEANUP COMPLETE (0 shapes) ===")

if __name__ == "__main__":
    main()
