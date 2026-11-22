"""
Automated verification script for all Cotton Leaf AI Backend APIs.
"""

import io
import json
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app


def main():
    print("=" * 70)
    print("COTTON LEAF AI - BACKEND API END-TO-END VERIFICATION")
    print("=" * 70)

    client = TestClient(app)

    # 1. Health check
    print("\n1. Testing Health Endpoint (/api/health)...")
    res = client.get("/api/health")
    print(f"   Status: {res.status_code}")
    print(f"   Response: {json.dumps(res.json(), indent=6)}")
    assert res.status_code == 200

    # 2. Authentication
    print("\n2. Testing Authentication...")
    reg_res = client.post("/api/auth/register", json={
        "username": "field_officer",
        "email": "officer@cotton.ai",
        "password": "SecurePassword2026!",
        "role": "farmer"
    })
    # Accept 201 or 400 (if already registered)
    print(f"   Register Status: {reg_res.status_code}")

    login_res = client.post("/api/auth/login", json={
        "username_or_email": "field_officer",
        "password": "SecurePassword2026!"
    })
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token = login_res.json()["access_token"]
    print(f"   Login Status: 200 OK (JWT Token generated successfully)")

    headers = {"Authorization": f"Bearer {token}"}
    me_res = client.get("/api/auth/me", headers=headers)
    assert me_res.status_code == 200
    print(f"   Profile Check: Logged in as '{me_res.json()['username']}' ({me_res.json()['role']})")

    # 3. Image Analysis & Inference
    print("\n3. Testing End-to-End Image Inference (/api/predictions/analyze)...")
    buf = io.BytesIO()
    test_leaf = Image.new("RGB", (224, 224), color=(60, 150, 45))
    test_leaf.save(buf, format="JPEG")
    buf.seek(0)

    pred_res = client.post(
        "/api/predictions/analyze",
        files={"file": ("cotton_leaf_test.jpg", buf, "image/jpeg")},
        headers=headers
    )
    assert pred_res.status_code == 200, f"Analyze failed: {pred_res.text}"
    pred_data = pred_res.json()
    pred_id = pred_data["id"]

    print(f"   Prediction ID:       #{pred_id}")
    print(f"   Predicted Condition: {pred_data['predicted_class']}")
    print(f"   Confidence:          {pred_data['confidence'] * 100:.2f}%")
    print(f"   Severity Level:      {pred_data['severity_level']}")
    print(f"   Affected Area:       {pred_data['affected_area_percentage']}")
    print(f"   Recommendations:")
    for step in pred_data["recommendations"]["immediate_first_steps"][:2]:
        print(f"     - {step}")

    # 4. History
    print("\n4. Testing Prediction History (/api/predictions/history)...")
    hist_res = client.get("/api/predictions/history", headers=headers)
    assert hist_res.status_code == 200
    print(f"   History Status: 200 OK ({len(hist_res.json())} user prediction records retrieved)")

    # 5. Dashboard Statistics
    print("\n5. Testing Dashboard Statistics (/api/dashboard/statistics)...")
    dash_res = client.get("/api/dashboard/statistics", headers=headers)
    assert dash_res.status_code == 200
    stats = dash_res.json()
    print(f"   Total Analyzed: {stats['total_analyzed']}")
    print(f"   Healthy Count:  {stats['healthy_count']}")
    print(f"   Diseased Count: {stats['diseased_count']}")
    print(f"   Condition Breakdown: {stats['condition_distribution']}")

    # 6. PDF Report
    print(f"\n6. Testing PDF Report Generation (/api/reports/{pred_id})...")
    rep_res = client.get(f"/api/reports/{pred_id}", headers=headers)
    assert rep_res.status_code == 200
    assert rep_res.headers["content-type"] == "application/pdf"
    assert rep_res.content.startswith(b"%PDF-")
    print(f"   Report Status: 200 OK (Content-Type: application/pdf, Size: {len(rep_res.content):,} bytes)")

    # 7. Grad-CAM Explainability Endpoint
    print("\n7. Testing Grad-CAM Explainability Endpoint (/api/explainability/gradcam)...")
    cam_buf = io.BytesIO()
    test_leaf.save(cam_buf, format="JPEG")
    cam_buf.seek(0)

    cam_res = client.post(
        "/api/explainability/gradcam",
        files={"file": ("cotton_leaf_gradcam_test.jpg", cam_buf, "image/jpeg")},
        data={"alpha": "0.4", "colormap": "jet"},
        headers=headers
    )
    assert cam_res.status_code == 200, f"Grad-CAM failed: {cam_res.text}"
    cam_data = cam_res.json()
    print(f"   Target Layer:        {cam_data['target_layer']}")
    print(f"   Predicted Condition: {cam_data['predicted_class']}")
    print(f"   Confidence:          {cam_data['confidence'] * 100:.2f}%")
    print(f"   Heatmap URL:         {cam_data['heatmap_url']}")
    print(f"   Overlay URL:         {cam_data['overlay_url']}")
    print(f"   Comparison URL:      {cam_data['comparison_url']}")
    print(f"   Disclaimer:          {cam_data['scientific_disclaimer'][:65]}...")

    # Verify static file serving for generated artifacts
    for url_key in ["heatmap_url", "overlay_url", "comparison_url"]:
        asset_res = client.get(cam_data[url_key])
        assert asset_res.status_code == 200, f"Failed to retrieve static asset: {cam_data[url_key]}"
        assert len(asset_res.content) > 1000, f"Asset too small: {cam_data[url_key]}"
    print("   Static Asset Serving: 200 OK for Heatmap, Overlay, and Comparison images")

    print("\n" + "=" * 70)
    print("ALL BACKEND ENDPOINTS AND INFERENCE PIPELINES VERIFIED (100% PASS)")
    print("=" * 70)


if __name__ == "__main__":
    main()
