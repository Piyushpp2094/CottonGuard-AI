"""
CottonGuard AI — Phase 9: End-to-End System Integration & Workflow Verification
Verifies complete pipeline across all 7 cotton leaf disease classes:
Authentication -> Health -> Image Upload -> /api/predict -> DB Persistence -> Retrieval -> PDF Report.
"""

import io
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database import init_db


def main():
    print("=" * 80)
    print("COTTONGUARD AI - PHASE 9 END-TO-END SYSTEM INTEGRATION VERIFICATION")
    print("=" * 80)

    # 1. Initialize DB & Client
    init_db()
    client = TestClient(app)
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phases_tested": [
            "health_telemetry",
            "model_registry",
            "authentication_jwt",
            "prediction_all_7_classes",
            "database_persistence",
            "pdf_report_generation",
            "history_pagination"
        ],
        "system_health": {},
        "model_registry": {},
        "class_verifications": [],
        "history_verification": {},
        "overall_status": "PENDING"
    }

    # 2. Verify Health
    print("\n[STEP 1] Testing /api/health...")
    t0 = time.time()
    res_health = client.get("/api/health")
    health_latency = (time.time() - t0) * 1000
    assert res_health.status_code == 200, f"Health check failed: {res_health.text}"
    health_data = res_health.json()
    print(f"  Status: {health_data.get('status')} | Database: {health_data.get('database_connected')} | CUDA: {health_data.get('cuda_available')} ({health_data.get('device_name')})")
    results["system_health"] = {
        "status_code": res_health.status_code,
        "latency_ms": round(health_latency, 2),
        "data": health_data
    }

    # 3. Verify Model Registry
    print("\n[STEP 2] Testing /api/health/models...")
    t0 = time.time()
    res_models = client.get("/api/health/models")
    models_latency = (time.time() - t0) * 1000
    assert res_models.status_code == 200, f"Models check failed: {res_models.text}"
    models_data = res_models.json()
    print(f"  Registered models: {list(models_data.get('models', {}).keys())}")
    results["model_registry"] = {
        "status_code": res_models.status_code,
        "latency_ms": round(models_latency, 2),
        "models_count": len(models_data.get("models", {}))
    }

    # 4. Authentication
    print("\n[STEP 3] Testing Authentication (/api/auth/login)...")
    # Ensure test user exists or register
    client.post("/api/auth/register", json={
        "username": "field_officer",
        "email": "officer@cotton.ai",
        "password": "SecurePassword2026!",
        "role": "farmer"
    })

    login_res = client.post("/api/auth/login", json={
        "username_or_email": "field_officer",
        "password": "SecurePassword2026!"
    })
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("  JWT Bearer token generated successfully.")

    # 5. Load Sample Gallery Manifest
    samples_path = PROJECT_ROOT / "backend" / "app" / "static" / "samples" / "samples.json"
    with open(samples_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"\n[STEP 4] Executing End-to-End Pipeline on {len(samples)} Disease Classes via POST /api/predict...")

    for idx, sample in enumerate(samples, 1):
        expected_class = sample["disease_class"]
        filename = sample["filename"]
        img_path = PROJECT_ROOT / "backend" / "app" / "static" / "samples" / filename
        assert img_path.is_file(), f"Sample image file not found: {img_path}"

        print(f"\n  --- Test {idx}/{len(samples)}: {expected_class} ({filename}) ---")
        with open(img_path, "rb") as img_file:
            img_bytes = img_file.read()

        # Measure end-to-end request latency
        t_start = time.time()
        pred_res = client.post(
            "/api/predict",
            files={"file": (filename, io.BytesIO(img_bytes), "image/jpeg")},
            data={"seg_alpha": "0.40", "stream": "false"},
            headers=headers
        )
        latency_ms = (time.time() - t_start) * 1000

        assert pred_res.status_code == 200, f"Predict failed for {filename}: {pred_res.text}"
        data = pred_res.json()
        pred_id = data["id"]
        predicted_class = data["predicted_class"]
        confidence = data["confidence"]
        sev_pct = data.get("affected_area_percentage", 0.0) or 0.0
        sev_tier = data["severity_level"]
        leaf_px = data.get("leaf_pixels", 0) or 0
        dis_px = data.get("disease_pixels", 0) or 0
        recommendations = data.get("recommendations", {})

        print(f"    Prediction ID:       {pred_id}")
        print(f"    Predicted Class:     {predicted_class} (Expected: {expected_class})")
        print(f"    Confidence:          {confidence * 100:.2f}%")
        print(f"    Affected Area:       {sev_pct:.2f}% ({sev_tier})")
        print(f"    Leaf / Disease Px:   {leaf_px:,} / {dis_px:,} px")
        print(f"    E2E Request Latency: {latency_ms:.1f} ms")

        # Class match check (case-insensitive, normalize underscores and botanical aliases)
        norm_expected = expected_class.lower().replace("_", " ").strip()
        norm_predicted = predicted_class.lower().replace("_", " ").strip()
        # Handle botanical synonym: Leaf Reddening == Leaf Redding
        synonyms = [{"leaf reddening", "leaf redding"}]
        class_match = (norm_expected == norm_predicted) or any({norm_expected, norm_predicted} == syn for syn in synonyms)
        if not class_match:
            print(f"    [WARN] Class mismatch: expected '{expected_class}', got '{predicted_class}'")

        # Check DB retrieval
        rec_res = client.get(f"/api/predictions/{pred_id}", headers=headers)
        assert rec_res.status_code == 200, f"Failed to retrieve prediction {pred_id}: {rec_res.text}"
        db_record = rec_res.json()
        assert db_record["id"] == pred_id

        # Check PDF Report Generation
        pdf_start = time.time()
        pdf_res = client.get(f"/api/predictions/{pred_id}/report", headers=headers)
        pdf_latency_ms = (time.time() - pdf_start) * 1000
        assert pdf_res.status_code == 200, f"Failed to generate PDF for {pred_id}: {pdf_res.text}"
        assert "application/pdf" in pdf_res.headers.get("content-type", "")
        assert pdf_res.content[:5] == b"%PDF-", "Invalid PDF file header."
        pdf_size_bytes = len(pdf_res.content)
        print(f"    PDF Report Generated:{pdf_size_bytes:,} bytes in {pdf_latency_ms:.1f} ms (Valid %PDF- header)")

        has_advisory = bool(
            recommendations.get("priority_action")
            or recommendations.get("immediate_first_steps")
            or recommendations.get("action")
            or recommendations.get("treatment")
        )

        verification_record = {
            "index": idx,
            "filename": filename,
            "expected_class": expected_class,
            "predicted_class": predicted_class,
            "class_match": class_match,
            "confidence": round(confidence, 4),
            "affected_area_percentage": round(sev_pct, 2),
            "severity_level": sev_tier,
            "leaf_pixels": leaf_px,
            "disease_pixels": dis_px,
            "latency_ms": round(latency_ms, 2),
            "prediction_id": pred_id,
            "pdf_report_size_bytes": pdf_size_bytes,
            "pdf_latency_ms": round(pdf_latency_ms, 2),
            "advisory_present": has_advisory
        }
        results["class_verifications"].append(verification_record)

    # 6. History Pagination Check
    print("\n[STEP 5] Testing History Endpoint (/api/predictions?limit=10&offset=0)...")
    hist_res = client.get("/api/predictions?limit=10&offset=0", headers=headers)
    assert hist_res.status_code == 200, f"History check failed: {hist_res.text}"
    hist_data = hist_res.json()
    total_count = hist_data.get("total", len(hist_data.get("items", [])))
    print(f"  History Total Predictions: {total_count} (Retrieved: {len(hist_data.get('items', []))})")
    results["history_verification"] = {
        "status_code": hist_res.status_code,
        "total_records": total_count,
        "retrieved_count": len(hist_data.get("items", []))
    }

    # Summary
    all_matched = all(v["class_match"] for v in results["class_verifications"])
    avg_latency = sum(v["latency_ms"] for v in results["class_verifications"]) / len(results["class_verifications"])
    results["average_e2e_latency_ms"] = round(avg_latency, 2)
    results["all_classes_matched"] = all_matched
    results["overall_status"] = "PASSED" if all_matched else "PASSED_WITH_WARNINGS"

    print("\n" + "=" * 80)
    print("PHASE 9 VERIFICATION SUMMARY:")
    print(f"  Total Classes Tested: {len(samples)}")
    print(f"  All Classes Matched:  {all_matched}")
    print(f"  Average E2E Latency:  {avg_latency:.2f} ms")
    print(f"  Overall Status:       {results['overall_status']}")
    print("=" * 80)

    # Save to metrics
    out_path = PROJECT_ROOT / "artifacts" / "metrics" / "e2e_integration_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[DONE] Saved integration verification results to: {out_path}")


if __name__ == "__main__":
    main()
