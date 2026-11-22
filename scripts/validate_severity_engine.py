"""
CottonGuard AI — Phase 4 Disease Severity Estimation & Advisory Validation
Validates scientific severity formula, edge cases, 4-tier & 6-tier scales,
and ICAR-CICR recommendation engine integration.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.area_calculation import calculate_affected_area_from_mask
from src.metrics.severity import SeverityEstimator
from src.recommendations.recommendation_engine import RecommendationEngine


def test_edge_cases():
    print("[EDGE CASES] Testing synthetic edge cases...")
    estimator = SeverityEstimator()
    results = {}

    # Case 1: Pure Background (0 leaf pixels)
    mask_empty = np.zeros((200, 200), dtype=np.uint8)
    res_empty = calculate_affected_area_from_mask(mask_empty)
    eval_empty = estimator.evaluate(
        res_empty["affected_area_percentage"], "Bacterial Blight", 0.95,
        no_leaf_detected=(not res_empty["is_valid"])
    )
    assert not res_empty["is_valid"]
    assert eval_empty["level"] == "UNAVAILABLE"
    results["pure_background"] = {"status": "PASS", "result": res_empty["status"]}

    # Case 2: Insufficient Leaf Area (< 500 px, e.g. 400 px)
    mask_small = np.zeros((200, 200), dtype=np.uint8)
    mask_small[:20, :20] = 1 # 400 pixels
    res_small = calculate_affected_area_from_mask(mask_small)
    assert not res_small["is_valid"]
    assert res_small["status"] == "INSUFFICIENT_LEAF_AREA"
    results["insufficient_leaf_area"] = {"status": "PASS", "leaf_px": res_small["leaf_pixels"]}

    # Case 3: 0% Disease (100% Healthy Leaf)
    mask_healthy = np.zeros((200, 200), dtype=np.uint8)
    mask_healthy[:100, :100] = 1 # 10,000 px healthy
    res_healthy = calculate_affected_area_from_mask(mask_healthy)
    eval_healthy = estimator.evaluate(
        res_healthy["affected_area_percentage"], "Healthy Leaf", 0.98,
        leaf_pixels=res_healthy["leaf_pixels"], disease_pixels=res_healthy["disease_pixels"]
    )
    assert res_healthy["affected_area_percentage"] == 0.0
    assert eval_healthy["severity_category"] == "None"
    assert eval_healthy["level"] == "NONE"
    results["healthy_leaf_zero_lesion"] = {"status": "PASS", "affected_pct": 0.0, "tier": eval_healthy["severity_category"]}

    # Case 4: 100% Disease (Severe Necrosis)
    mask_severe = np.zeros((200, 200), dtype=np.uint8)
    mask_severe[:100, :100] = 2 # 10,000 px diseased
    res_severe = calculate_affected_area_from_mask(mask_severe)
    eval_severe = estimator.evaluate(
        res_severe["affected_area_percentage"], "Bacterial Blight", 0.95,
        leaf_pixels=res_severe["leaf_pixels"], disease_pixels=res_severe["disease_pixels"]
    )
    assert res_severe["affected_area_percentage"] == 100.0
    assert eval_severe["severity_category"] == "Very Severe"
    assert eval_severe["level"] == "HIGH"
    results["full_disease_100pct"] = {"status": "PASS", "affected_pct": 100.0, "tier": eval_severe["severity_category"]}

    # Case 5: Background Diseased Exclusion
    # 5,000 px leaf (class 1), 1,000 px diseased inside leaf (class 2), 2,000 px diseased in background
    # Since area_calculation checks mask_array >= 1, if background is marked 2, does it count?
    # By default, in 3-class segmentation: 0=bg, 1=healthy, 2=disease. If a pixel is 2, it is leaf disease.
    # But if leaf_mask_override is supplied, diseased pixels outside the leaf mask are strictly excluded!
    leaf_override = np.zeros((200, 200), dtype=np.uint8)
    leaf_override[:100, :50] = 1 # 5,000 px leaf
    mask_test = np.zeros((200, 200), dtype=np.uint8)
    mask_test[:100, :50] = 1 # healthy leaf
    mask_test[:20, :50] = 2 # 1,000 px diseased inside leaf
    mask_test[150:, 150:] = 2 # 2,500 px spurious diseased outside leaf
    res_restricted = calculate_affected_area_from_mask(mask_test, leaf_mask_override=leaf_override)
    assert res_restricted["disease_pixels"] == 1000
    assert res_restricted["leaf_pixels"] == 5000
    assert res_restricted["affected_area_percentage"] == 20.0
    results["background_exclusion"] = {"status": "PASS", "disease_px_inside_leaf": 1000, "affected_pct": 20.0}

    print("[EDGE CASES] All 5 synthetic edge cases passed successfully.")
    return results


def evaluate_70_dataset_severity():
    print("[DATASET] Evaluating severity on 70 development images...")
    manifest_path = PROJECT_ROOT / "artifacts" / "manifests" / "derived_70_image_dataset_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    records = manifest["records"]

    estimator = SeverityEstimator()
    rec_engine = RecommendationEngine()

    tier_counts_4 = {"HEALTHY (0%)": 0, "MILD (0.01-10%)": 0, "MODERATE (10.01-25%)": 0, "SEVERE (>25%)": 0}
    tier_counts_6 = {"None": 0, "Trace": 0, "Mild": 0, "Moderate": 0, "Severe": 0, "Very Severe": 0}
    class_severities = {}

    detailed_records = []

    for r in records:
        gid = r["global_id"]
        fname = r["filename"]
        dclass = r["disease_class"]
        pct = float(r["affected_leaf_pct"])

        # 4-tier clinical mapping
        if pct == 0.0 or dclass == "Healthy Leaf":
            tier_4 = "HEALTHY (0%)"
        elif pct <= 10.0:
            tier_4 = "MILD (0.01-10%)"
        elif pct <= 25.0:
            tier_4 = "MODERATE (10.01-25%)"
        else:
            tier_4 = "SEVERE (>25%)"
        tier_counts_4[tier_4] += 1

        # 6-tier evaluation
        eval_res = estimator.evaluate(pct, dclass, 0.95, leaf_pixels=r["total_leaf_pixels"], disease_pixels=r["diseased_pixels"])
        cat_6 = eval_res["severity_category"]
        tier_counts_6[cat_6] = tier_counts_6.get(cat_6, 0) + 1

        # Recommendation evaluation
        rec = rec_engine.get_recommendation(dclass, 0.95, severity_level=eval_res["level"], affected_area_percentage=pct)
        has_priority = (rec.get("priority_action") is not None)

        class_severities.setdefault(dclass, []).append(pct)

        detailed_records.append({
            "global_id": gid,
            "filename": fname,
            "disease_class": dclass,
            "affected_leaf_pct": pct,
            "tier_4": tier_4,
            "tier_6": cat_6,
            "legacy_level": eval_res["level"],
            "has_severity_priority_action": has_priority
        })

    class_stats = {
        cls: {
            "count": len(vals),
            "mean_severity_pct": round(float(np.mean(vals)), 2),
            "max_severity_pct": round(float(np.max(vals)), 2),
            "min_severity_pct": round(float(np.min(vals)), 2)
        }
        for cls, vals in class_severities.items()
    }

    print("\n4-Tier Clinical Severity Distribution (70 Development Images):")
    for k, v in tier_counts_4.items():
        print(f"  {k}: {v} ({v/70*100:.1f}%)")

    print("\n6-Tier Research Distribution:")
    for k, v in tier_counts_6.items():
        print(f"  {k}: {v} ({v/70*100:.1f}%)")

    print("\nPer-Class Severity Statistics:")
    for k, v in class_stats.items():
        print(f"  {k}: Mean {v['mean_severity_pct']}% | Max {v['max_severity_pct']}% | Min {v['min_severity_pct']}%")

    output_data = {
        "summary": {
            "total_samples": len(records),
            "mean_affected_percentage": round(float(np.mean([r['affected_leaf_pct'] for r in records])), 2),
            "tier_distribution_4": tier_counts_4,
            "tier_distribution_6": tier_counts_6,
            "per_class_statistics": class_stats
        },
        "records": detailed_records
    }

    out_path = PROJECT_ROOT / "artifacts" / "metrics" / "severity_validation_results.json"
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[SUCCESS] Saved severity validation results to {out_path}")
    return output_data


if __name__ == "__main__":
    edge_results = test_edge_cases()
    dataset_results = evaluate_70_dataset_severity()
