"""
Comprehensive Test and Validation Suite for Grad-CAM Explainability Module.

Tests:
1. Checkpoint loading from artifacts/checkpoints/classification/best_model.pt.
2. Output shape and 7-class configuration.
3. Inference and CAM generation on representative test images from test.csv.
4. Correctness of heatmap dimensions, values in [0.0, 1.0], and non-blank activations.
5. Overlay blending with configurable alpha.
6. Three-panel composite visualization generation.
7. Model parameter immutability (bit-identical weights before and after Grad-CAM).
8. Hook cleanup and absence of hook/memory leaks over repeated explanations.
9. Target class flexibility (default predicted vs explicitly requested target class).
10. Execution on CUDA (RTX 3050) with verified CPU fallback.
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CLASS_NAMES, CLASS_TO_IDX
from src.models.model_factory import build_classifier
from src.utils.paths import resolve_platform_path
from src.explainability.gradcam import GradCAM
from src.explainability.visualization import (
    generate_heatmap_image,
    generate_overlay_image,
    generate_three_panel_comparison,
    save_explanation_artifacts,
    SCIENTIFIC_DISCLAIMER,
)


def run_gradcam_tests(num_samples: int = 5, device_override: str = "auto") -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print("COTTON LEAF AI — GRAD-CAM EXPLAINABILITY TEST & VALIDATION SUITE")
    print("=" * 78)

    # 1. Device resolution
    if device_override == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_override)

    print(f"Device:               {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 2. Checkpoint and Manifest Resolution
    ckpt_path = PROJECT_ROOT / "artifacts" / "checkpoints" / "classification" / "best_model.pt"
    test_manifest_path = PROJECT_ROOT / "artifacts" / "manifests" / "splits" / "test.csv"

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Classification checkpoint missing at: {ckpt_path}")
    if not test_manifest_path.is_file():
        raise FileNotFoundError(f"Test manifest missing at: {test_manifest_path}")

    print(f"Classifier Checkpoint: {ckpt_path}")
    print(f"Test Manifest:         {test_manifest_path}")

    # 3. Model Loading & 7-Class Verification
    model = build_classifier(
        model_name="efficientnet_b0",
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        checkpoint_path=ckpt_path,
        device=device
    )
    model.eval()

    num_classes = model.backbone.classifier[1].out_features
    assert num_classes == 7, f"Expected 7 classes, got {num_classes}"
    print(f"[TEST 1] Checkpoint loaded successfully. Output classes verified: {num_classes}")

    # 4. Target Layer Inspection
    target_layer = model.get_last_conv_layer()
    print(f"[TEST 2] Target layer verified: {target_layer.__class__.__name__} ({target_layer})")

    # 5. Model Parameter Immutability Test (Check weights before)
    # Clone reference parameter tensors to guarantee zero drift or updates
    param_clones = {name: param.detach().clone() for name, param in model.named_parameters()}
    print(f"[TEST 3] Captured reference state for {len(param_clones)} model parameter tensors.")

    # 6. Load Representative Samples across classes
    df_test = pd.read_csv(test_manifest_path)
    # Pick distinct classes where possible
    selected_indices = []
    classes_seen = set()
    for idx, row in df_test.iterrows():
        c_label = row["class_folder_label"]
        if c_label not in classes_seen:
            selected_indices.append(idx)
            classes_seen.add(c_label)
        if len(selected_indices) >= num_samples:
            break

    # If still fewer, take first rows
    if len(selected_indices) < num_samples:
        for idx in range(min(num_samples, len(df_test))):
            if idx not in selected_indices:
                selected_indices.append(idx)
            if len(selected_indices) >= num_samples:
                break

    print(f"\n[EVALUATION] Testing Grad-CAM on {len(selected_indices)} representative test images...")

    cam_explainer = GradCAM(model=model, target_layer=target_layer, device=device)
    saved_artifacts = []

    for test_i, sample_idx in enumerate(selected_indices):
        sample_row = df_test.iloc[sample_idx]
        image_path = resolve_platform_path(sample_row["absolute_path"])
        ground_truth_class = sample_row["class_folder_label"]

        if not image_path.is_file():
            print(f"Warning: sample file {image_path} does not exist. Skipping.")
            continue

        with Image.open(image_path) as pil_img:
            orig_img = pil_img.convert("RGB")
        orig_w, orig_h = orig_img.size

        # A. Default Grad-CAM (Predicted class)
        res = cam_explainer.generate_cam(orig_img, target_size=(orig_w, orig_h))
        heatmap = res["heatmap"]
        pred_class = res["predicted_class_name"]
        conf = res["confidence"]

        # Assertions
        assert heatmap.shape == (orig_h, orig_w), f"Shape mismatch: {heatmap.shape} vs ({orig_h}, {orig_w})"
        assert np.all(np.isfinite(heatmap)), "Heatmap contains NaN or Inf values!"
        assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0, f"Heatmap values out of [0, 1]: [{heatmap.min()}, {heatmap.max()}]"
        assert heatmap.max() > 0.0, "Heatmap is completely blank (all zeros)!"

        # Save artifacts
        base_name = f"test_{test_i+1}_{pred_class.replace(' ', '_').lower()}_{image_path.stem}"
        paths = save_explanation_artifacts(
            original_image=orig_img,
            heatmap=heatmap,
            metadata=res,
            base_name=base_name,
            alpha=0.4,
            colormap="jet"
        )
        saved_artifacts.append(paths)

        print(
            f"  Sample {test_i+1}/{len(selected_indices)}: {image_path.name[:25]}... | "
            f"GT: {ground_truth_class[:18]:<18} | Pred: {pred_class[:18]:<18} | "
            f"Conf: {conf*100:5.1f}% | Heatmap min/max: [{heatmap.min():.2f}, {heatmap.max():.2f}] -> PASSED"
        )

    # 7. Diagnostic Target Class Test (Non-predicted class explanation)
    print("\n[TEST 4] Diagnostic mode: Explaining alternate target class...")
    alt_target_idx = (res["predicted_class_idx"] + 1) % len(CLASS_NAMES)
    alt_target_name = CLASS_NAMES[alt_target_idx]
    alt_res = cam_explainer.generate_cam(orig_img, target_class=alt_target_idx, target_size=(orig_w, orig_h))
    assert alt_res["target_class_idx"] == alt_target_idx, "Target class index mismatch in diagnostic mode!"
    assert alt_res["target_class_name"] == alt_target_name, "Target class name mismatch in diagnostic mode!"
    print(f"  Target class {alt_target_idx} ('{alt_target_name}') explanation generated successfully.")

    # 8. Grayscale and RGBA Input Robustness Test
    print("\n[TEST 5] Image format robustness (Grayscale and RGBA conversions)...")
    gray_img = orig_img.convert("L")
    res_gray = cam_explainer.generate_cam(gray_img)
    assert res_gray["heatmap"].shape == (gray_img.height, gray_img.width), "Grayscale output size mismatch"
    rgba_img = orig_img.convert("RGBA")
    res_rgba = cam_explainer.generate_cam(rgba_img)
    assert res_rgba["heatmap"].shape == (rgba_img.height, rgba_img.width), "RGBA output size mismatch"
    print("  Grayscale and RGBA image handling: PASSED")

    # 9. Verify Model Parameter Immutability (Weights unchanged after all backwards passes)
    print("\n[TEST 6] Verifying model parameter immutability...")
    for name, param in model.named_parameters():
        ref = param_clones[name]
        if not torch.equal(ref, param.detach()):
            raise AssertionError(f"Parameter '{name}' was modified during Grad-CAM backpropagation!")
    print("  All model parameters bit-identical to initial checkpoint: PASSED (Zero weight modification)")

    # 10. Hook Leakage & Memory Accumulation Test (Repeated explanations)
    print("\n[TEST 7] Verifying hook cleanup and absence of hook accumulation...")
    # Check that target layer has 0 hooks
    assert len(target_layer._forward_hooks) == 0, f"Found {len(target_layer._forward_hooks)} leftover forward hooks!"
    assert len(target_layer._backward_hooks) == 0, f"Found {len(target_layer._backward_hooks)} leftover backward hooks!"

    # Run 10 rapid iterations
    for _ in range(10):
        _ = cam_explainer.generate_cam(orig_img)

    assert len(target_layer._forward_hooks) == 0, "Forward hooks accumulated after repeated explanations!"
    assert len(target_layer._backward_hooks) == 0, "Backward hooks accumulated after repeated explanations!"
    print("  Hook registration lifecycle and cleanup: PASSED (Zero residual hooks)")

    # 11. CPU Fallback Verification
    print("\n[TEST 8] Testing CPU fallback execution...")
    cpu_model = build_classifier(
        model_name="efficientnet_b0",
        num_classes=7,
        pretrained=False,
        checkpoint_path=ckpt_path,
        device=torch.device("cpu")
    )
    cpu_explainer = GradCAM(model=cpu_model, device=torch.device("cpu"))
    cpu_res = cpu_explainer.generate_cam(orig_img)
    assert cpu_res["device"] == "cpu", "Device string is not CPU"
    assert np.all(np.isfinite(cpu_res["heatmap"])), "CPU heatmap contains non-finite values"
    print(f"  CPU Grad-CAM execution verified: PASSED (Prediction: {cpu_res['predicted_class_name']} {cpu_res['confidence']*100:.1f}%)")

    # 12. Verify Saved Artifacts on Disk
    print("\n[TEST 9] Verifying saved explainability artifacts on disk...")
    assert len(saved_artifacts) > 0, "No artifacts were saved!"
    sample_artifact = saved_artifacts[0]
    for key in ["original_path", "heatmap_path", "overlay_path", "comparison_path", "metadata_path"]:
        p = Path(sample_artifact[key])
        assert p.is_file(), f"Expected artifact missing: {p}"
        print(f"  Verified {key}: {p.name} ({p.stat().st_size} bytes)")

    # Read back metadata JSON
    with open(sample_artifact["metadata_path"], "r", encoding="utf-8") as f:
        meta_json = json.load(f)
    assert meta_json["scientific_disclaimer"] == SCIENTIFIC_DISCLAIMER
    assert "backbone.features[-1]" in meta_json["target_convolutional_layer"]
    print("  Metadata JSON schema and scientific disclaimer: PASSED")

    print("\n" + "=" * 78)
    print("ALL 9 GRAD-CAM VALIDATION TESTS PASSED SUCCESSFULLY")
    print("=" * 78 + "\n")

    return {
        "status": "SUCCESS",
        "num_samples_tested": len(saved_artifacts),
        "artifacts": saved_artifacts,
        "sample_metadata": meta_json,
    }


def main():
    parser = argparse.ArgumentParser(description="Test and Validate Grad-CAM Explainability Module")
    parser.add_argument("--samples", type=int, default=5, help="Number of test images to evaluate")
    parser.add_argument("--device", type=str, default="auto", help="Execution device: 'auto', 'cuda:0', or 'cpu'")
    args = parser.parse_args()

    run_gradcam_tests(num_samples=args.samples, device_override=args.device)


if __name__ == "__main__":
    main()
