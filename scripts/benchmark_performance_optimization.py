"""
CottonGuard AI — Phase 11: Performance Benchmarking & Hardware Optimization
Profiles inference latency, throughput (FPS), peak VRAM allocation, and memory stability
across EfficientNet-B0, U-Net ResNet-34, Faster R-CNN, Grad-CAM, and the unified E2E pipeline
on NVIDIA GeForce RTX 3050 6GB Laptop GPU and CPU fallback.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CLASS_NAMES
from src.models.model_factory import build_classifier
from src.models.unet_resnet34 import build_unet_resnet34
from src.explainability.gradcam import GradCAM
from backend.app.services.prediction_service import prediction_service


def get_peak_vram_mb() -> float:
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    return 0.0


def reset_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def benchmark_tensor_model(model, input_tensor, device, iterations=50, warmup=10):
    model.eval()
    model.to(device)
    x = input_tensor.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timing
    reset_vram()
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

    mean_lat = float(np.mean(latencies))
    std_lat = float(np.std(latencies))
    fps = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0
    vram = get_peak_vram_mb() if device.type == "cuda" else 0.0

    return {
        "mean_latency_ms": round(mean_lat, 2),
        "std_latency_ms": round(std_lat, 2),
        "min_latency_ms": round(float(np.min(latencies)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "fps": round(fps, 1),
        "peak_vram_mb": vram
    }


def main():
    print("=" * 80)
    print("COTTONGUARD AI - HARDWARE PERFORMANCE & OPTIMIZATION BENCHMARK")
    print("=" * 80)

    device_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")

    telemetry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0
    }
    print(f"Hardware: {telemetry['device_name']} ({telemetry['vram_total_mb']} MB VRAM)")
    print(f"PyTorch:  {telemetry['pytorch_version']} | CUDA: {torch.version.cuda}")

    benchmark_data = {
        "telemetry": telemetry,
        "models": {},
        "pipeline_e2e": {},
        "stress_memory_profile": {}
    }

    # 1. Benchmark EfficientNet-B0 Classifier
    print("\n[1/5] Benchmarking EfficientNet-B0 Classifier (224x224 RGB)...")
    clf_gpu = prediction_service.classifier
    clf_params = sum(p.numel() for p in clf_gpu.parameters())
    input_cls = torch.randn(1, 3, 224, 224)

    gpu_cls = benchmark_tensor_model(clf_gpu, input_cls, device_gpu, iterations=50)

    # CPU copy
    clf_cpu = build_classifier(
        model_name="efficientnet_b0",
        num_classes=len(CLASS_NAMES),
        pretrained=False,
        checkpoint_path=prediction_service.classifier_ckpt,
        device=device_cpu,
    )
    cpu_cls = benchmark_tensor_model(clf_cpu, input_cls, device_cpu, iterations=20)
    del clf_cpu

    print(f"  EfficientNet-B0: {clf_params:,} parameters (4.02M)")
    print(f"  -> GPU: {gpu_cls['mean_latency_ms']} ms ({gpu_cls['fps']} FPS) | Peak VRAM: {gpu_cls['peak_vram_mb']} MB")
    print(f"  -> CPU: {cpu_cls['mean_latency_ms']} ms ({cpu_cls['fps']} FPS)")

    benchmark_data["models"]["classifier_efficientnet_b0"] = {
        "parameters": clf_params,
        "parameters_m": round(clf_params / 1e6, 2),
        "gpu": gpu_cls,
        "cpu": cpu_cls,
        "speedup_factor": round(cpu_cls["mean_latency_ms"] / max(gpu_cls["mean_latency_ms"], 0.001), 2)
    }

    # 2. Benchmark U-Net ResNet-34 Segmentor
    print("\n[2/5] Benchmarking U-Net ResNet-34 Segmentor (224x224 RGB -> 3-class mask)...")
    unet_gpu = prediction_service.unet
    unet_params = sum(p.numel() for p in unet_gpu.parameters())
    input_unet = torch.randn(1, 3, 224, 224)

    gpu_unet = benchmark_tensor_model(unet_gpu, input_unet, device_gpu, iterations=50)

    unet_cpu = build_unet_resnet34(
        num_classes=3,
        checkpoint_path=prediction_service.unet_ckpt,
        device=device_cpu
    )
    cpu_unet = benchmark_tensor_model(unet_cpu, input_unet, device_cpu, iterations=20)
    del unet_cpu

    print(f"  U-Net ResNet-34: {unet_params:,} parameters (24.4M)")
    print(f"  -> GPU: {gpu_unet['mean_latency_ms']} ms ({gpu_unet['fps']} FPS) | Peak VRAM: {gpu_unet['peak_vram_mb']} MB")
    print(f"  -> CPU: {cpu_unet['mean_latency_ms']} ms ({cpu_unet['fps']} FPS)")

    benchmark_data["models"]["segmentor_unet_resnet34"] = {
        "parameters": unet_params,
        "parameters_m": round(unet_params / 1e6, 2),
        "gpu": gpu_unet,
        "cpu": cpu_unet,
        "speedup_factor": round(cpu_unet["mean_latency_ms"] / max(gpu_unet["mean_latency_ms"], 0.001), 2)
    }

    # 3. Benchmark Grad-CAM Explainability
    print("\n[3/5] Benchmarking Grad-CAM Explainability Generation...")
    gradcam_latencies = []
    reset_vram()
    for _ in range(30):
        t0 = time.perf_counter()
        _ = prediction_service.cam_explainer.generate_cam(input_cls.to(device_gpu), target_class=0)
        if device_gpu.type == "cuda":
            torch.cuda.synchronize()
        gradcam_latencies.append((time.perf_counter() - t0) * 1000)

    gradcam_vram = get_peak_vram_mb()
    mean_gradcam = float(np.mean(gradcam_latencies))
    print(f"  -> Grad-CAM GPU: {mean_gradcam:.2f} ms | Peak VRAM: {gradcam_vram} MB")

    benchmark_data["models"]["explainability_gradcam"] = {
        "mean_latency_ms": round(mean_gradcam, 2),
        "std_latency_ms": round(float(np.std(gradcam_latencies)), 2),
        "fps": round(1000.0 / mean_gradcam, 1),
        "peak_vram_mb": gradcam_vram
    }

    # 4. Benchmark Unified End-to-End Pipeline
    print("\n[4/5] Benchmarking Full AI Multi-Model Pipeline (Preprocessing -> Clf -> Det -> UNet -> GradCAM)...")
    sample_img_path = PROJECT_ROOT / "backend" / "app" / "static" / "samples" / "BBC00226.jpg"
    test_img = Image.open(sample_img_path).convert("RGB")

    def run_full_pipeline(img):
        _p_class, _conf, _probs, _dev, _ = prediction_service._run_classifier_inference(img)
        _det = prediction_service._run_detection(img, base_name="bench_det", save_visual=False)
        _seg = prediction_service._run_unet_segmentation(img, base_name="bench_seg", alpha=0.40)
        _cam = prediction_service.cam_explainer.generate_cam(img, target_class=_p_class)
        return _p_class, _det, _seg, _cam

    # Warmup
    _ = run_full_pipeline(test_img)

    e2e_latencies = []
    reset_vram()
    for _ in range(30):
        t0 = time.perf_counter()
        _ = run_full_pipeline(test_img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        e2e_latencies.append((time.perf_counter() - t0) * 1000)

    mean_e2e = float(np.mean(e2e_latencies))
    e2e_vram = get_peak_vram_mb()
    print(f"  -> Full AI Pipeline GPU Latency: {mean_e2e:.2f} ms ({1000.0 / mean_e2e:.1f} FPS)")
    print(f"  -> Total Co-Loaded Models Peak VRAM: {e2e_vram} MB (Budget: 6,143.5 MB, Usage: {e2e_vram / 6143.5 * 100:.1f}%)")

    benchmark_data["pipeline_e2e"] = {
        "mean_latency_ms": round(mean_e2e, 2),
        "std_latency_ms": round(float(np.std(e2e_latencies)), 2),
        "min_latency_ms": round(float(np.min(e2e_latencies)), 2),
        "max_latency_ms": round(float(np.max(e2e_latencies)), 2),
        "fps": round(1000.0 / mean_e2e, 1),
        "peak_vram_mb": e2e_vram,
        "vram_utilization_pct": round(e2e_vram / 6143.5 * 100, 2)
    }

    # 5. Stress Memory Leak Profile (50 continuous inferences)
    print("\n[5/5] Executing 50-Cycle Continuous Inference Memory Leak Stress Test...")
    memory_checkpoints = []
    for i in range(1, 51):
        _ = run_full_pipeline(test_img)
        if i % 10 == 0:
            current_allocated = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
            memory_checkpoints.append({"cycle": i, "allocated_mb": current_allocated})
            print(f"  Cycle {i:02d}: Allocated VRAM = {current_allocated} MB")

    vram_growth = memory_checkpoints[-1]["allocated_mb"] - memory_checkpoints[0]["allocated_mb"]
    no_leak = abs(vram_growth) < 1.0  # Less than 1MB change across 40 cycles
    print(f"  VRAM Drift across 40 cycles: {vram_growth:+.2f} MB -> Memory Leak Status: {'NO LEAK (STABLE)' if no_leak else 'WARNING: LEAK DETECTED'}")

    benchmark_data["stress_memory_profile"] = {
        "cycles_executed": 50,
        "checkpoints": memory_checkpoints,
        "vram_growth_mb": round(vram_growth, 2),
        "memory_leak_free": no_leak
    }

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY COMPLETE:")
    print(f"  EfficientNet-B0 Latency:  {gpu_cls['mean_latency_ms']} ms ({gpu_cls['fps']} FPS)")
    print(f"  U-Net ResNet-34 Latency: {gpu_unet['mean_latency_ms']} ms ({gpu_unet['fps']} FPS)")
    print(f"  Grad-CAM Latency:        {mean_gradcam:.2f} ms")
    print(f"  Full E2E Pipeline:       {mean_e2e:.2f} ms ({benchmark_data['pipeline_e2e']['fps']} FPS)")
    print(f"  Total Peak VRAM:         {e2e_vram} MB / 6,143.5 MB ({e2e_vram / 6143.5 * 100:.1f}%)")
    print(f"  Memory Stability:        {'PASS (ZERO LEAK)' if no_leak else 'FAIL'}")
    print("=" * 80)

    out_file = PROJECT_ROOT / "artifacts" / "metrics" / "performance_benchmark_rtx3050.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"\n[DONE] Persisted hardware benchmark results to: {out_file}")


if __name__ == "__main__":
    main()
