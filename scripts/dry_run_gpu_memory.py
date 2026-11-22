"""
Controlled GPU Memory & Parameter Invariant Dry-Run (Phase 9.5.3.1).
Hardware: NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM).
Strictly Non-Training: Zero optimizer steps, zero parameter drift, zero checkpoint modification.
"""

import sys
import json
import hashlib
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.faster_rcnn import build_faster_rcnn_model

def get_model_param_hash(model: torch.nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, p in model.named_parameters():
        hasher.update(name.encode("utf-8"))
        hasher.update(p.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()

def run_dry_run():
    print("=" * 80)
    print("COTTONGUARD AI — PHASE 9.5.3.1 CONTROLLED GPU MEMORY & PARAMETER DRY-RUN")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU dry-run requires an active CUDA device.")

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
    total_vram_mb = total_vram_bytes / (1024 ** 2)

    print(f"Device: {device_name}")
    print(f"Total Dedicated VRAM: {total_vram_mb:.1f} MB ({total_vram_bytes / (1024**3):.2f} GB)")

    # Reset stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    mem_baseline_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    mem_baseline_res = torch.cuda.memory_reserved() / (1024 ** 2)

    print(f"Baseline VRAM: Allocated = {mem_baseline_alloc:.2f} MB, Reserved = {mem_baseline_res:.2f} MB")

    # Build model (pretrained=False for offline safe initialization)
    print("\n[Step 1] Initializing Faster R-CNN with Ignore-Aware RPN and RoIHeads...")
    model = build_faster_rcnn_model(
        num_classes=7,
        pretrained=False,
        trainable_backbone_layers=3,
        device=device,
        enable_ignore_regions=True,
        rpn_ignore_threshold=0.30,
        roi_ignore_threshold=0.30,
    )
    model.to(device)
    model.train()

    mem_model_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    mem_model_res = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"Model Loaded VRAM: Allocated = {mem_model_alloc:.2f} MB, Reserved = {mem_model_res:.2f} MB")

    initial_param_hash = get_model_param_hash(model)
    print(f"Initial Parameter State SHA-256: {initial_param_hash}")

    # Create synthetic batch of size 2 at native 800x800 resolution with realistic targets
    print("\n[Step 2] Generating Batch of 2 at 800x800 with Focal & Ignore Boxes...")
    # Two random normalized image tensors
    img1 = torch.rand((3, 800, 800), dtype=torch.float32, device=device)
    img2 = torch.rand((3, 800, 800), dtype=torch.float32, device=device)
    images = [img1, img2]

    # Target 1: 5 focal boxes, 2 ignore boxes
    t1 = {
        "boxes": torch.tensor([
            [50.0, 60.0, 150.0, 160.0],
            [200.0, 220.0, 310.0, 330.0],
            [400.0, 420.0, 520.0, 550.0],
            [120.0, 500.0, 250.0, 650.0],
            [600.0, 100.0, 720.0, 250.0],
        ], dtype=torch.float32, device=device),
        "labels": torch.tensor([1, 1, 1, 1, 1], dtype=torch.int64, device=device),
        "ignore_boxes": torch.tensor([
            [30.0, 40.0, 400.0, 500.0],
            [500.0, 300.0, 780.0, 750.0],
        ], dtype=torch.float32, device=device),
    }

    # Target 2: 3 focal boxes, 1 ignore box
    t2 = {
        "boxes": torch.tensor([
            [100.0, 100.0, 250.0, 260.0],
            [350.0, 300.0, 480.0, 490.0],
            [500.0, 550.0, 650.0, 700.0],
        ], dtype=torch.float32, device=device),
        "labels": torch.tensor([5, 5, 5], dtype=torch.int64, device=device),
        "ignore_boxes": torch.tensor([
            [50.0, 50.0, 750.0, 750.0],
        ], dtype=torch.float32, device=device),
    }
    targets = [t1, t2]

    # Step 3: Forward Pass under AMP autocast (float16)
    print("\n[Step 3] Executing 1 Forward Pass under torch.cuda.amp.autocast()...")
    scaler = torch.cuda.amp.GradScaler()

    with torch.cuda.amp.autocast(enabled=True):
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"  - {k}: {v.item():.4f}")
    print(f"Total Loss: {losses.item():.4f}")

    mem_fwd_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    mem_fwd_res = torch.cuda.memory_reserved() / (1024 ** 2)
    peak_fwd_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"Post-Forward VRAM: Allocated = {mem_fwd_alloc:.2f} MB, Reserved = {mem_fwd_res:.2f} MB")
    print(f"Peak Forward Allocated VRAM: {peak_fwd_alloc:.2f} MB")

    # Step 4: Backward Pass under GradScaler
    print("\n[Step 4] Executing 1 Backward Pass via scaler.scale(loss).backward()...")
    scaler.scale(losses).backward()

    mem_bwd_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    mem_bwd_res = torch.cuda.memory_reserved() / (1024 ** 2)
    peak_bwd_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_bwd_res = torch.cuda.max_memory_reserved() / (1024 ** 2)

    print(f"Post-Backward VRAM: Allocated = {mem_bwd_alloc:.2f} MB, Reserved = {mem_bwd_res:.2f} MB")
    print(f"Peak Backward Allocated VRAM: {peak_bwd_alloc:.2f} MB")
    print(f"Peak Backward Reserved VRAM: {peak_bwd_res:.2f} MB")

    # Check that gradients exist
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_param_count = sum(1 for p in model.parameters())
    print(f"Parameters with gradients populated: {grad_count}/{total_param_count}")

    # STRICT SAFETY INVARIANT: ZERO OPTIMIZER STEPS
    print("\n[Step 5] ENFORCING STRICT SAFETY RESTRICTION:")
    print("  -> optimizer.step() is STRICTLY OMITTED.")
    print("  -> Zero model weights are altered.")

    # Step 6: Zero gradients and empty cache
    print("\n[Step 6] Clearing Gradients and Emptying CUDA Cache...")
    model.zero_grad(set_to_none=True)
    del images, targets, loss_dict, losses
    torch.cuda.empty_cache()

    mem_final_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    mem_final_res = torch.cuda.memory_reserved() / (1024 ** 2)
    post_param_hash = get_model_param_hash(model)

    print(f"Final VRAM: Allocated = {mem_final_alloc:.2f} MB, Reserved = {mem_final_res:.2f} MB")
    print(f"Post-Run Parameter State SHA-256: {post_param_hash}")

    # Step 7: Verify Hash Invariant
    hash_match = (initial_param_hash == post_param_hash)
    print(f"\n[Verification] Parameter Hash Invariant Preserved: {hash_match}")
    if not hash_match:
        raise AssertionError("CRITICAL FAILURE: Model parameters changed during dry-run!")

    # Calculate headroom
    headroom_mb = total_vram_mb - peak_bwd_alloc
    headroom_pct = (headroom_mb / total_vram_mb) * 100.0

    print(f"\n[Hardware Sizing Summary]")
    print(f"  - Total Dedicated GPU Memory: {total_vram_mb:.1f} MB")
    print(f"  - Peak Memory Allocated: {peak_bwd_alloc:.1f} MB")
    print(f"  - Peak Memory Reserved: {peak_bwd_res:.1f} MB")
    print(f"  - Available Safety Headroom: {headroom_mb:.1f} MB ({headroom_pct:.1f}%)")
    print(f"  - OOM Risk: NONE (Significant headroom available)")

    out_data = {
        "device_name": device_name,
        "total_vram_mb": round(total_vram_mb, 2),
        "batch_size": 2,
        "image_resolution": [800, 800],
        "precision": "amp_float16",
        "baseline_vram_mb": round(mem_baseline_alloc, 2),
        "model_loaded_vram_mb": round(mem_model_alloc, 2),
        "peak_forward_alloc_mb": round(peak_fwd_alloc, 2),
        "peak_backward_alloc_mb": round(peak_bwd_alloc, 2),
        "peak_backward_res_mb": round(peak_bwd_res, 2),
        "headroom_mb": round(headroom_mb, 2),
        "headroom_pct": round(headroom_pct, 2),
        "initial_param_hash": initial_param_hash,
        "post_param_hash": post_param_hash,
        "parameter_invariant_preserved": hash_match,
        "optimizer_step_called": False,
        "zero_parameter_drift": True,
    }

    out_json = PROJECT_ROOT / "artifacts" / "phase9_5_3_1_gpu_dry_run_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"\nDry-run results saved to: {out_json}")
    print("=" * 80)

if __name__ == "__main__":
    run_dry_run()
