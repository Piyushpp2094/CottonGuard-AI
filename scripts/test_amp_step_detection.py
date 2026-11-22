"""
Synthetic AMP Optimizer-Step Detection & Scheduler Progression Verification (Phase 9.5.3.3).
Tests:
1. Finite gradients: scale_after >= scale_before -> optimizer_stepped=True, scheduler advances.
2. Inf/NaN gradients: scale_after < scale_before -> optimizer_stepped=False, scheduler does NOT advance.
3. Skipped-step counter correctly records skipped updates.
4. Strictly isolated: Uses a 1-parameter dummy model on CPU/CUDA, zero checkpoints touched.
"""

import sys
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

def run_amp_verification():
    print("=" * 80)
    print("PHASE 9.5.3.3 — AMP OPTIMIZER-STEP DETECTION & SCHEDULER PROGRESSION TEST")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dummy 1-parameter linear layer
    layer = nn.Linear(1, 1, bias=False).to(device)
    layer.weight.data.fill_(1.0)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4)

    # Simple step scheduler
    def lr_lambda(step: int) -> float:
        return 1.0 + step * 0.1

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Initial GradScaler with known initial scale
    scaler = torch.amp.GradScaler(device.type, init_scale=1024.0)

    step_counter = 0
    successful_steps = 0
    skipped_steps = 0

    print(f"Initial LR: {scheduler.get_last_lr()[0]:.6f}, Initial Scale: {scaler.get_scale()}")

    # Scenario 1: Step with normal, finite gradients
    print("\n--- Test Case 1: Normal Finite Gradient ---")
    optimizer.zero_grad()
    x = torch.tensor([[2.0]], device=device)
    loss = (layer(x) - 0.0).pow(2).sum()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    scale_after = scaler.get_scale()

    optimizer_stepped = (scale_after >= scale_before)
    if optimizer_stepped:
        scheduler.step()
        successful_steps += 1
    else:
        skipped_steps += 1

    print(f"Scale Before: {scale_before}, Scale After: {scale_after}")
    print(f"Optimizer Stepped: {optimizer_stepped}")
    print(f"Successful Steps: {successful_steps}, Skipped Steps: {skipped_steps}")
    print(f"New LR: {scheduler.get_last_lr()[0]:.6f}")

    assert optimizer_stepped is True, "Case 1 failed: finite gradient did not step"
    assert successful_steps == 1, "Case 1 failed: successful_steps should be 1"
    assert skipped_steps == 0, "Case 1 failed: skipped_steps should be 0"

    # Scenario 2: Step with injected NaN gradient (simulating FP16 numerical overflow)
    print("\n--- Test Case 2: Injected NaN Gradient (Simulating FP16 Overflow) ---")
    optimizer.zero_grad()
    loss = (layer(x) - 0.0).pow(2).sum()
    scaler.scale(loss).backward()
    # Inject NaN into gradient
    layer.weight.grad.fill_(float("nan"))

    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    scale_after = scaler.get_scale()

    optimizer_stepped = (scale_after >= scale_before)
    if optimizer_stepped:
        scheduler.step()
        successful_steps += 1
    else:
        skipped_steps += 1

    print(f"Scale Before: {scale_before}, Scale After: {scale_after}")
    print(f"Optimizer Stepped: {optimizer_stepped}")
    print(f"Successful Steps: {successful_steps}, Skipped Steps: {skipped_steps}")
    print(f"LR after skipped step: {scheduler.get_last_lr()[0]:.6f}")

    assert optimizer_stepped is False, "Case 2 failed: NaN gradient should trigger skipped step"
    assert scale_after < scale_before, "Case 2 failed: scale did not decrease on skipped step"
    assert successful_steps == 1, "Case 2 failed: successful_steps should NOT increment"
    assert skipped_steps == 1, "Case 2 failed: skipped_steps should be 1"
    # LR must NOT advance when step is skipped!
    import math
    assert math.isclose(scheduler.get_last_lr()[0], 1.1e-4, rel_tol=1e-5), "Case 2 failed: scheduler advanced despite skipped step!"

    # Scenario 3: Recover with normal finite gradient on next step
    print("\n--- Test Case 3: Recovery with Finite Gradient ---")
    optimizer.zero_grad()
    loss = (layer(x) - 0.0).pow(2).sum()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    scale_before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    scale_after = scaler.get_scale()

    optimizer_stepped = (scale_after >= scale_before)
    if optimizer_stepped:
        scheduler.step()
        successful_steps += 1
    else:
        skipped_steps += 1

    print(f"Scale Before: {scale_before}, Scale After: {scale_after}")
    print(f"Optimizer Stepped: {optimizer_stepped}")
    print(f"Successful Steps: {successful_steps}, Skipped Steps: {skipped_steps}")
    print(f"New LR after recovery: {scheduler.get_last_lr()[0]:.6f}")

    assert optimizer_stepped is True, "Case 3 failed: did not recover after backoff"
    assert successful_steps == 2, "Case 3 failed: successful_steps should be 2"
    assert skipped_steps == 1, "Case 3 failed: skipped_steps should remain 1"
    assert scheduler.get_last_lr()[0] == 1.2e-4, "Case 3 failed: scheduler did not advance cleanly"

    print("\nVERIFICATION CONFIRMED: AMP optimizer-step detection, backoff, and scheduler gating are 100% verified.")
    print("=" * 80)

if __name__ == "__main__":
    run_amp_verification()
