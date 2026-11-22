"""
Exact Learning-Rate Progression Verification (Phase 9.5.3.3).
Tracks:
1. Learning rate APPLIED during each optimizer update.
2. Learning rate immediately AFTER scheduler.step() is called.
3. Specific checkpoints: Update 1, Update 35, Update 105, Update 106, Update 1050.
"""

import sys
import math
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_faster_rcnn import build_warmup_cosine_scheduler

def verify_lr_schedule():
    print("=" * 80)
    print("PHASE 9.5.3.3 — EXACT LEARNING-RATE PROGRESSION AUDIT")
    print("=" * 80)

    # Configuration values
    base_lr = 0.0001
    min_lr = 0.000001
    warmup_factor = 0.333333
    warmup_steps = 105
    total_steps = 1050

    # Dummy model and optimizer
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

    scheduler = build_warmup_cosine_scheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        warmup_factor=warmup_factor,
        base_lr=base_lr,
        min_lr=min_lr,
    )

    tracked_updates = [1, 35, 105, 106, 1050]
    history = {}

    for update_idx in range(1, total_steps + 1):
        # 1. LR applied to this optimizer update
        applied_lr = optimizer.param_groups[0]["lr"]

        # 2. Advance scheduler (as happens after successful optimizer step)
        scheduler.step()

        # 3. LR state immediately after update
        post_step_lr = optimizer.param_groups[0]["lr"]

        if update_idx in tracked_updates:
            history[update_idx] = {
                "applied_lr": applied_lr,
                "post_step_lr": post_step_lr,
            }

    print(f"{'Update #':<12} | {'Epoch':<8} | {'Phase':<20} | {'LR Applied to Update':<24} | {'LR Immediately After':<24}")
    print("-" * 96)

    epoch_map = {
        1: (1, "Warmup Start"),
        35: (1, "End of Epoch 1"),
        105: (3, "End of Warmup (Ep 3)"),
        106: (4, "Cosine Decay Start"),
        1050: (30, "Final Step (Ep 30)"),
    }

    for u in tracked_updates:
        ep, phase = epoch_map[u]
        app = history[u]["applied_lr"]
        post = history[u]["post_step_lr"]
        print(f"Update {u:<5} | Ep {ep:<5} | {phase:<20} | {app:.8f} ({app:.2e}) | {post:.8f} ({post:.2e})")

    # Verification assertions:
    # Update 1: applied LR should be base_lr * warmup_factor ~ 3.33333e-5
    assert math.isclose(history[1]["applied_lr"], 3.33333e-5, rel_tol=1e-4), "Update 1 applied LR mismatch"

    # Update 105: applied LR should be near base_lr ~ 9.9365e-5, post-step LR should be exactly base_lr = 1.0e-4
    assert math.isclose(history[105]["post_step_lr"], 1.0e-4, rel_tol=1e-5), "Update 105 post-step LR should reach base_lr"

    # Update 106: applied LR should be base_lr = 1.0e-4 (beginning cosine decay)
    assert math.isclose(history[106]["applied_lr"], 1.0e-4, rel_tol=1e-5), "Update 106 applied LR should be base_lr"

    # Update 1050: post-step LR should reach min_lr = 1.0e-6
    assert math.isclose(history[1050]["post_step_lr"], 1.0e-6, rel_tol=1e-5), "Update 1050 post-step LR should reach min_lr"

    print("\nVERIFICATION CONFIRMED: Learning-rate progression matches theoretical specification exactly.")
    print("=" * 80)

if __name__ == "__main__":
    verify_lr_schedule()
