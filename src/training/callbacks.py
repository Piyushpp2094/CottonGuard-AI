"""
Training callbacks: EarlyStopping and ModelCheckpoint.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 1e-4,
        mode: str = "max",
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.verbose = verbose
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

        if self.mode not in ["min", "max"]:
            raise ValueError("EarlyStopping mode must be 'min' or 'max'.")

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "max":
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class ModelCheckpoint:
    """
    Saves model checkpoints when validation metric improves, plus optionally a last_model checkpoint.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        filename: str = "best_model.pt",
        mode: str = "max",
        save_last: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.checkpoint_dir / filename
        self.last_path = self.checkpoint_dir / "last_model.pt"
        self.mode = mode.lower()
        self.save_last = save_last
        self.best_score: Optional[float] = None

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, Any],
        scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config or {},
        }
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        if self.save_last:
            torch.save(state, self.last_path)

        if is_best:
            torch.save(state, self.best_path)
            print(f"--> Saved new best model checkpoint to {self.best_path}")

    def step(
        self,
        current_score: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, Any],
        scheduler: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            self.save(model, optimizer, epoch, metrics, scheduler, config, is_best=True)
            return True

        if self.mode == "max":
            improved = current_score > self.best_score
        else:
            improved = current_score < self.best_score

        if improved:
            self.best_score = current_score
            self.save(model, optimizer, epoch, metrics, scheduler, config, is_best=True)
            return True
        else:
            if self.save_last:
                self.save(model, optimizer, epoch, metrics, scheduler, config, is_best=False)
            return False
