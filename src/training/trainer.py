"""
Training and validation engine for Cotton Leaf AI classifiers.
Includes Mixed Precision (AMP), gradient clipping, and metric tracking.
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.training.callbacks import EarlyStopping, ModelCheckpoint


class ClassificationTrainer:
    """
    Manages the training, validation, metric evaluation, and checkpointing
    for cotton leaf disease classification models.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[Any] = None,
        checkpoint_callback: Optional[ModelCheckpoint] = None,
        early_stopping: Optional[EarlyStopping] = None,
        use_amp: bool = True,
        max_grad_norm: float = 1.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_callback = checkpoint_callback
        self.early_stopping = early_stopping
        self.use_amp = use_amp and device.type == "cuda"
        self.max_grad_norm = max_grad_norm
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_macro_f1": [],
            "val_weighted_f1": [],
            "learning_rate": [],
        }

    def train_epoch(self) -> Tuple[float, float]:
        """Runs one full training epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []

        for images, targets in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            self.scaler.scale(loss).backward()

            if self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = accuracy_score(all_targets, all_preds)
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def evaluate(self, data_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Runs validation / evaluation loop and returns comprehensive metrics."""
        loader = data_loader if data_loader is not None else self.val_loader
        self.model.eval()

        running_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

        val_loss = running_loss / len(loader.dataset)
        val_acc = accuracy_score(all_targets, all_preds)
        val_macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        val_weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
        val_precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
        val_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)

        return {
            "loss": float(val_loss),
            "accuracy": float(val_acc),
            "macro_f1": float(val_macro_f1),
            "weighted_f1": float(val_weighted_f1),
            "precision": float(val_precision),
            "recall": float(val_recall),
        }

    def fit(self, num_epochs: int, config: Optional[Dict[str, Any]] = None) -> Dict[str, List[float]]:
        """Executes the full training loop across the specified number of epochs."""
        print(f"\n[TRAINER] Starting training on device: {self.device} (AMP enabled: {self.use_amp})")
        print(f"[TRAINER] Total Epochs: {num_epochs} | Training samples: {len(self.train_loader.dataset)} | Val samples: {len(self.val_loader.dataset)}")
        print("-" * 90)

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["macro_f1"])
                else:
                    self.scheduler.step()

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_macro_f1"].append(val_metrics["macro_f1"])
            self.history["val_weighted_f1"].append(val_metrics["weighted_f1"])
            self.history["learning_rate"].append(current_lr)

            elapsed = time.time() - epoch_start
            print(
                f"Epoch {epoch:02d}/{num_epochs:02d} [{elapsed:.1f}s] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']*100:.1f}% "
                f"Macro-F1: {val_metrics['macro_f1']:.4f} | LR: {current_lr:.2e}"
            )

            # Checkpoint step
            if self.checkpoint_callback is not None:
                self.checkpoint_callback.step(
                    current_score=val_metrics["macro_f1"],
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    scheduler=self.scheduler,
                    config=config
                )

            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping(val_metrics["macro_f1"]):
                    print(f"\n[TRAINER] Early stopping triggered at epoch {epoch} (No improvement in {self.early_stopping.patience} epochs).")
                    break

            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        total_time = time.time() - start_time
        print("-" * 90)
        print(f"[TRAINER] Training completed in {total_time/60:.2f} minutes.")
        return self.history
