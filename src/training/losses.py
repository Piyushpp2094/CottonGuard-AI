"""
Loss functions for Cotton Leaf AI model training.
"""

from typing import Optional
import torch
import torch.nn as nn


def get_classification_loss(
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.05
) -> nn.CrossEntropyLoss:
    """
    Returns CrossEntropyLoss configured with class weights and label smoothing.

    Args:
        class_weights: Tensor of shape (num_classes,) to counteract class imbalance.
        label_smoothing: Regularization parameter (e.g. 0.05) to prevent overconfident predictions.
    """
    return nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=label_smoothing
    )
