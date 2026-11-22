"""
Gradient-weighted Class Activation Mapping (Grad-CAM) for Cotton Leaf AI.

Implements standard Grad-CAM (Selvaraju et al., 2017) to provide visual explanations
for decisions made by the EfficientNet-B0 foliar condition classifier.

Mathematical formulation:
1. Target layer feature activation maps: A^k in R^(H x W) for channel k in {1, ..., C}.
2. Spatially pooled gradients of score Y^c with respect to A^k:
       alpha_k^c = (1 / Z) * sum_i sum_j (d Y^c / d A_{i,j}^k)
3. Rectified linear combination:
       L_Grad-CAM^c = ReLU( sum_k alpha_k^c * A^k )
4. Safe min-max normalization to [0.0, 1.0] and bilinear upsampling to input resolution.

Scientific Notice:
Grad-CAM heatmaps highlight salient receptive field regions that positively influence
the classifier's logit score for a selected class. Grad-CAM is an interpretability aid
and DOES NOT represent pixel-accurate lesion boundaries or physical disease severity.
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import CLASS_NAMES, CLASS_TO_IDX
from src.data.preprocessing import get_inference_transforms


class GradCAM:
    """
    Grad-CAM engine for convolutional neural network classifiers.
    
    Supports:
    - EfficientNet-B0 backbone (default: features[-1] / Conv2dNormActivation)
    - Forward and backward hook registration with guaranteed cleanup
    - Prediction-guided or user-specified target class diagnostic analysis
    - CUDA acceleration with CPU fallback
    - Zero parameter modification (weights are strictly immutable)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initializes the Grad-CAM explainer.

        Args:
            model: PyTorch classifier (CottonLeafClassifier).
            target_layer: Convolutional feature module to hook. If None, uses
                          model.get_last_conv_layer() or model.backbone.features[-1].
            device: Execution device. If None, infers from model parameters.
        """
        self.model = model
        
        # Determine device
        if device is not None:
            self.device = device
        else:
            try:
                self.device = next(model.parameters()).device
            except (StopIteration, AttributeError):
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Resolve target layer
        if target_layer is not None:
            self.target_layer = target_layer
        elif hasattr(model, "get_last_conv_layer") and callable(model.get_last_conv_layer):
            self.target_layer = model.get_last_conv_layer()
        elif hasattr(model, "backbone") and hasattr(model.backbone, "features"):
            self.target_layer = model.backbone.features[-1]
        else:
            raise ValueError(
                "Could not automatically locate the final convolutional feature layer. "
                "Please explicitly supply 'target_layer'."
            )

        self.transform = get_inference_transforms(image_size=224)
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

    def _forward_hook(self, module: nn.Module, input_tensor: Any, output_tensor: torch.Tensor) -> None:
        """Captures forward activation maps from the target convolutional layer."""
        self.activations = output_tensor.detach().clone()

    def _backward_hook(self, module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]) -> None:
        """Captures backward gradient tensors with respect to target layer activations."""
        if grad_output and grad_output[0] is not None:
            self.gradients = grad_output[0].detach().clone()

    def generate_cam(
        self,
        image_input: Union[torch.Tensor, Image.Image, np.ndarray],
        target_class: Optional[Union[int, str]] = None,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Dict[str, Any]:
        """
        Computes Grad-CAM heatmap for a leaf image.

        Args:
            image_input: Input photograph as a PIL Image, preprocessed (1, 3, H, W) Tensor,
                         or RGB NumPy array.
            target_class: Class index (0-6) or class name to explain. If None,
                          explains the predicted class (argmax logit).
            target_size: (width, height) to resize the resulting heatmap. Defaults to
                         the original PIL image size or input spatial resolution.

        Returns:
            Dictionary containing:
            - "heatmap": 2D float32 NumPy array in [0.0, 1.0] of shape (H, W).
            - "predicted_class_idx": int
            - "predicted_class_name": str
            - "confidence": float in [0.0, 1.0]
            - "target_class_idx": int
            - "target_class_name": str
            - "all_probabilities": Dict[str, float]
            - "device": str
            - "target_layer_name": str
        """
        orig_width, orig_height = 224, 224

        # 1. Standardize input to PyTorch tensor
        if isinstance(image_input, Image.Image):
            rgb_image = image_input.convert("RGB")
            orig_width, orig_height = rgb_image.size
            tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        elif isinstance(image_input, np.ndarray):
            rgb_image = Image.fromarray(image_input).convert("RGB")
            orig_width, orig_height = rgb_image.size
            tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)
        elif isinstance(image_input, torch.Tensor):
            if image_input.dim() == 3:
                tensor = image_input.unsqueeze(0).to(self.device)
            elif image_input.dim() == 4:
                tensor = image_input.to(self.device)
            else:
                raise ValueError(f"Expected 3D or 4D tensor, got shape {image_input.shape}")
            orig_height, orig_width = tensor.shape[2], tensor.shape[3]
        else:
            raise TypeError(f"Unsupported input type: {type(image_input)}")

        if target_size is not None:
            out_width, out_height = target_size
        else:
            out_width, out_height = orig_width, orig_height

        # 2. Preserve model state and register hooks
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        hook_fwd = None
        hook_bwd = None

        try:
            hook_fwd = self.target_layer.register_forward_hook(self._forward_hook)
            hook_bwd = self.target_layer.register_full_backward_hook(self._backward_hook)

            # 3. Forward pass
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=-1)

            pred_idx = int(torch.argmax(logits, dim=-1).item())
            confidence = float(probs[0, pred_idx].item())

            # 4. Resolve target class
            if target_class is None:
                tgt_idx = pred_idx
            elif isinstance(target_class, str):
                if target_class not in CLASS_TO_IDX:
                    raise ValueError(f"Unknown target class name '{target_class}'. Valid: {list(CLASS_TO_IDX.keys())}")
                tgt_idx = CLASS_TO_IDX[target_class]
            elif isinstance(target_class, int):
                if not (0 <= target_class < len(CLASS_NAMES)):
                    raise ValueError(f"Target class index {target_class} out of bounds [0, {len(CLASS_NAMES) - 1}]")
                tgt_idx = target_class
            else:
                raise TypeError(f"Invalid target_class type: {type(target_class)}")

            # 5. Backward pass for target score
            score = logits[0, tgt_idx]
            score.backward(retain_graph=False)

            if self.activations is None or self.gradients is None:
                raise RuntimeError("Failed to capture feature activations or gradients from target layer.")

            # 6. Global Average Pooling over spatial dimensions (H, W)
            # gradients: (1, C, H, W) -> weights: (1, C, 1, 1)
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)

            # 7. Weighted combination of activation maps
            # (1, C, 1, 1) * (1, C, H, W) summed across channels -> (1, H, W)
            cam = torch.sum(weights * self.activations, dim=1)

            # 8. ReLU to discard negative contributions
            cam = F.relu(cam)

            # 9. Safe min-max normalization
            cam_min = cam.min()
            cam_max = cam.max()
            if (cam_max - cam_min) > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)

            # 10. Bilinear interpolation to target dimensions
            cam_upsampled = F.interpolate(
                cam.unsqueeze(1),
                size=(out_height, out_width),
                mode="bilinear",
                align_corners=False
            ).squeeze(0).squeeze(0)

            heatmap_np = cam_upsampled.detach().cpu().numpy().astype(np.float32)

            all_probs = {
                CLASS_NAMES[i]: float(probs[0, i].item())
                for i in range(len(CLASS_NAMES))
            }

            return {
                "heatmap": heatmap_np,
                "predicted_class_idx": pred_idx,
                "predicted_class_name": CLASS_NAMES[pred_idx],
                "confidence": confidence,
                "target_class_idx": tgt_idx,
                "target_class_name": CLASS_NAMES[tgt_idx],
                "all_probabilities": all_probs,
                "device": str(self.device),
                "target_layer_name": self.target_layer.__class__.__name__,
                "input_resolution": (orig_width, orig_height),
                "heatmap_resolution": (out_width, out_height),
            }

        finally:
            # Guaranteed cleanup
            if hook_fwd is not None:
                hook_fwd.remove()
            if hook_bwd is not None:
                hook_bwd.remove()
            self.model.zero_grad(set_to_none=True)
            self.activations = None
            self.gradients = None
            self.model.train(was_training)
