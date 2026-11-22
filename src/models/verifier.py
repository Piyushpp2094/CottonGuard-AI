"""
Cotton Leaf Verification Model Architecture & Decision Gate.

Distinguishes:
0: cotton_leaf (Positive: genuine cotton foliar pathology or healthy lamina)
1: non_cotton_leaf (Negative: other crop and plant foliage like tomato, potato, apple, corn, etc.)
2: non_leaf (Negative: human photos, documents, vehicles, animals, indoor objects)

Enforces strict False Acceptance Rate (FAR) minimization before disease analysis.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

VERIFIER_CLASSES = ["cotton_leaf", "non_cotton_leaf", "non_leaf"]


@dataclass
class VerificationDecision:
    is_cotton_leaf: Optional[bool]  # True = Cotton Leaf, False = Rejected, None = Uncertain
    status: str  # "COTTON_LEAF_VERIFIED" | "REJECTED_NON_COTTON" | "REJECTED_NON_PLANT" | "UNCERTAIN_IMAGE"
    confidence: float
    predicted_class: str
    probabilities: Dict[str, float]
    rejection_reason: Optional[str] = None
    user_message: Optional[str] = None
    threshold: float = 0.70

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CottonLeafVerifier(nn.Module):
    """
    EfficientNet-B0 backbone fine-tuned for tri-class foliar domain verification.
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.30):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_cotton_verifier(
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> CottonLeafVerifier:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CottonLeafVerifier(num_classes=len(VERIFIER_CLASSES))
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if p.is_file():
            try:
                ckpt = torch.load(p, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(p, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            print(f"[VERIFIER] Loaded checkpoint from {p} on {device}")
        else:
            print(f"[VERIFIER] Warning: Checkpoint not found at {p}, initialized with ImageNet weights.")

    model.to(device)
    model.eval()
    return model


class VerificationService:
    """
    Inference service for cotton leaf verification.
    Applies strict calibrated thresholds to prevent false acceptance of invalid images.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        verification_threshold: float = 0.70,
        uncertainty_lower_bound: float = 0.45,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verification_threshold = verification_threshold
        self.uncertainty_lower_bound = uncertainty_lower_bound
        self.model = build_cotton_verifier(checkpoint_path=checkpoint_path, device=self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def verify(
        self,
        image: Union[Image.Image, np.ndarray],
        threshold: Optional[float] = None,
    ) -> VerificationDecision:
        th = threshold if threshold is not None else self.verification_threshold

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                pil_img = Image.fromarray(image).convert("RGB")
            elif image.shape[2] == 4:
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGBA2RGB))
            else:
                pil_img = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        p_cotton = float(probs[0])
        p_non_cotton = float(probs[1])
        p_non_leaf = float(probs[2])

        prob_dict = {
            "cotton_leaf": round(p_cotton, 4),
            "non_cotton_leaf": round(p_non_cotton, 4),
            "non_leaf": round(p_non_leaf, 4),
        }

        pred_idx = int(np.argmax(probs))
        pred_class = VERIFIER_CLASSES[pred_idx]

        # Case 1: Predicted as non-leaf (human photo, document, vehicle, animal, object)
        if pred_class == "non_leaf" or p_non_leaf >= 0.40:
            return VerificationDecision(
                is_cotton_leaf=False,
                status="REJECTED_NON_PLANT",
                confidence=round(p_non_leaf, 4),
                predicted_class="non_leaf",
                probabilities=prob_dict,
                rejection_reason="The uploaded image does not contain plant foliage. Non-plant content (such as a document, human photograph, or object) was detected.",
                user_message="The uploaded image is not a cotton leaf. Please upload a clear photograph of a cotton leaf to perform disease analysis.",
                threshold=th,
            )

        # Case 2: Predicted as non-cotton leaf (e.g. tomato, potato, corn, apple leaf)
        if pred_class == "non_cotton_leaf" or p_non_cotton >= 0.50:
            return VerificationDecision(
                is_cotton_leaf=False,
                status="REJECTED_NON_COTTON",
                confidence=round(p_non_cotton, 4),
                predicted_class="non_cotton_leaf",
                probabilities=prob_dict,
                rejection_reason="The uploaded image contains plant foliage, but it does not appear to be a cotton leaf.",
                user_message="The uploaded image is not a cotton leaf. Cotton disease analysis cannot be performed on other plant species.",
                threshold=th,
            )

        # Case 3: Predicted as cotton leaf
        if pred_class == "cotton_leaf":
            if p_cotton >= th:
                return VerificationDecision(
                    is_cotton_leaf=True,
                    status="COTTON_LEAF_VERIFIED",
                    confidence=round(p_cotton, 4),
                    predicted_class="cotton_leaf",
                    probabilities=prob_dict,
                    rejection_reason=None,
                    user_message="Valid cotton leaf verified.",
                    threshold=th,
                )
            else:
                # Uncertain / Ambiguous
                return VerificationDecision(
                    is_cotton_leaf=None,
                    status="UNCERTAIN_IMAGE",
                    confidence=round(p_cotton, 4),
                    predicted_class="cotton_leaf",
                    probabilities=prob_dict,
                    rejection_reason=f"Cotton leaf verification confidence ({p_cotton*100:.1f}%) is below the required acceptance threshold ({th*100:.1f}%).",
                    user_message="Cotton leaf verification is uncertain. Please upload a clearer photograph with the leaf centered and unobstructed.",
                    threshold=th,
                )

        # Fallback uncertain
        return VerificationDecision(
            is_cotton_leaf=None,
            status="UNCERTAIN_IMAGE",
            confidence=round(float(np.max(probs)), 4),
            predicted_class="uncertain",
            probabilities=prob_dict,
            rejection_reason="Verification confidence is insufficient.",
            user_message="The system could not confidently verify that this image contains a cotton leaf. Please upload a clearer image.",
            threshold=th,
        )
