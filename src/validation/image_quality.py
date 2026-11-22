"""
Image Quality Validation Module for CottonGuard AI.

Evaluates uploaded foliar imagery BEFORE verification or disease classification.
Rejects:
- Blurry images (Laplacian variance < threshold)
- Severely underexposed / dark images (mean pixel intensity < threshold)
- Severely overexposed / washed out images (mean pixel intensity > threshold)
- Low-contrast / uniform blank images (pixel intensity std dev < threshold)
- Sub-resolution images (min dimension < 120 px)
- Malformed, empty, or unreadable image tensors
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image


@dataclass
class ImageQualityResult:
    is_valid: bool
    status: str  # "PASS" | "REJECTED_QUALITY"
    error_code: Optional[str] = None
    user_message: Optional[str] = None
    metrics: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["metrics"] is None:
            d["metrics"] = {}
        return d


class ImageQualityValidator:
    """
    High-speed deterministic image quality assessment designed for agricultural field photos.
    Ensures valid field photos with natural lighting are accepted while unusable inputs are rejected.
    """

    def __init__(
        self,
        min_dimension: int = 120,
        max_dimension: int = 8192,
        min_blur_score: float = 40.0,
        min_brightness: float = 24.0,
        max_brightness: float = 240.0,
        min_contrast: float = 14.0,
    ):
        self.min_dimension = min_dimension
        self.max_dimension = max_dimension
        self.min_blur_score = min_blur_score
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast

    def validate(self, image: Union[Image.Image, np.ndarray]) -> ImageQualityResult:
        # Convert to numpy uint8 RGB
        if isinstance(image, Image.Image):
            rgb = np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            rgb = image
            if rgb.ndim == 2:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
            elif rgb.shape[2] == 4:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
        else:
            return ImageQualityResult(
                is_valid=False,
                status="REJECTED_QUALITY",
                error_code="INVALID_IMAGE_TYPE",
                user_message="Image quality is insufficient for reliable analysis. Please upload a clearer photograph of a cotton leaf.",
                metrics={},
            )

        h, w = rgb.shape[:2]

        # 1. Dimension checks
        if h < self.min_dimension or w < self.min_dimension:
            return ImageQualityResult(
                is_valid=False,
                status="REJECTED_QUALITY",
                error_code="IMAGE_TOO_SMALL",
                user_message="Image resolution is too low for reliable foliar diagnosis. Minimum resolution is 120x120 pixels.",
                metrics={"width": w, "height": h, "min_required": self.min_dimension},
            )

        if h > self.max_dimension or w > self.max_dimension:
            return ImageQualityResult(
                is_valid=False,
                status="REJECTED_QUALITY",
                error_code="IMAGE_TOO_LARGE",
                user_message="Image dimensions exceed maximum supported resolution (8192x8192).",
                metrics={"width": w, "height": h},
            )

        # 2. Convert to Grayscale for photometric analysis
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))

        # 3. Exposure checks
        if mean_val < self.min_brightness:
            return ImageQualityResult(
                is_valid=False,
                status="REJECTED_QUALITY",
                error_code="IMAGE_UNDEREXPOSED",
                user_message="Image is too dark for reliable analysis. Please photograph the leaf under adequate ambient lighting.",
                metrics={"brightness": round(mean_val, 2), "min_required": self.min_brightness},
            )

        if mean_val > self.max_brightness:
            return ImageQualityResult(
                is_valid=False,
                status="REJECTED_QUALITY",
                error_code="IMAGE_OVEREXPOSED",
                user_message="Image is overexposed or washed out. Please avoid extreme flash glare or direct sun reflections.",
                metrics={"brightness": round(mean_val, 2), "max_allowed": self.max_brightness},
            )

        # 4. Contrast check (flat / uniform image)
        if std_val < self.min_contrast:
            return ImageQualityResult(
                is_valid=False,
                status="REJECTED_QUALITY",
                error_code="IMAGE_LOW_CONTRAST",
                user_message="Image has insufficient visual contrast or contains a blank surface. Please upload a clear photo of a cotton leaf.",
                metrics={"contrast": round(std_val, 2), "min_required": self.min_contrast},
            )

        # 5. Blur / Sharpness check via Laplacian Variance
        # Downscale for consistent Laplacian variance calculation regardless of sensor megapixel count
        scale_size = (512, 512)
        gray_resized = cv2.resize(gray, scale_size, interpolation=cv2.INTER_AREA)
        laplacian = cv2.Laplacian(gray_resized, cv2.CV_64F)
        blur_score = float(np.var(laplacian))

        if blur_score < self.min_blur_score:
            return ImageQualityResult(
                is_valid=False,
                status="REJECTED_QUALITY",
                error_code="IMAGE_BLURRY",
                user_message="Image is too blurry or out of focus for reliable foliar diagnosis. Please capture a sharp, stabilized photo.",
                metrics={"blur_score": round(blur_score, 2), "min_required": self.min_blur_score},
            )

        # Image quality is validated
        return ImageQualityResult(
            is_valid=True,
            status="PASS",
            error_code=None,
            user_message=None,
            metrics={
                "width": w,
                "height": h,
                "brightness": round(mean_val, 2),
                "contrast": round(std_val, 2),
                "blur_score": round(blur_score, 2),
            },
        )


image_quality_validator = ImageQualityValidator()
