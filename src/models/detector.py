"""
Permissive Pure-PyTorch Anchor-Free YOLO Object Detector for Cotton Leaf AI.

License Notice:
  Implemented strictly using PyTorch (BSD-3), TorchVision (BSD-3), OpenCV (Apache-2.0),
  and Pillow (HPND). Deliberately avoids AGPL-3.0 / GPL-3.0 YOLO packages so the
  platform remains free of copyleft hosting restrictions.

Architecture:
  - Feature Backbone: Multi-scale feature maps P3 (stride 8), P4 (stride 16), P5 (stride 32)
    built with lightweight depthwise/CSP convolutional blocks + SPPF.
  - Neck: Top-down FPN + Bottom-up PANet path aggregation.
  - Head: Decoupled anchor-free YOLO detection heads predicting:
      * 4 distances (left, top, right, bottom) from cell center (FCOS/YOLOX/YOLOv8 style)
      * 1 objectness logit
      * C disease-region class logits (default C=6 symptomatic classes)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DETECTION_CONFIG_PATH = PROJECT_ROOT / "configs" / "detection_classes.json"

DEFAULT_DETECTION_CLASSES: List[str] = [
    "Bacterial Blight",
    "Curl Virus",
    "Herbicide Growth Damage",
    "Leaf Hopper Jassids",
    "Leaf Redding",
    "Leaf Variegation",
]

CLASS_COLORS_RGB: List[Tuple[int, int, int]] = [
    (239, 68, 68),   # Bacterial Blight - Red
    (249, 115, 22),  # Curl Virus - Orange
    (234, 179, 8),   # Herbicide Growth Damage - Yellow
    (139, 92, 246),  # Leaf Hopper Jassids - Violet
    (236, 72, 153),  # Leaf Redding - Pink
    (6, 182, 212),   # Leaf Variegation - Cyan
]


def load_detection_classes() -> List[str]:
    """Loads ordered YOLO detection class names from configs/detection_classes.json."""
    if DETECTION_CONFIG_PATH.is_file():
        with open(DETECTION_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        classes = sorted(data.get("classes", []), key=lambda c: c["id"])
        if classes:
            return [c["name"] for c in classes]
    return list(DEFAULT_DETECTION_CLASSES)


class ConvBNSiLU(nn.Module):
    """Standard YOLO Conv2d + BatchNorm2d + SiLU block."""

    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Bottleneck(nn.Module):
    """CSP bottleneck block with optional residual skip connection."""

    def __init__(self, c1: int, c2: int, shortcut: bool = True, expansion: float = 0.5):
        super().__init__()
        hidden = max(8, int(c2 * expansion))
        self.cv1 = ConvBNSiLU(c1, hidden, k=1, s=1)
        self.cv2 = ConvBNSiLU(hidden, c2, k=3, s=1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2fBlock(nn.Module):
    """Lightweight Cross-Stage Partial (C2f-style) feature block."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        self.c_mid = max(8, c2 // 2)
        self.cv1 = ConvBNSiLU(c1, 2 * self.c_mid, k=1, s=1)
        self.blocks = nn.ModuleList(
            [Bottleneck(self.c_mid, self.c_mid, shortcut=shortcut, expansion=1.0) for _ in range(n)]
        )
        self.cv2 = ConvBNSiLU((2 + n) * self.c_mid, c2, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        for m in self.blocks:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (multi-scale receptive field aggregation)."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_mid = max(8, c1 // 2)
        self.cv1 = ConvBNSiLU(c1, c_mid, k=1, s=1)
        self.cv2 = ConvBNSiLU(c_mid * 4, c2, k=1, s=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class DecoupledYOLOHead(nn.Module):
    """Anchor-free decoupled head outputting box distances, objectness, and class logits."""

    def __init__(self, in_c: int, num_classes: int):
        super().__init__()
        self.stem = ConvBNSiLU(in_c, in_c, k=1, s=1)
        self.reg_conv = nn.Sequential(ConvBNSiLU(in_c, in_c, k=3), ConvBNSiLU(in_c, in_c, k=3))
        self.cls_conv = nn.Sequential(ConvBNSiLU(in_c, in_c, k=3), ConvBNSiLU(in_c, in_c, k=3))

        self.box_out = nn.Conv2d(in_c, 4, kernel_size=1)
        self.obj_out = nn.Conv2d(in_c, 1, kernel_size=1)
        self.cls_out = nn.Conv2d(in_c, num_classes, kernel_size=1)

        # Standard objectness/class focal bias init (-4.59 corresponds to prior prob ~0.01)
        nn.init.constant_(self.obj_out.bias, -4.5)
        nn.init.constant_(self.cls_out.bias, -4.5)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.stem(x)
        reg_feat = self.reg_conv(feat)
        cls_feat = self.cls_conv(feat)
        ltrb = F.softplus(self.box_out(reg_feat))
        obj_logit = self.obj_out(reg_feat)
        cls_logits = self.cls_out(cls_feat)
        return ltrb, obj_logit, cls_logits


class CottonLeafYOLODetector(nn.Module):
    """
    Lightweight Multi-Scale Anchor-Free YOLO Object Detector (nano/small scale).
    Engineered for 640x640 (or 512x512) inputs within 6 GB RTX 3050 VRAM.
    """

    STRIDES: Tuple[int, int, int] = (8, 16, 32)

    def __init__(
        self,
        num_classes: int = 6,
        width_mult: float = 0.5,
        input_size: int = 640,
        class_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.width_mult = width_mult
        self.input_size = input_size
        self.class_names = class_names or load_detection_classes()

        def ch(base: int) -> int:
            return max(16, int(round(base * width_mult / 8.0) * 8))

        c1, c2, c3, c4, c5 = ch(64), ch(128), ch(256), ch(512), ch(768)

        # CSP-style Backbone
        self.stem = ConvBNSiLU(3, c1, k=3, s=2)              # /2
        self.stage1 = nn.Sequential(ConvBNSiLU(c1, c2, k=3, s=2), C2fBlock(c2, c2, n=1))   # /4
        self.stage2 = nn.Sequential(ConvBNSiLU(c2, c3, k=3, s=2), C2fBlock(c3, c3, n=2))   # /8  -> P3
        self.stage3 = nn.Sequential(ConvBNSiLU(c3, c4, k=3, s=2), C2fBlock(c4, c4, n=2))   # /16 -> P4
        self.stage4 = nn.Sequential(ConvBNSiLU(c4, c5, k=3, s=2), C2fBlock(c5, c5, n=1), SPPF(c5, c5))  # /32 -> P5

        # FPN + PANet Neck
        self.lat_p4 = C2fBlock(c5 + c4, c4, n=1, shortcut=False)
        self.lat_p3 = C2fBlock(c4 + c3, c3, n=1, shortcut=False)
        self.down_p3 = ConvBNSiLU(c3, c3, k=3, s=2)
        self.pan_p4 = C2fBlock(c3 + c4, c4, n=1, shortcut=False)
        self.down_p4 = ConvBNSiLU(c4, c4, k=3, s=2)
        self.pan_p5 = C2fBlock(c4 + c5, c5, n=1, shortcut=False)

        # Decoupled Multi-Scale Detection Heads
        self.head_p3 = DecoupledYOLOHead(c3, num_classes)
        self.head_p4 = DecoupledYOLOHead(c4, num_classes)
        self.head_p5 = DecoupledYOLOHead(c5, num_classes)

    def forward(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
        """
        Forward pass returning per-scale predictions:
            [(ltrb, obj_logit, cls_logits, stride), ...] for strides 8, 16, 32.
        """
        s1 = self.stage1(self.stem(x))
        p3 = self.stage2(s1)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)

        # Top-down FPN
        up_p5 = F.interpolate(p5, size=p4.shape[2:], mode="nearest")
        n4 = self.lat_p4(torch.cat([up_p5, p4], dim=1))

        up_n4 = F.interpolate(n4, size=p3.shape[2:], mode="nearest")
        n3 = self.lat_p3(torch.cat([up_n4, p3], dim=1))

        # Bottom-up PANet
        out_p3 = n3
        out_p4 = self.pan_p4(torch.cat([self.down_p3(out_p3), n4], dim=1))
        out_p5 = self.pan_p5(torch.cat([self.down_p4(out_p4), p5], dim=1))

        return [
            (*self.head_p3(out_p3), self.STRIDES[0]),
            (*self.head_p4(out_p4), self.STRIDES[1]),
            (*self.head_p5(out_p5), self.STRIDES[2]),
        ]

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Runs single-image YOLO inference with NMS and maps coordinates back to original size.
        """
        self.eval()
        device = next(self.parameters()).device
        rgb = image.convert("RGB")
        orig_w, orig_h = rgb.size

        resized = rgb.resize((self.input_size, self.input_size), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

        scale_outputs = self.forward(tensor)
        boxes, scores, class_ids = decode_yolo_predictions(
            scale_outputs=scale_outputs,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
        )

        scale_x = orig_w / float(self.input_size)
        scale_y = orig_h / float(self.input_size)

        results: List[Dict[str, Any]] = []
        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1 = float(np.clip(box[0] * scale_x, 0.0, orig_w))
            y1 = float(np.clip(box[1] * scale_y, 0.0, orig_h))
            x2 = float(np.clip(box[2] * scale_x, 0.0, orig_w))
            y2 = float(np.clip(box[3] * scale_y, 0.0, orig_h))
            if (x2 - x1) < 2.0 or (y2 - y1) < 2.0:
                continue

            cid = int(cls_id)
            cname = self.class_names[cid] if cid < len(self.class_names) else f"Class_{cid}"
            results.append({
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "normalized_bbox": [
                    round(((x1 + x2) / 2.0) / orig_w, 4),
                    round(((y1 + y2) / 2.0) / orig_h, 4),
                    round((x2 - x1) / orig_w, 4),
                    round((y2 - y1) / orig_h, 4),
                ],
                "class_id": cid,
                "class_name": cname,
                "confidence": round(float(score), 4),
            })

        return results


def decode_yolo_predictions(
    scale_outputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decodes anchor-free YOLO predictions from a batch size of 1 into pixel boxes [x1, y1, x2, y2].
    """
    all_boxes = []
    all_scores = []
    all_classes = []

    for ltrb, obj_logit, cls_logits, stride in scale_outputs:
        _, _, grid_h, grid_w = ltrb.shape
        device = ltrb.device

        yv, xv = torch.meshgrid(
            torch.arange(grid_h, device=device, dtype=ltrb.dtype),
            torch.arange(grid_w, device=device, dtype=ltrb.dtype),
            indexing="ij",
        )
        cx = (xv + 0.5) * stride
        cy = (yv + 0.5) * stride

        dist = ltrb[0] * stride
        x1 = (cx - dist[0]).reshape(-1)
        y1 = (cy - dist[1]).reshape(-1)
        x2 = (cx + dist[2]).reshape(-1)
        y2 = (cy + dist[3]).reshape(-1)
        boxes_scale = torch.stack([x1, y1, x2, y2], dim=1)

        obj_prob = torch.sigmoid(obj_logit[0, 0]).reshape(-1, 1)
        cls_prob = torch.sigmoid(cls_logits[0]).permute(1, 2, 0).reshape(-1, cls_logits.shape[1])
        combined = obj_prob * cls_prob

        max_scores, cls_idx = torch.max(combined, dim=1)
        keep = max_scores >= conf_threshold
        if torch.any(keep):
            all_boxes.append(boxes_scale[keep])
            all_scores.append(max_scores[keep])
            all_classes.append(cls_idx[keep])

    if not all_boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    cat_boxes = torch.cat(all_boxes, dim=0)
    cat_scores = torch.cat(all_scores, dim=0)
    cat_classes = torch.cat(all_classes, dim=0)

    # Class-aware NMS offset trick
    max_wh = 4096.0
    offsets = cat_classes.float().unsqueeze(1) * max_wh
    keep_idx = nms(cat_boxes + offsets, cat_scores, iou_threshold)[:max_detections]

    return (
        cat_boxes[keep_idx].detach().cpu().numpy(),
        cat_scores[keep_idx].detach().cpu().numpy(),
        cat_classes[keep_idx].detach().cpu().numpy(),
    )


def draw_detection_boxes(image: Image.Image, detections: List[Dict[str, Any]]) -> Image.Image:
    """Renders bounding boxes and confidence labels on a copy of the leaf photograph."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    line_w = max(2, min(canvas.size) // 180)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cid = int(det.get("class_id", 0))
        color = CLASS_COLORS_RGB[cid % len(CLASS_COLORS_RGB)]
        label = f"{det['class_name']} {det['confidence'] * 100:.1f}%"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
        text_bbox = draw.textbbox((x1, max(0.0, y1 - 16)), label, font=font)
        pad = 2
        draw.rectangle(
            [text_bbox[0] - pad, text_bbox[1] - pad, text_bbox[2] + pad, text_bbox[3] + pad],
            fill=color,
        )
        draw.text((x1, max(0.0, y1 - 16)), label, fill=(255, 255, 255), font=font)

    return canvas


def build_detector(
    num_classes: Optional[int] = None,
    width_mult: float = 0.5,
    input_size: int = 640,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
) -> CottonLeafYOLODetector:
    """Constructs CottonLeafYOLODetector and optionally loads verified trained weights."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    class_names = load_detection_classes()
    resolved_classes = num_classes if num_classes is not None else len(class_names)

    model = CottonLeafYOLODetector(
        num_classes=resolved_classes,
        width_mult=width_mult,
        input_size=input_size,
        class_names=class_names[:resolved_classes],
    )

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"YOLO checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

    model.to(device)
    return model
