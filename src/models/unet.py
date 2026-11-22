"""
U-Net Semantic Segmentation Architecture for Cotton Leaf AI.

Standard encoder-decoder architecture with skip connections designed for
3-class cotton leaf tissue segmentation:
  - Class 0: Background
  - Class 1: Healthy Leaf Lamina
  - Class 2: Diseased / Necrotic Lesion Tissue
"""

from pathlib import Path
from typing import Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscaling with MaxPool2D then DoubleConv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upscaling with ConvTranspose2D then DoubleConv with skip connection"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Handle possible padding differences if spatial dimensions are odd
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CottonLeafUNet(nn.Module):
    """
    Standard U-Net for Cotton Leaf semantic segmentation.
    Configured with 3 output channels (Background, Healthy Leaf, Diseased Tissue).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        init_features: int = 32,
        bilinear: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        f = init_features
        self.inc = DoubleConv(in_channels, f)
        self.down1 = DownBlock(f, f * 2)
        self.down2 = DownBlock(f * 2, f * 4)
        self.down3 = DownBlock(f * 4, f * 8)
        factor = 2 if bilinear else 1
        self.down4 = DownBlock(f * 8, f * 16 // factor)

        self.up1 = UpBlock(f * 16, f * 8 // factor, bilinear)
        self.up2 = UpBlock(f * 8, f * 4 // factor, bilinear)
        self.up3 = UpBlock(f * 4, f * 2 // factor, bilinear)
        self.up4 = UpBlock(f * 2, f, bilinear)
        self.outc = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def build_unet(
    in_channels: int = 3,
    num_classes: int = 3,
    init_features: int = 32,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None
) -> CottonLeafUNet:
    """Builds and optionally loads a CottonLeafUNet model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CottonLeafUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        init_features=init_features
    )

    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"U-Net checkpoint not found: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)

    model.to(device)
    return model
