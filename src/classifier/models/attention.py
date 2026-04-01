"""Attention modules for chest X-ray classification.

Implements SE-Block, CBAM, and ECA attention mechanisms.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block.

    Architecture:
    1. Squeeze: Global Average Pooling → [B, C, 1, 1]
    2. Excitation: FC → ReLU → FC → Sigmoid
    3. Scale: Multiply original features

    Reference: Hu et al. "Squeeze-and-Excitation Networks" (CVPR 2018)
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for channel dimension
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Scaled feature tensor of shape [B, C, H, W]
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Architecture:
    1. Channel Attention Module (CAM)
    2. Spatial Attention Module (SAM)
    3. Sequential application

    Reference: Woo et al. "CBAM: Convolutional Block Attention Module" (ECCV 2018)
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        """
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for channel attention
            kernel_size: Kernel size for spatial attention convolution
        """
        super().__init__()
        # Channel Attention Module
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

        # Spatial Attention Module
        self.spatial_conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Refined feature tensor of shape [B, C, H, W]
        """
        # Channel Attention
        b, c, _, _ = x.size()
        avg_out = self.channel_fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.channel_fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        channel_weight = self.sigmoid(avg_out + max_out)
        x = x * channel_weight

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_weight = self.sigmoid(self.spatial_conv(spatial_input))
        x = x * spatial_weight

        return x


class ECA(nn.Module):
    """Efficient Channel Attention.

    Architecture:
    1. Global Average Pooling
    2. 1D Convolution with adaptive kernel
    3. Sigmoid activation

    Reference: Wang et al. "ECA-Net: Efficient Channel Attention" (CVPR 2020)
    """

    def __init__(self, channels: int, gamma: float = 2.0, b: float = 1.0):
        """
        Args:
            channels: Number of input channels
            gamma: Gamma parameter for adaptive kernel size
            b: Beta parameter for adaptive kernel size
        """
        super().__init__()
        # Adaptive kernel size calculation
        kernel_size = int(abs((math.log2(channels) + b) / gamma))
        kernel_size = max(3, kernel_size if kernel_size % 2 else kernel_size + 1)

        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Scaled feature tensor of shape [B, C, H, W]
        """
        # Global Average Pooling
        y = x.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

        # 1D Convolution
        y = y.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]

        # Sigmoid and scale
        return x * self.sigmoid(y)


def create_attention_module(
    attention_type: str, channels: int, **kwargs
) -> nn.Module:
    """Factory function to create attention modules.

    Args:
        attention_type: Type of attention ('se', 'cbam', 'eca', or None)
        channels: Number of input channels
        **kwargs: Additional arguments for specific attention modules

    Returns:
        Attention module

    Raises:
        ValueError: If attention_type is not recognized
    """
    if attention_type is None or attention_type.lower() == "none":
        return nn.Identity()
    elif attention_type.lower() == "se":
        return SEBlock(channels, reduction=kwargs.get("reduction", 16))
    elif attention_type.lower() == "cbam":
        return CBAM(
            channels,
            reduction=kwargs.get("reduction", 16),
            kernel_size=kwargs.get("kernel_size", 7),
        )
    elif attention_type.lower() == "eca":
        return ECA(channels, gamma=kwargs.get("gamma", 2.0), b=kwargs.get("b", 1.0))
    else:
        raise ValueError(
            f"Unsupported attention_type: {attention_type}. "
            "Choose from: None, 'se', 'cbam', 'eca'"
        )
