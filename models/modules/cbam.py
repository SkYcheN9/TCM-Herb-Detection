"""CBAM attention module compatible with Ultralytics model YAML parsing."""

from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    """Channel attention from the CBAM paper."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to an input feature map."""

        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)
        attention = self.shared_mlp(avg_pool) + self.shared_mlp(max_pool)
        return x * self.sigmoid(attention)


class SpatialAttention(nn.Module):
    """Spatial attention from the CBAM paper."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size not in {3, 7}:
            raise ValueError("CBAM spatial kernel_size must be 3 or 7")
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention to an input feature map."""

        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.amax(x, dim=1, keepdim=True)
        attention = torch.cat((avg_pool, max_pool), dim=1)
        return x * self.sigmoid(self.conv(attention))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Args:
        reduction: Reduction ratio for the channel-attention MLP.
        kernel_size: Spatial-attention kernel size, either 3 or 7.

    The channel count is inferred lazily on the first forward pass. This keeps
    the module compatible with Ultralytics' generic YAML parser, which treats
    unknown modules as shape-preserving layers and does not inject channels.
    """

    def __init__(self, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.channel_attention: ChannelAttention | None = None
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def _build_channel_attention(
        self,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.channel_attention = ChannelAttention(
            channels=channels,
            reduction=self.reduction,
        ).to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel and spatial attention without changing tensor shape."""

        if self.channel_attention is None:
            self._build_channel_attention(x.shape[1], x.device, x.dtype)
        x = self.channel_attention(x)
        return self.spatial_attention(x)


def register_ultralytics_modules() -> None:
    """Expose custom modules to Ultralytics' YAML parser."""

    import ultralytics.nn.tasks as tasks

    tasks.CBAM = CBAM
