"""BiFPN neck modules compatible with Ultralytics model YAML parsing."""

from __future__ import annotations

import inspect

import torch
from torch import nn
import torch.nn.functional as F


class SeparableConvBlock(nn.Module):
    """Depthwise separable convolution used after BiFPN feature fusion."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise separable convolution."""

        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class BiFPNFusion(nn.Module):
    """Weighted BiFPN feature fusion for same-channel feature maps.

    Args:
        num_inputs: Number of feature maps to fuse.
        channels: Feature channel count after lateral projection.
        epsilon: Small value used by fast normalized fusion.

    The fusion weights are learned automatically and normalized on each forward
    pass, following the fast normalized fusion idea from BiFPN.
    """

    def __init__(self, num_inputs: int, channels: int, epsilon: float = 1e-4) -> None:
        super().__init__()
        if num_inputs < 2:
            raise ValueError("BiFPNFusion requires at least two input feature maps")
        self.num_inputs = num_inputs
        self.channels = channels
        self.epsilon = epsilon
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))
        self.conv = SeparableConvBlock(channels)

    def normalized_weights(self) -> torch.Tensor:
        """Return non-negative fusion weights normalized to sum to one."""

        weights = F.relu(self.weights)
        return weights / (weights.sum() + self.epsilon)

    def forward(self, inputs: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Fuse feature maps with automatically normalized learnable weights."""

        if len(inputs) != self.num_inputs:
            raise ValueError(
                f"BiFPNFusion expected {self.num_inputs} inputs, got {len(inputs)}"
            )

        target_size = inputs[0].shape[-2:]
        weights = self.normalized_weights().to(device=inputs[0].device, dtype=inputs[0].dtype)
        fused = 0
        for weight, feature in zip(weights, inputs):
            if feature.shape[-2:] != target_size:
                feature = F.interpolate(feature, size=target_size, mode="nearest")
            fused = fused + weight * feature
        return self.conv(fused)


def register_ultralytics_modules() -> None:
    """Expose BiFPN modules and parser rules to Ultralytics."""

    import ultralytics.nn.tasks as tasks

    tasks.BiFPNFusion = BiFPNFusion
    if getattr(tasks.parse_model, "_tcm_bifpn_patched", False):
        return

    source = inspect.getsource(tasks.parse_model)
    anchor = "        elif m is AIFI:"
    patch = (
        "        elif m is BiFPNFusion:\n"
        "            c2 = args[0]\n"
        "            if c2 != nc:\n"
        "                c2 = make_divisible(min(c2, max_channels) * width, 8)\n"
        "            args = [len(f) if isinstance(f, list) else 1, c2, *args[1:]]\n"
    )
    if anchor not in source:
        raise RuntimeError("Unable to patch Ultralytics parser for BiFPNFusion")

    exec(source.replace(anchor, patch + anchor), tasks.__dict__)
    tasks.parse_model._tcm_bifpn_patched = True
