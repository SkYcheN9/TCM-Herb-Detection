"""Utilities for transferring YOLOv8 pretrained weights into modified models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import torch
from torch import nn


LAYER_KEY_RE = re.compile(r"^model\.(\d+)\.")


@dataclass(frozen=True)
class TransferReport:
    """Summary of a conservative pretrained-weight transfer."""

    source: str
    exact_tensors: int
    remapped_tensors: int
    total_target_tensors: int

    @property
    def transferred_tensors(self) -> int:
        """Return the total number of tensors copied into the target model."""

        return self.exact_tensors + self.remapped_tensors

    @property
    def skipped_tensors(self) -> int:
        """Return target tensors that kept their random initialization."""

        return max(self.total_target_tensors - self.transferred_tensors, 0)

    def summary(self) -> str:
        """Return a concise human-readable transfer summary."""

        return (
            f"Transferred {self.transferred_tensors}/{self.total_target_tensors} tensors "
            f"from {self.source} "
            f"(exact={self.exact_tensors}, remapped={self.remapped_tensors}, "
            f"skipped={self.skipped_tensors})"
        )


def load_source_model(source: str | Path | nn.Module | dict[str, Any]) -> tuple[nn.Module, str]:
    """Load a pretrained source model or normalize an already loaded model."""

    if isinstance(source, nn.Module):
        return source, source.__class__.__name__
    if isinstance(source, dict):
        model = source.get("model")
        if not isinstance(model, nn.Module):
            raise TypeError("Pretrained checkpoint dict does not contain a torch model")
        return model, "checkpoint"

    from ultralytics import YOLO

    source_path = str(source)
    return YOLO(source_path).model, source_path


def layer_index(key: str) -> int | None:
    """Extract the top-level YOLO model layer index from a state-dict key."""

    match = LAYER_KEY_RE.match(key)
    return int(match.group(1)) if match else None


def same_layer_type(source_layers: list[nn.Module], target_layers: list[nn.Module], key: str) -> bool:
    """Return whether a same-name tensor belongs to matching layer classes."""

    index = layer_index(key)
    if index is None:
        return True
    if index >= len(source_layers) or index >= len(target_layers):
        return False
    return source_layers[index].__class__ is target_layers[index].__class__


def relative_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached CPU-friendly view of module state keyed relative to the module."""

    return {key: value for key, value in module.state_dict().items()}


def copy_exact_tensors(
    source_model: nn.Module,
    target_model: nn.Module,
    target_state: dict[str, torch.Tensor],
) -> set[str]:
    """Copy tensors with identical state-dict keys, shapes and layer classes."""

    copied: set[str] = set()
    source_state = source_model.state_dict()
    source_layers = list(getattr(source_model, "model", []))
    target_layers = list(getattr(target_model, "model", []))
    for key, source_tensor in source_state.items():
        target_tensor = target_state.get(key)
        if target_tensor is None:
            continue
        if target_tensor.shape != source_tensor.shape:
            continue
        if not same_layer_type(source_layers, target_layers, key):
            continue
        target_state[key].copy_(source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype))
        copied.add(key)
    return copied


def matching_relative_keys(source_module: nn.Module, target_module: nn.Module) -> list[str]:
    """Return same-name relative state keys that can be copied between two modules."""

    source_state = relative_state(source_module)
    target_state = relative_state(target_module)
    return [
        key
        for key, source_tensor in source_state.items()
        if key in target_state and target_state[key].shape == source_tensor.shape
    ]


def copy_remapped_modules(
    source_model: nn.Module,
    target_model: nn.Module,
    target_state: dict[str, torch.Tensor],
    already_copied: set[str],
) -> set[str]:
    """Copy tensors from same-type modules in order, allowing layer indices to shift."""

    copied: set[str] = set()
    source_layers = list(getattr(source_model, "model", []))
    target_layers = list(getattr(target_model, "model", []))
    backbone_only = any(layer.__class__.__name__ == "BiFPNFusion" for layer in target_layers)
    source_cursor = 0
    for target_index, target_module in enumerate(target_layers):
        if not relative_state(target_module):
            continue
        for source_index in range(source_cursor, len(source_layers)):
            source_module = source_layers[source_index]
            if source_module.__class__ is not target_module.__class__:
                continue
            rel_keys = matching_relative_keys(source_module, target_module)
            if not rel_keys:
                continue
            source_prefix = f"model.{source_index}."
            target_prefix = f"model.{target_index}."
            source_state = source_model.state_dict()
            for rel_key in rel_keys:
                target_key = target_prefix + rel_key
                source_key = source_prefix + rel_key
                if target_key in already_copied:
                    continue
                if target_key not in target_state or source_key not in source_state:
                    continue
                target_tensor = target_state[target_key]
                source_tensor = source_state[source_key]
                target_state[target_key].copy_(
                    source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
                )
                copied.add(target_key)
            source_cursor = source_index + 1
            if backbone_only and target_module.__class__.__name__ == "SPPF":
                return copied
            break
    return copied


def transfer_pretrained_weights(
    target_model: nn.Module,
    source: str | Path | nn.Module | dict[str, Any],
) -> TransferReport:
    """Transfer compatible pretrained tensors into a modified YOLO model.

    The transfer is intentionally conservative. It first copies same-name,
    same-shape tensors whose top-level layer class still matches. It then walks
    both YOLO module lists in order and copies same-type module tensors whose
    relative names and shapes match, which recovers weights after CBAM or other
    inserted layers shift numeric indices.
    """

    source_model, source_name = load_source_model(source)
    source_model = source_model.float()
    target_state = target_model.state_dict()
    exact = copy_exact_tensors(source_model, target_model, target_state)
    remapped = copy_remapped_modules(source_model, target_model, target_state, exact)
    target_model.load_state_dict(target_state, strict=False)
    return TransferReport(
        source=source_name,
        exact_tensors=len(exact),
        remapped_tensors=len(remapped),
        total_target_tensors=len(target_state),
    )
