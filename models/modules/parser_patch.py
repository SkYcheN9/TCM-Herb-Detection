"""Utilities for safely extending the Ultralytics YAML parser."""

from __future__ import annotations

import inspect
from collections.abc import Callable


def patch_parse_model(feature: str, transform: Callable[[str], str]) -> None:
    """Apply an idempotent source transform to ``ultralytics.nn.tasks.parse_model``."""

    import ultralytics.nn.tasks as tasks

    features = set(getattr(tasks.parse_model, "_tcm_features", set()))
    if feature in features:
        return

    source = getattr(tasks.parse_model, "_tcm_source", None)
    if source is None:
        source = tasks.__dict__.get("_tcm_parse_model_source")
    if source is None:
        source = inspect.getsource(tasks.parse_model)

    patched_source = transform(source)
    if patched_source == source and feature not in patched_source:
        raise RuntimeError(f"Unable to patch Ultralytics parser for {feature}")

    exec(patched_source, tasks.__dict__)
    features.add(feature)
    tasks.parse_model._tcm_features = features
    tasks.parse_model._tcm_source = patched_source
    tasks.__dict__["_tcm_parse_model_source"] = patched_source
