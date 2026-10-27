"""Data-only compatibility summaries for configured components and preparation inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import math
from pathlib import Path
from typing import Any


def type_name(value: Any) -> str:
    """Fully qualified class path of a configured object, without inspecting its state."""
    return f"{type(value).__module__}:{type(value).__qualname__}"


def summary_value(value: Any) -> Any:
    """Summarize configuration data; never inspect arbitrary object state or live handles."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Configuration summaries require finite numbers")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Configuration parameter mappings require string keys")
        return {key: summary_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [summary_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {"type": type_name(value), "parameters": {
            item.name: summary_value(getattr(value, item.name)) for item in fields(value) if item.init
        }}
    import numpy as np

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Configuration arrays must not contain Python objects")
        return {"array_dtype": value.dtype.str, "array_shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return summary_value(value.item())
    raise TypeError(f"Cannot summarize configuration parameter {type(value).__name__}; "
                    "override Attack.configuration_parameters() with a data description")


