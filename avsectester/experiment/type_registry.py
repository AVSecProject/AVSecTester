"""Resolve component names and discover attack classes without constructing components."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
import re
from typing import Any

from avsectester.attack import Attack
from avsectester.type_paths import import_type_path


def resolve_type(value: Any) -> Callable[..., Any]:
    """Resolve a built-in class name, or user code written as ``package.module:name``."""
    if not isinstance(value, str) or not value:
        raise ValueError("type must be a built-in class name or 'package.module:name'")
    return import_type_path(value) if ":" in value else _builtin(value)


_BUILTINS = {
    "RuleBasedAVStack": "avsectester.closed_loop.avstacks.implemented.rule_based:RuleBasedAVStack",
    "CameraForwardCollisionAVStack": "avsectester.closed_loop.avstacks.implemented.camera_forward_collision:CameraForwardCollisionAVStack",
    "DrivingMetrics": "avsectester.evaluation.metric:DrivingMetrics",
    "DetectionMetrics": "avsectester.evaluation.metric:DetectionMetrics",
    "TrajectoryDeviation": "avsectester.evaluation.metric:TrajectoryDeviation",
    "StopImpact": "avsectester.evaluation.metric:StopImpact",
    "StopLineMetrics": "avsectester.evaluation.metric:StopLineMetrics",
    "ReasoningPhraseRate": "avsectester.evaluation.metric:ReasoningPhraseRate",

    "CarlaDataset": "avsectester.datasets.carla:CarlaDataset",
    "NuRecDataset": "avsectester.datasets.nurec:NuRecDataset",
    "NuScenesDataset": "avsectester.datasets.nuscenes:NuScenesDataset",
    "CarlaSimulator": "avsectester.closed_loop.simulators.carla:CarlaSimulator",
    "NuRecSimulator": "avsectester.closed_loop.simulators.nurec:NuRecSimulator",
    "RecordedReplaySimulator": "avsectester.closed_loop.simulators.recorded:RecordedReplaySimulator",
    "ModularAVStack": "avsectester.closed_loop.avstacks.implemented.modular:ModularAVStack",
    "AlpamayoAVStack": "avsectester.closed_loop.avstacks.implemented.alpamayo:AlpamayoAVStack",
}


def attack_directory_name(class_name: str) -> str:
    """Convert PascalCase to snake_case, preserving acronym groups (MyNNAttack -> my_nn_attack)."""
    words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", class_name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", words).lower()


def _attack_class_name(name: str) -> bool:
    return name.isidentifier() and name[0].isupper()


def _exported_attack(module: Any, name: str, folder: str) -> type[Attack] | None:
    """Apply the same name/class/export/folder rule to lookup and discovery."""
    attack = getattr(module, name, None)
    if (not _attack_class_name(name) or not isinstance(attack, type)
            or not issubclass(attack, Attack) or attack is Attack
            or attack.__name__ != name or attack_directory_name(name) != folder):
        return None
    return attack


def _implemented_attack(value: str) -> Callable[..., Any] | None:
    package = import_module("avsectester.implemented_attacks")
    if not _attack_class_name(value):
        return None
    folder = attack_directory_name(value)
    if not any((Path(root) / folder / "__init__.py").is_file() for root in package.__path__):
        return None
    module = import_module(f"{package.__name__}.{folder}")
    attack = _exported_attack(module, value, folder)
    if attack is None:
        raise TypeError(f"{module.__name__} must export the Attack subclass {value}")
    return attack


def _discover_attacks() -> tuple[tuple[str, ...], dict[str, str]]:
    """Return accepted attack class names and the folders that failed to import."""
    package = import_module("avsectester.implemented_attacks")
    names: set[str] = set()
    failures: dict[str, str] = {}
    for root in package.__path__:
        for source in Path(root).glob("*/__init__.py"):
            folder = source.parent.name
            try:
                module = import_module(f"{package.__name__}.{folder}")
            except Exception as error:
                failures[folder] = f"{type(error).__name__}: {error}"
                continue
            names.update(name for name in vars(module)
                         if _exported_attack(module, name, folder) is not None)
    return tuple(sorted(names)), failures


def available_attacks() -> tuple[str, ...]:
    """List exactly the exported Attack classes accepted by class-name lookup.

    Attack folders that fail to import are omitted; an unknown type's error message names them.
    """
    return _discover_attacks()[0]


def _builtin(value: str) -> Callable[..., Any]:
    if value in _BUILTINS:
        return import_type_path(_BUILTINS[value])
    attack = _implemented_attack(value)
    if attack is not None:
        return attack
    names, failures = _discover_attacks()
    message = (f"Unknown built-in type {value!r}. Use an exact class name or "
               f"'package.module:name'. Available attacks: {', '.join(names) or 'none'}")
    if failures:
        message += ". Attack folders that failed to import: " + "; ".join(
            f"{folder} ({reason})" for folder, reason in sorted(failures.items()))
    raise ValueError(message)


def is_builtin_type_name(value: str) -> bool:
    """Whether a name belongs to the explicit component-name registry.

    Discovered attack names are resolved separately. Native third-party registry
    names are left to their owning constructor when nested in configuration options.
    """
    return value in _BUILTINS


__all__ = [
    'resolve_type',
    'available_attacks',
]
