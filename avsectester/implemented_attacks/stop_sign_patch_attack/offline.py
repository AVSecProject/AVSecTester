"""Validated detector/rollout optimization options and offline dispatch."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from avsectester.attack import OfflineContext
from avsectester.attack.optimization import GradientOptimizer, NES, Scorer
from avsectester.validation import require_positive_integer, require_positive_number

from .assets import initial_patch_texture

if TYPE_CHECKING:
    from .attack import StopSignPatchAttack


def _output_path(value: str | Path) -> None:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("output_directory must be a nonempty filesystem path")


@dataclass(frozen=True)
class DetectorPatchOptimization:
    """PGD sampling and scoring settings for one shared physical patch."""

    detector: Scorer
    optimizer: GradientOptimizer
    output_directory: str | Path
    camera_id: str = "camera_front_wide_120fov"
    scene_count: int = 4
    samples_per_scene: int = 8
    min_visible_pixels: int = 100
    detection_threshold: float = .5
    braking_deceleration_mps2: float = 3.4

    def __post_init__(self) -> None:
        from avsectester.attack.optimization import PGD

        if not isinstance(self.detector, Scorer):
            raise TypeError("detector must be a Scorer")
        if not isinstance(self.optimizer, PGD):
            raise TypeError("optimizer must be PGD for detector patch optimization")
        _output_path(self.output_directory)
        if not isinstance(self.camera_id, str) or not self.camera_id:
            raise ValueError("camera_id must be a nonempty sensor name")
        require_positive_integer(self.scene_count, "scene_count")
        require_positive_integer(self.samples_per_scene, "samples_per_scene")
        if self.samples_per_scene < 4:
            raise ValueError("samples_per_scene must be at least 4")
        require_positive_integer(self.min_visible_pixels, "min_visible_pixels")
        if (isinstance(self.detection_threshold, bool)
                or not isinstance(self.detection_threshold, (int, float))
                or not np.isfinite(self.detection_threshold)
                or not 0 <= self.detection_threshold <= 1):
            raise ValueError("detection_threshold must be in [0, 1]")
        require_positive_number(self.braking_deceleration_mps2, "braking_deceleration_mps2")


@dataclass(frozen=True)
class RolloutPatchOptimization:
    """NES settings for one-decision planned-forward-distance queries."""

    optimizer: GradientOptimizer
    output_directory: str | Path
    start_distances_m: tuple[float, ...] = (40., 30., 20.)
    horizon_s: float = 3.
    scene_count: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer, NES):
            raise TypeError("optimizer must be NES for rollout patch optimization")
        _output_path(self.output_directory)
        distances = tuple(self.start_distances_m)
        if not distances:
            raise ValueError("start_distances_m must not be empty")
        for value in distances:
            require_positive_number(value, "start_distances_m")
        object.__setattr__(self, "start_distances_m", distances)
        require_positive_number(self.horizon_s, "horizon_s")
        require_positive_integer(self.scene_count, "scene_count")


def run_offline(attack: StopSignPatchAttack, context: OfflineContext) -> np.ndarray:
    """Create the configured initial patch, then optionally optimize that same texture."""
    initial = initial_patch_texture(attack.initial_patch, attack.mask,
                                   noise_std=attack.noise_std, noise_seed=attack.noise_seed)
    if attack.optimization is None:
        return initial
    if isinstance(attack.optimization, DetectorPatchOptimization):
        from .detector_optimization import optimize_patch

        return optimize_patch(attack, context, initial)
    if isinstance(attack.optimization, RolloutPatchOptimization):
        from .rollout_optimization import optimize_patch_by_rollouts

        return optimize_patch_by_rollouts(attack, context, initial)
    raise TypeError("optimization must be DetectorPatchOptimization or RolloutPatchOptimization")
