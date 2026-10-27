"""Validated detector/rollout optimization options and offline dispatch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from avsectester.attack import OfflineContext
from avsectester.attack.optimization import GradientOptimizer, NES
from avsectester.validation import require_positive_integer, require_positive_number

from .assets import initial_patch_texture

if TYPE_CHECKING:
    from .attack import StopSignPatchAttack


@dataclass(frozen=True)
class DetectorPatchOptimization:
    """PGD settings; an optional segment cap limits cost within the training metadata."""

    optimizer: GradientOptimizer
    scorer_id: str | None = None
    device: str = "cpu"
    max_training_segments: int | None = None
    samples_per_scene: int = 8
    min_visible_pixels: int = 100
    braking_deceleration_mps2: float = 3.4

    def __post_init__(self) -> None:
        from avsectester.attack.optimization import PGD

        if not isinstance(self.optimizer, PGD):
            raise TypeError("optimizer must be PGD for detector patch optimization")
        if self.scorer_id is not None and (not isinstance(self.scorer_id, str) or not self.scorer_id):
            raise ValueError("scorer_id must be a nonempty string or None")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty string")
        if self.max_training_segments is not None:
            require_positive_integer(self.max_training_segments, "max_training_segments")
        require_positive_integer(self.samples_per_scene, "samples_per_scene")
        if self.samples_per_scene < 4:
            raise ValueError("samples_per_scene must be at least 4")
        require_positive_integer(self.min_visible_pixels, "min_visible_pixels")
        require_positive_number(self.braking_deceleration_mps2, "braking_deceleration_mps2")


@dataclass(frozen=True)
class RolloutPatchOptimization:
    """NES settings; use all training segments unless a cost-limiting cap is supplied."""

    optimizer: GradientOptimizer
    start_distances_m: tuple[float, ...] = (40., 30., 20.)
    horizon_s: float = 3.
    max_training_segments: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer, NES):
            raise TypeError("optimizer must be NES for rollout patch optimization")
        distances = tuple(self.start_distances_m)
        if not distances:
            raise ValueError("start_distances_m must not be empty")
        for value in distances:
            require_positive_number(value, "start_distances_m")
        object.__setattr__(self, "start_distances_m", distances)
        require_positive_number(self.horizon_s, "horizon_s")
        if self.max_training_segments is not None:
            require_positive_integer(self.max_training_segments, "max_training_segments")


def run_offline(attack: StopSignPatchAttack, context: OfflineContext) -> np.ndarray:
    """Create the configured initial patch, then optionally optimize that same texture."""
    initial = initial_patch_texture(attack.initial_patch, attack.mask,
                                   noise_std=attack.noise_std, noise_seed=attack.noise_seed)
    context.report.set_image(
        "initial_texture", attack.replica_texture(initial),
        description="Replica face with the initial patch, before any optimization.",
    )
    if attack.optimization is None:
        return initial
    if isinstance(attack.optimization, DetectorPatchOptimization):
        from .detector_optimization import optimize_patch

        return optimize_patch(attack, context, initial)
    if isinstance(attack.optimization, RolloutPatchOptimization):
        from .rollout_optimization import optimize_patch_by_rollouts

        return optimize_patch_by_rollouts(attack, context, initial)
    raise TypeError("optimization must be DetectorPatchOptimization or RolloutPatchOptimization")
