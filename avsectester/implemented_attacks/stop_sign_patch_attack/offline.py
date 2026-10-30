"""Validated detector/rollout optimization options and offline dispatch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from avsectester.attack import OfflineContext
from avsectester.attack.optimization import GradientOptimizer, NES
from avsectester.validation import (
    require_nonnegative_integer, require_nonnegative_number, require_positive_integer, require_positive_number,
)

from .assets import initial_patch_texture

if TYPE_CHECKING:
    from .attack import StopSignPatchAttack


@dataclass(frozen=True)
class DetectorPatchOptimization:
    """White-box attack settings (algorithm used: PGD).

    ``braking_deceleration_mps2`` is the driving system's strongest braking. It sets the closest
    trained distance to the stop line. ``samples_per_scene="all"`` trains on every eligible frame.
    An integer of at least four selects that many time-uniform frames per training segment.
    ``confidence_margin`` sets the hinge target: PGD lowers each frame's confidence to the driving
    system's decision threshold minus this margin. Every ``progress_interval`` PGD epochs, and after
    the last, the current patch is scored on every training frame for the offline report.
    """

    optimizer: GradientOptimizer
    scorer_id: str | None = None
    device: str = "cpu"
    max_training_segments: int | None = None
    samples_per_scene: int | Literal["all"] = "all"
    min_visible_pixels: int = 100
    braking_deceleration_mps2: float = 6.
    confidence_margin: float = .1
    progress_interval: int = 1

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
        if self.samples_per_scene != "all":
            if (isinstance(self.samples_per_scene, bool) or not isinstance(self.samples_per_scene, int)
                    or self.samples_per_scene < 4):
                raise ValueError('samples_per_scene must be "all" or an integer of at least 4')
        require_positive_integer(self.min_visible_pixels, "min_visible_pixels")
        require_positive_number(self.braking_deceleration_mps2, "braking_deceleration_mps2")
        require_nonnegative_number(self.confidence_margin, "confidence_margin")
        if self.confidence_margin > 1:
            raise ValueError("confidence_margin must be at most 1")
        require_positive_integer(self.progress_interval, "progress_interval")


@dataclass(frozen=True)
class RolloutPatchOptimization:
    """Black-box attack settings (algorithm used: NES)."""

    optimizer: GradientOptimizer
    stop_speed_mps: float = .5
    min_stop_command_step: int = 5
    max_training_segments: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.optimizer, NES):
            raise TypeError("optimizer must be NES for rollout patch optimization")
        require_positive_number(self.stop_speed_mps, "stop_speed_mps")
        require_nonnegative_integer(self.min_stop_command_step, "min_stop_command_step")
        if self.max_training_segments is not None:
            require_positive_integer(self.max_training_segments, "max_training_segments")


def run_offline(attack: StopSignPatchAttack, context: OfflineContext) -> np.ndarray:
    """Create the configured initial patch, then optionally optimize that same texture."""
    initial = initial_patch_texture(attack.initial_patch, attack.mask,
                                   noise_std=attack.noise_std, noise_seed=attack.noise_seed,
                                   noise_cell_texels=attack.noise_cell_texels)
    context.report.set_image(
        "initial_texture", attack.replica_texture(initial),
        description="Initial texture, before any optimization.",
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


__all__ = [
    'DetectorPatchOptimization',
    'RolloutPatchOptimization',
]
