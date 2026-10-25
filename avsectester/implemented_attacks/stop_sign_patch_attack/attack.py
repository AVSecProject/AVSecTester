"""Connect stop-approach selection, optional patch optimization and deployment."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from avsectester.attack import Attack, AttackRuntime, AttackSegment, OfflineContext
from avsectester.execution.services import ScenarioView
from avsectester.rendering import ObjectAsset
from avsectester.validation import (
    require_nonnegative_integer, require_nonnegative_number, require_positive_number,
)

from .assets import DEFAULT_MASK_PATH, patch_asset, patch_pose, read_patch_mask, validate_patch_texture
from .metabuilder import control_from_user_data, select_stop_approach
from .offline import DetectorPatchOptimization, RolloutPatchOptimization, run_offline
from .online import StopSignPatchAttackRuntime


class StopSignPatchAttack(Attack):
    """Place masked pixels on recorded stop signs, preserving the reconstructed sign.

    One approach is selected per scenario. Optional detector PGD or rollout NES
    prepares a shared texture before each selected segment creates its runtime.
    """

    def __init__(
        self, initial_patch: str = "white",
        optimization: DetectorPatchOptimization | RolloutPatchOptimization | None = None,
        mask_path: str | Path = DEFAULT_MASK_PATH, noise_std: float = .25, noise_seed: int = 0,
        approach_m: float = 50., hold_steps: int = 20, offset_m: float = .10,
        width_scale: float = 1.,
    ) -> None:
        if initial_patch not in ("white", "black", "gaussian_noise"):
            raise ValueError("initial_patch must be white, black or gaussian_noise")
        if optimization is not None and not isinstance(
            optimization, (DetectorPatchOptimization, RolloutPatchOptimization),
        ):
            raise TypeError("optimization must be DetectorPatchOptimization or RolloutPatchOptimization")
        require_nonnegative_number(noise_std, "noise_std")
        require_nonnegative_integer(noise_seed, "noise_seed")
        require_positive_number(approach_m, "approach_m")
        require_nonnegative_integer(hold_steps, "hold_steps")
        require_nonnegative_number(offset_m, "offset_m")
        require_positive_number(width_scale, "width_scale")
        self.initial_patch = initial_patch
        self.optimization = optimization
        self.mask_path = mask_path
        self.noise_std = noise_std
        self.noise_seed = noise_seed
        self.approach_m = approach_m
        self.hold_steps = hold_steps
        self.offset_m = offset_m
        self.width_scale = width_scale
        self.mask = read_patch_mask(mask_path)
        self.mask.setflags(write=False)

    def build_metadata(self, scenario: ScenarioView) -> Sequence[AttackSegment]:
        selected = select_stop_approach(scenario, self.approach_m, self.hold_steps)
        return () if selected is None else (selected.to_segment(),)

    def run_offline(self, context: OfflineContext) -> np.ndarray:
        return run_offline(self, context)

    def patch_placement(
        self, segment: AttackSegment, texture: np.ndarray,
    ) -> tuple[ObjectAsset, np.ndarray]:
        """Use the same panel size and face-centre placement offline and online."""
        control = control_from_user_data(segment.user_data)
        asset = patch_asset(control, texture, self.width_scale)
        pose = patch_pose(control, self.offset_m, panel_height_m=asset.extent[2])
        return asset, pose

    def patch_texture(self, artifacts: object) -> np.ndarray:
        return validate_patch_texture(artifacts, self.mask)

    def create_runtime(self, segment: AttackSegment, artifacts: object) -> AttackRuntime:
        texture = self.patch_texture(artifacts)
        asset, pose = self.patch_placement(segment, texture)
        return StopSignPatchAttackRuntime(asset, pose)
