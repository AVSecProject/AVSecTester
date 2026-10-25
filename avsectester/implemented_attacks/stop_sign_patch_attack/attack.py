"""Connect stop-approach selection, optional patch optimization and replica deployment."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from avsectester.attack import Attack, AttackRuntime, AttackSegment, OfflineContext
from avsectester.execution.services import ScenarioView
from avsectester.rendering import Harmonizer, ObjectAsset
from avsectester.validation import (
    require_nonnegative_integer, require_nonnegative_number, require_positive_number,
)

from .assets import (
    DEFAULT_MASK_PATH, INITIAL_PATCHES, read_patch_mask, replica_asset, replica_pose,
    validate_replica_texture,
)
from .metabuilder import control_from_user_data, select_stop_approach
from .offline import DetectorPatchOptimization, RolloutPatchOptimization, run_offline
from .online import StopSignPatchAttackRuntime


class StopSignPatchAttack(Attack):
    """Cover a recorded stop sign with a replica face whose masked pixels form a patch.

    One approach is selected per scenario. The replica is the canonical MUTCD face, scaled
    by the MUTCD size closest to the mapped height and placed upright in front of the mapped
    face centre. Optional map orientation and width modes preserve the annotation geometry.
    The patch lies in face-texture coordinates and never moves the legend.
    ``initial_patch="none"`` deploys the unpatched replica as a control. Optional detector PGD
    or rollout NES optimizes the masked pixels of the initial patch offline.
    """

    def __init__(
        self, initial_patch: str = "white",
        optimization: DetectorPatchOptimization | RolloutPatchOptimization | None = None,
        mask_path: str | Path = DEFAULT_MASK_PATH, noise_std: float = .25, noise_seed: int = 0,
        approach_m: float = 50., hold_steps: int = 20, offset_m: float = .10,
        width_scale: float = 1.1, harmonizer: Harmonizer | None = None,
        face_orientation: str = "upright", face_size: str = "mutcd_standard",
        min_approach_m: float = 20.,
    ) -> None:
        if initial_patch not in INITIAL_PATCHES:
            raise ValueError("initial_patch must be none, white, black or gaussian_noise")
        if initial_patch == "none" and optimization is not None:
            raise ValueError("initial_patch none deploys the unpatched replica and cannot be optimized")
        if optimization is not None and not isinstance(
            optimization, (DetectorPatchOptimization, RolloutPatchOptimization),
        ):
            raise TypeError("optimization must be DetectorPatchOptimization or RolloutPatchOptimization")
        if face_orientation not in ("upright", "map"):
            raise ValueError("face_orientation must be upright or map")
        if face_size not in ("mutcd_standard", "map_width"):
            raise ValueError("face_size must be mutcd_standard or map_width")
        require_nonnegative_number(noise_std, "noise_std")
        require_nonnegative_integer(noise_seed, "noise_seed")
        require_positive_number(approach_m, "approach_m")
        require_nonnegative_number(min_approach_m, "min_approach_m")
        if min_approach_m > approach_m:
            raise ValueError("min_approach_m must not exceed approach_m")
        require_nonnegative_integer(hold_steps, "hold_steps")
        require_nonnegative_number(offset_m, "offset_m")
        require_positive_number(width_scale, "width_scale")
        self.initial_patch = initial_patch
        self.optimization = optimization
        self.mask_path = mask_path
        self.noise_std = noise_std
        self.noise_seed = noise_seed
        self.approach_m = approach_m
        self.min_approach_m = min_approach_m
        self.hold_steps = hold_steps
        self.offset_m = offset_m
        self.width_scale = width_scale
        self.harmonizer = harmonizer
        self.face_orientation = face_orientation
        self.face_size = face_size
        self.mask = read_patch_mask(mask_path)
        self.mask.setflags(write=False)

    def build_metadata(self, scenario: ScenarioView) -> Sequence[AttackSegment]:
        selected = select_stop_approach(scenario, self.approach_m, self.hold_steps, self.min_approach_m)
        return () if selected is None else (selected.to_segment(),)

    def run_offline(self, context: OfflineContext) -> np.ndarray:
        return run_offline(self, context)

    def replica_placement(
        self, segment: AttackSegment, texture: np.ndarray,
    ) -> tuple[ObjectAsset, np.ndarray]:
        """Share the selected MUTCD/map width and upright/map orientation across all phases.

        Preserve the mapped face centre and offset along the selected normal; no positional
        or yaw correction is applied to individual scenarios.
        """
        control = control_from_user_data(segment.user_data)
        asset = replica_asset(control, texture, self.width_scale, self.face_size)
        pose = replica_pose(control, self.offset_m, panel_height_m=asset.extent[2],
                            face_orientation=self.face_orientation)
        return asset, pose

    def replica_texture(self, artifacts: object) -> np.ndarray:
        return validate_replica_texture(artifacts, self.mask)

    def create_runtime(self, segment: AttackSegment, artifacts: object) -> AttackRuntime:
        texture = self.replica_texture(artifacts)
        asset, pose = self.replica_placement(segment, texture)
        return StopSignPatchAttackRuntime(asset, pose, self.harmonizer)
