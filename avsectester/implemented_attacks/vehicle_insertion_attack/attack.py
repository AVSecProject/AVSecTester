"""Configure recorded-trajectory placement and world insertion of a vehicle mesh."""
from __future__ import annotations

import math
from numbers import Real
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

from avsectester.attack import Attack, AttackRuntime, AttackSegment
from avsectester.geometry.poses import validated_transform
from avsectester.rendering import Harmonizer, MeshAsset
from avsectester.validation import require_positive_integer, require_positive_number

from .metabuilder import build_metadata
from .online import VehicleInsertionAttackRuntime

if TYPE_CHECKING:
    from avsectester.execution.services import ScenarioView


class VehicleInsertionAttack(Attack):
    """Place a vehicle mesh ahead on the recorded ego path and keep it fixed in the world.

    Each segment uses the first recorded pose at least ``distance_m`` along the path from its
    start. The full pose preserves the recorded slope and heading. ``lateral_offset_m`` shifts
    the vehicle along that pose's left axis. The asset supplies its own origin and dimensions.
    No offline optimization is performed.
    """

    def __init__(
        self, asset: MeshAsset, *, distance_m: float = 25.0, lateral_offset_m: float = 0.0,
        decision_steps: int = 20, max_segments: int | None = None,
        harmonizer: Harmonizer | None = None,
    ) -> None:
        if not isinstance(asset, MeshAsset):
            raise TypeError("asset must be a MeshAsset")
        require_positive_number(distance_m, "distance_m")
        if (isinstance(lateral_offset_m, bool) or not isinstance(lateral_offset_m, Real)
                or not math.isfinite(lateral_offset_m)):
            raise ValueError("lateral_offset_m must be a finite number")
        require_positive_integer(decision_steps, "decision_steps")
        if max_segments is not None:
            require_positive_integer(max_segments, "max_segments")
        if harmonizer is not None and not callable(harmonizer):
            raise TypeError("harmonizer must be callable or None")
        self.asset: MeshAsset = asset
        self.distance_m: float = float(distance_m)
        self.lateral_offset_m: float = float(lateral_offset_m)
        self.decision_steps: int = decision_steps
        self.max_segments: int | None = max_segments
        self.harmonizer: Harmonizer | None = harmonizer

    def build_metadata(self, scenario: ScenarioView) -> Sequence[AttackSegment]:
        return build_metadata(self, scenario)

    def validate_segment(self, segment: AttackSegment) -> None:
        """Require a rigid world-from-object pose in selected or loaded segment data."""
        if "world_from_object" not in segment.user_data:
            raise ValueError("VehicleInsertionAttack metadata requires world_from_object")
        validated_transform(segment.user_data["world_from_object"])

    def create_runtime(self, segment: AttackSegment, artifacts: Any) -> AttackRuntime:
        self.validate_segment(segment)
        return VehicleInsertionAttackRuntime(
            self.asset, segment.user_data["world_from_object"], self.harmonizer)


__all__ = ['VehicleInsertionAttack']
