"""Connect object assets, selection and online image insertion."""
from __future__ import annotations

import math

from avsectester.attack.attack import Attack
from avsectester.attack.insertion.placement import (AttachedPlacement, WorldPlacement, orientation_from_spec,
                                        placement_from_spec)
from avsectester.attack.segments import validate_segment_options
from avsectester.attack.insertion import Insertion
from avsectester.attack.insertion.compositor import PatchCompositor
from .assets import SignAsset
from .metabuilder import build_metadata
from .online import ObjectInsertionAttackRuntime


class ObjectInsertionAttack(Attack):
    """Composite a placed sign, person poster or traffic-light asset in one camera.

    With ``placement_relative_to_ego_origin``, the configured world placement is interpreted
    once in the first reference ego actor's local frame. That origin is the actor origin
    (normally ground level), not the bounding-box center. Resolved metadata stores world meters.
    """

    def __init__(self, *, decision_steps, placement, assets=None, orientation=None,
                 sensor="front", min_visible_fraction=0.0, max_segments=None, keep_tail=False,
                 placement_relative_to_ego_origin=False, compositor=None):
        validate_segment_options(decision_steps, max_segments, keep_tail)
        self.decision_steps, self.max_segments, self.keep_tail = decision_steps, max_segments, keep_tail
        self.assets = SignAsset() if assets is None else assets
        if not callable(getattr(self.assets, "planes", None)):
            raise TypeError("assets must be a planar asset object configured with type and options")
        self.placement = placement_from_spec(placement)
        if not isinstance(placement_relative_to_ego_origin, bool):
            raise TypeError("placement_relative_to_ego_origin must be boolean")
        if placement_relative_to_ego_origin and not isinstance(self.placement, WorldPlacement):
            raise ValueError("placement_relative_to_ego_origin requires WorldPlacement")
        self.placement_relative_to_ego_origin = placement_relative_to_ego_origin
        self.orientation = orientation_from_spec(orientation)
        self.sensor, self.min_visible_fraction = sensor, float(min_visible_fraction)
        if not isinstance(sensor, str) or not sensor:
            raise ValueError("sensor must be a nonempty declared sensor name")
        if not math.isfinite(self.min_visible_fraction) or not 0 <= self.min_visible_fraction <= 1:
            raise ValueError("min_visible_fraction must be in [0,1]")
        if compositor is not None and not isinstance(compositor, PatchCompositor):
            raise TypeError("compositor must be PatchCompositor")
        self.compositor = compositor

    def insertion(self, artifacts, placement=None):
        """Build a sensor insertion from the selected segment and prepared assets."""
        if artifacts is None:
            raise ValueError("ObjectInsertionAttack requires explicit artifacts")
        return Insertion("object", artifacts, self.placement if placement is None else placement,
                         self.orientation)

    def build_metadata(self, scenario):
        return build_metadata(self, scenario)

    def run_offline(self, context):
        return self.assets

    def selected_placement(self, segment):
        """Rebuild the placement selected by this attack from data-only metadata."""
        if "placement" not in segment.user_data:
            raise ValueError("ObjectInsertionAttack metadata is missing keys: placement. "
                             "Required keys: placement")
        values = dict(segment.user_data["placement"])
        if "kind" not in values:
            raise ValueError("ObjectInsertionAttack placement metadata is missing keys: kind. "
                             "Required keys: kind, host (attached) or position_m (world)")
        kind = values.pop("kind")
        if kind == "attached":
            if "host" not in values:
                raise ValueError("ObjectInsertionAttack attached placement metadata is missing keys: host. "
                                 "Required keys: kind, host")
            return AttachedPlacement(**values)
        if kind == "world":
            if "position_m" not in values:
                raise ValueError("ObjectInsertionAttack world placement metadata is missing keys: position_m. "
                                 "Required keys: kind, position_m")
            return WorldPlacement(**values)
        raise ValueError(f"Unknown selected object placement kind: {kind!r}")

    def create_runtime(self, segment, artifacts):
        return ObjectInsertionAttackRuntime(
            self.sensor, self.insertion(artifacts, self.selected_placement(segment)), self.compositor)


__all__ = [
    'ObjectInsertionAttack',
]
