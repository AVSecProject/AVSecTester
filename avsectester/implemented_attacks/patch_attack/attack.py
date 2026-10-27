"""Connect patch selection, optional optimization and online deployment."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from avsectester.validation import require_positive_integer
from avsectester.attack.attack import Attack
from avsectester.attack.insertion.placement import (AttachedPlacement, Orientation, orientation_from_spec,
                                        validate_attached_mount)
from avsectester.attack.segments import validate_segment_options
from avsectester.attack.insertion import Insertion, PlaneAsset
from avsectester.rendering.textures import CheckerboardTexture, Texture
from avsectester.rendering.textures import load_rgba_image
from avsectester.attack.insertion.compositor import PatchCompositor
from .metabuilder import build_metadata
from .offline import PatchOptimization, run_offline
from .online import DEFAULT_PROP, PatchAttackRuntime


class PatchAttack(Attack):
    """Select the best visible hosts per time window, then deploy one patch artifact.

    ``candidate_hosts=None`` considers all non-ego actors. ``hosts_per_window`` keeps
    the highest-scoring hosts independently in each window; ties retain candidate order.
    ``max_windows`` limits the number of time windows, not the number of hosts.
    ``anchor`` and ``offset_m`` specify a box-local mounting point. A selected
    host is assigned only when constructing its actual ``AttachedPlacement``. CARLA native
    mesh dimensions must match the declared plane dimensions.
    """

    def __init__(self, *, decision_steps, assets: np.ndarray | str | Path | Texture | None = None,
                 candidate_hosts=None, hosts_per_window=1,
                 sensor="front", width_m=1.0, height_m=0.5, anchor="rear_center", offset_m=(-0.01, 0.0, 0.0),
                 orientation=None,
                 max_distance=None, min_visible_fraction=0.0, max_windows=None, keep_tail=False,
                 optimization=None, prop=DEFAULT_PROP, emissive=True, compositor=None):
        validate_segment_options(decision_steps, None, keep_tail)
        if max_windows is not None:
            require_positive_integer(max_windows, "max_windows")
        if candidate_hosts is not None:
            if isinstance(candidate_hosts, (str, bytes)):
                raise TypeError("candidate_hosts must be a sequence of stable actor IDs or None")
            candidate_hosts = tuple(candidate_hosts)
            if (not candidate_hosts or any(not isinstance(host, str) or not host for host in candidate_hosts)
                    or len(set(candidate_hosts)) != len(candidate_hosts)):
                raise ValueError("candidate_hosts must contain unique nonempty stable actor IDs")
        require_positive_integer(hosts_per_window, "hosts_per_window")
        self.decision_steps, self.max_windows, self.keep_tail = decision_steps, max_windows, keep_tail
        self.candidate_hosts, self.hosts_per_window = candidate_hosts, hosts_per_window
        assets = CheckerboardTexture() if assets is None else assets
        rgba = load_rgba_image(assets)
        self.sensor, self.width_m, self.height_m = sensor, float(width_m), float(height_m)
        if not isinstance(sensor, str) or not sensor:
            raise ValueError("sensor must be a nonempty declared sensor name")
        self.anchor = anchor
        self.offset_m = validate_attached_mount(self.anchor, offset_m)
        self.orientation = (orientation_from_spec(orientation) if orientation is not None else
                            Orientation("follow_host", (0.0, 0.0, 180.0)))
        self.max_distance = None if max_distance is None else float(max_distance)
        self.min_visible_fraction = float(min_visible_fraction)
        if self.max_distance is not None and (not math.isfinite(self.max_distance) or self.max_distance <= 0):
            raise ValueError("max_distance must be finite and positive or None")
        if not math.isfinite(self.min_visible_fraction) or not 0 <= self.min_visible_fraction <= 1:
            raise ValueError("min_visible_fraction must be in [0,1]")
        if optimization is not None and not isinstance(optimization, PatchOptimization):
            raise TypeError("optimization must be PatchOptimization")
        if not isinstance(prop, str) or not prop:
            raise ValueError("prop must be a nonempty CARLA blueprint name")
        if not isinstance(emissive, bool):
            raise TypeError("emissive must be boolean")
        if compositor is not None and not isinstance(compositor, PatchCompositor):
            raise TypeError("compositor must be PatchCompositor")
        self.optimization, self.prop, self.emissive, self.compositor = optimization, prop, emissive, compositor
        self.assets = PlaneAsset(rgba, self.width_m, self.height_m).texture

    def placement_for(self, host):
        """Resolve patch placement on the selected host actor."""
        return AttachedPlacement(host, self.anchor, self.offset_m)

    def selected_placement(self, segment):
        """Rebuild this attack's host-local placement from its data-only metadata."""
        required = ("host", "placement")
        missing = [key for key in required if key not in segment.user_data]
        if missing:
            raise ValueError("PatchAttack metadata is missing keys: " + ", ".join(missing)
                             + ". Required keys: " + ", ".join(required))
        return AttachedPlacement(segment.user_data["host"], **segment.user_data["placement"])

    def insertion(self, artifacts, *, host=None, placement=None):
        """Build a patch insertion using the selected host and prepared texture."""
        if artifacts is None:
            raise ValueError("PatchAttack requires explicit artifacts")
        if placement is None:
            if host is None:
                raise ValueError("Patch insertion requires an explicit host or selected placement")
            placement = self.placement_for(host)
        return Insertion("patch", PlaneAsset(artifacts, self.width_m, self.height_m),
                         placement, self.orientation)

    def build_metadata(self, scenario):
        return build_metadata(self, scenario)

    def run_offline(self, context):
        return run_offline(self, context)

    def deployment_ticks(self, segment):
        return 1 if segment.user_data.get("deployment") == "world" else 0

    def create_runtime(self, segment, artifacts):
        if artifacts is None:
            raise ValueError("PatchAttack requires explicit artifacts")
        self.selected_placement(segment)
        physical = segment.user_data.get("deployment") == "world"
        if physical and self.orientation.mode != "follow_host":
            raise ValueError("CARLA physical patches require follow_host orientation")
        return PatchAttackRuntime(self, artifacts, physical, segment)


__all__ = [
    'PatchAttack',
]
