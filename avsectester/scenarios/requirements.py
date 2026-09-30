"""Per-attack scenario requirements — each attack declares the scene it needs, once.

A new attack adds one entry to :data:`REQUIREMENTS` (or its own module exports a ``ScenarioRequirement``);
the providers (``source.py``) and the eval harness are unchanged. Keeping the requirement next to the
attack's *assumptions* — not its mechanism — is the point: the same object drives both real-data
filtering and CARLA construction.
"""

from __future__ import annotations

from avsectester.scenarios.requirement import (
    DistanceRange,
    ImageAreaFrac,
    InView,
    MinVisibility,
    ScenarioRequirement,
    TargetSpec,
    ViewpointRear,
)

#: Physical-patch attack that removes a vehicle detection: needs a target vehicle visible in the front
#: camera, close + rear-facing + mostly unoccluded so a patch can be placed on its rear surface, and not
#: so large it is a near-full-frame false positive.
PHYSICAL_PATCH_HIDE_VEHICLE = ScenarioRequirement(
    name="physical_patch_hide_vehicle",
    target=TargetSpec(category="vehicle", camera="front", select="nearest_ahead"),
    constraints=[
        InView("front"),
        DistanceRange(4.0, 25.0),
        ImageAreaFrac(0.02, 0.5, camera="front"),
        ViewpointRear(max_deg=35.0),
        MinVisibility(0.7),
    ],
)

#: Registry keyed by attack name. The eval harness looks a requirement up here.
REQUIREMENTS: dict[str, ScenarioRequirement] = {
    PHYSICAL_PATCH_HIDE_VEHICLE.name: PHYSICAL_PATCH_HIDE_VEHICLE,
}
