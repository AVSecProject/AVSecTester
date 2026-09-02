"""Seams — the named insertion points of an AV pipeline.

A *seam* is a *logical* interception point on a perception -> tracking -> planning ->
control stack (e.g. "the detector's output"). It is deliberately independent of any
concrete avstack module: :mod:`avsectester.hooks` maps a seam onto the actual avstack
pre/post-hook at runtime, and different backends expose different seams
(``Backend.supported_seams()``).

Attacks and defenses never name a raw avstack hook; they declare the list of *seams* they hook
into and the framework attaches them at every declared seam the current stack exposes
(:mod:`avsectester.core.binding`). Keeping the seam vocabulary in ``core`` (rather than in the
avstack-bridge layer) lets the plugin contract depend on it without importing the bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .escalation import Stage


class Phase(str, Enum):
    """Whether a seam intercepts a module's input (pre) or output (post)."""

    PRE = "pre"
    POST = "post"


@dataclass(frozen=True)
class Seam:
    """A named insertion point on an avstack module.

    For a ``PRE`` seam, ``arg_index`` is the positional slot of the payload the plugin
    operates on (a detector's input is ``args[0]``); the adapter splices the returned
    payload back into that slot and preserves everything else. ``POST`` seams operate on
    the module's single return value. ``stage``/``component`` label the trace record.
    """

    name: str
    phase: Phase
    stage: Stage
    component: str
    arg_index: int = 0


# The standard seams of a perception -> tracking -> planning -> control stack.
#
# ``raw_lidar``        — the detector *input* point cloud (mutate raw points).
# ``perception_input`` — the detector input at the object level (a ground-truth passthrough
#                        detector consumes ``ObjectState``s; applied pre-detector).
# ``perception_out``   — the detector output (detection-level, an avstack post-hook).
# ``tracking_out`` / ``planning_out`` / ``control_out`` — each replaces a module's output.
SEAMS: dict[str, Seam] = {
    "raw_lidar": Seam("raw_lidar", Phase.PRE, Stage.SENSOR, "lidar", arg_index=0),
    "perception_input": Seam("perception_input", Phase.PRE, Stage.PERCEPTION, "detector", arg_index=0),
    "perception_out": Seam("perception_out", Phase.POST, Stage.PERCEPTION, "detector"),
    "tracking_out": Seam("tracking_out", Phase.POST, Stage.TRACKING, "tracker"),
    "planning_out": Seam("planning_out", Phase.POST, Stage.PLANNING, "planner"),
    "control_out": Seam("control_out", Phase.POST, Stage.CONTROL, "controller"),
}

# Pipeline order of the seams, upstream -> downstream. Used to check that a defense is
# bound at or downstream of the attack it is meant to counter (:func:`binding.seams_downstream_of`).
SEAM_ORDER: tuple[str, ...] = (
    "raw_lidar",
    "perception_input",
    "perception_out",
    "tracking_out",
    "planning_out",
    "control_out",
)


def resolve_seam(seam: str | Seam) -> Seam:
    """Return the :class:`Seam` for ``seam`` (a name or an already-resolved seam)."""
    if isinstance(seam, Seam):
        return seam
    if seam not in SEAMS:
        raise KeyError(f"unknown seam {seam!r}; known seams: {sorted(SEAMS)}")
    return SEAMS[seam]
