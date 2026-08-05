"""Capabilities and the stack profile — what an AV stack exposes to plugins.

Different AV-stack settings (a ground-truth passthrough baseline, a neural detector with a
real LiDAR, an offline replay dataset, a future V2X stack) expose different interception
points and offer different affordances. Rather than hard-code a plugin to one setting, each
backend advertises a :class:`StackProfile` describing which seams exist and which
capabilities are available; a plugin declares the bindings it *could* use, and the framework
resolves the two (:mod:`avsectester.core.binding`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Capability(str, Enum):
    """An affordance a backend may or may not provide to attacks/defenses.

    A plugin's binding lists the capabilities it *requires*; a binding is only eligible on a
    stack whose profile provides all of them. This is what lets one attack pick a raw-LiDAR
    realization on a neural stack and an object-level one on a ground-truth stack.
    """

    GT_PERCEPTION = "gt_perception"          # ground-truth passthrough detector (object-level input)
    NEURAL_PERCEPTION = "neural_perception"  # a real learned detector runs each tick
    RAW_LIDAR = "raw_lidar"                  # a real LiDAR point cloud is available to perturb
    RAW_CAMERA = "raw_camera"                # a real camera image is available to perturb
    GRADIENTS = "gradients"                  # detector exposes differentiable forward (whitebox)
    TRACKER = "tracker"                      # a tracking stage exists downstream of perception
    PLANNER = "planner"                      # a planning stage exists
    CONTROLLER = "controller"                # a control stage exists
    LOCALIZATION = "localization"            # a localization/mapping stage exists
    V2X = "v2x"                              # collaborative-perception / V2X messaging exists


@dataclass(frozen=True)
class StackProfile:
    """What a backend exposes: the set of live seams and the available capabilities.

    ``seams`` names the interception points the backend can actually attach a plugin to
    (a subset of :data:`avsectester.core.seams.SEAMS`). ``capabilities`` names the
    affordances a binding may require.
    """

    seams: frozenset[str] = field(default_factory=frozenset)
    capabilities: frozenset[Capability] = field(default_factory=frozenset)

    def has_seam(self, seam: str) -> bool:
        return seam in self.seams

    def provides(self, capabilities: frozenset[Capability]) -> bool:
        return capabilities <= self.capabilities

    @classmethod
    def of(cls, seams: object, capabilities: object = ()) -> StackProfile:
        """Convenience constructor from any iterables of seam names / capabilities."""
        return cls(frozenset(seams), frozenset(capabilities))  # type: ignore[arg-type]
