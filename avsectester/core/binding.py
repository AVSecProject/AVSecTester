"""Bindings — how a plugin's intent is realized at a concrete seam on a given stack.

An attack/defense is not "at seam X". It is a capability that can be realized at one of
several seams, each with a different fidelity and different requirements. A plugin declares
a tuple of :class:`BindingSpec`; :func:`resolve` matches them against a backend's
:class:`~avsectester.core.capability.StackProfile` and returns the highest-fidelity binding
the stack can support — or raises :class:`IncompatiblePlugin` with a readable reason.

Example — a phantom-obstacle attack that works across stacks::

    bindings = (
        BindingSpec("raw_lidar",      "points",     requires={Capability.NEURAL_PERCEPTION,
                                                              Capability.RAW_LIDAR}, fidelity=3),
        BindingSpec("perception_out", "detections", requires={Capability.NEURAL_PERCEPTION}, fidelity=2),
        BindingSpec("perception_input","objects",   requires={Capability.GT_PERCEPTION},     fidelity=1),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .capability import Capability
from .seams import SEAM_ORDER, SEAMS

if TYPE_CHECKING:
    from .capability import StackProfile


class IncompatiblePlugin(RuntimeError):
    """Raised when no declared binding is supported by the target stack."""


@dataclass(frozen=True)
class BindingSpec:
    """One way a plugin can attach itself: a seam, the payload it expects there, the
    capabilities that must be present, and a fidelity rank (higher = preferred / more
    realistic). ``payload`` is informational — it documents what ``apply`` receives at this
    seam and lets a plugin dispatch to a per-seam handler.
    """

    seam: str
    payload: str = ""
    requires: frozenset[Capability] = field(default_factory=frozenset)
    fidelity: int = 0

    def __post_init__(self) -> None:
        if self.seam not in SEAMS:
            raise ValueError(f"unknown seam {self.seam!r}; known seams: {sorted(SEAMS)}")
        # normalise requires to a frozenset so equality/subset checks are cheap
        object.__setattr__(self, "requires", frozenset(self.requires))

    def supported_by(self, profile: StackProfile) -> bool:
        return profile.has_seam(self.seam) and profile.provides(self.requires)


def resolve(bindings: tuple[BindingSpec, ...], profile: StackProfile) -> BindingSpec:
    """Pick the best binding the ``profile`` supports.

    Highest ``fidelity`` wins; ties break toward the more upstream seam (a more realistic
    interception point). Raises :class:`IncompatiblePlugin` if nothing matches, naming the
    gap so the failure is actionable.
    """
    if not bindings:
        raise IncompatiblePlugin("plugin declares no bindings")
    candidates = [b for b in bindings if b.supported_by(profile)]
    if not candidates:
        raise IncompatiblePlugin(_why_none(bindings, profile))
    return max(candidates, key=lambda b: (b.fidelity, -SEAM_ORDER.index(b.seam)))


def _why_none(bindings: tuple[BindingSpec, ...], profile: StackProfile) -> str:
    lines = ["no declared binding is supported by this stack:"]
    for b in bindings:
        if not profile.has_seam(b.seam):
            lines.append(f"  - {b.seam!r}: seam not exposed by the backend")
        else:
            missing = sorted(c.value for c in (b.requires - profile.capabilities))
            lines.append(f"  - {b.seam!r}: missing capabilities {missing}")
    have = sorted(profile.seams)
    caps = sorted(c.value for c in profile.capabilities)
    lines.append(f"  stack exposes seams={have}, capabilities={caps}")
    return "\n".join(lines)


def seams_downstream_of(seam: str, *, inclusive: bool = True) -> frozenset[str]:
    """Seams at or after ``seam`` in pipeline order.

    A defense meant to counter an attack at ``seam`` must be bound within this set — a
    sanitizer upstream of the injection point can never see the injected payload.
    """
    if seam not in SEAM_ORDER:
        raise KeyError(f"unknown seam {seam!r}")
    i = SEAM_ORDER.index(seam)
    return frozenset(SEAM_ORDER[i if inclusive else i + 1:])
