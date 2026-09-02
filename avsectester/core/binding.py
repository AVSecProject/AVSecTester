"""Plugin ↔ stack attachment compatibility.

A plugin declares the **list of seams it hooks into** (`SecurityPlugin.seams`) plus an
optional set of required capabilities. The framework attaches the plugin at *every* declared
seam the backend exposes; if a declared seam (or a required capability) is missing, the whole
plugin is incompatible with that stack and :func:`check_support` raises
:class:`IncompatiblePlugin` with a readable reason. There is no "pick the best of several
alternatives" — a plugin's seams are the points it operates on, together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .seams import SEAM_ORDER, SEAMS

if TYPE_CHECKING:
    from .capability import Capability, StackProfile


class IncompatiblePlugin(RuntimeError):
    """Raised when a stack cannot support a plugin's declared seams / capabilities."""


def check_support(
    seams: tuple[str, ...], requires: frozenset[Capability], profile: StackProfile
) -> None:
    """Validate that ``profile`` exposes every declared seam and required capability.

    Raises :class:`IncompatiblePlugin` (naming the gap) or ``ValueError`` for an unknown seam.
    """
    if not seams:
        raise IncompatiblePlugin("plugin declares no seams to attach to")
    unknown = [s for s in seams if s not in SEAMS]
    if unknown:
        raise ValueError(f"unknown seam(s) {unknown}; known seams: {sorted(SEAMS)}")
    missing_seams = [s for s in seams if not profile.has_seam(s)]
    missing_caps = sorted(c.value for c in (requires - profile.capabilities))
    if missing_seams or missing_caps:
        have = sorted(profile.seams)
        raise IncompatiblePlugin(
            f"stack cannot support this plugin: missing seams {missing_seams}, "
            f"missing capabilities {missing_caps} (stack exposes seams={have})"
        )


def seams_downstream_of(seam: str, *, inclusive: bool = True) -> frozenset[str]:
    """Seams at or after ``seam`` in pipeline order.

    A defense meant to counter an attack at ``seam`` must hook within this set — a sanitizer
    upstream of the injection point can never see the injected payload.
    """
    if seam not in SEAM_ORDER:
        raise KeyError(f"unknown seam {seam!r}")
    i = SEAM_ORDER.index(seam)
    return frozenset(SEAM_ORDER[i if inclusive else i + 1:])
