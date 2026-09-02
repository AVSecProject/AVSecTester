"""Plugin ↔ stack attachment compatibility.

A plugin declares the **list of seams it hooks into** (`SecurityPlugin.seams`). A backend
advertises the **set of seams it exposes** (`Backend.supported_seams()`). The framework
attaches the plugin at every declared seam the backend exposes; if a declared seam is not
exposed (or is not a known seam at all), the plugin is incompatible with that stack and
:func:`check_support` raises :class:`IncompatiblePlugin`. There is no "pick the best of several
alternatives" — a plugin's seams are the points it operates on, together.
"""

from __future__ import annotations

from .seams import SEAM_ORDER, SEAMS


class IncompatiblePlugin(RuntimeError):
    """Raised when a stack does not expose every seam a plugin declares."""


def check_support(seams: tuple[str, ...], exposed: frozenset[str]) -> None:
    """Validate that ``exposed`` contains every declared seam.

    Raises :class:`IncompatiblePlugin` (naming the gap) or ``ValueError`` for an unknown seam.
    """
    if not seams:
        raise IncompatiblePlugin("plugin declares no seams to attach to")
    unknown = [s for s in seams if s not in SEAMS]
    if unknown:
        raise ValueError(f"unknown seam(s) {unknown}; known seams: {sorted(SEAMS)}")
    missing = [s for s in seams if s not in exposed]
    if missing:
        raise IncompatiblePlugin(
            f"stack does not expose seam(s) {missing} (it exposes {sorted(exposed)})"
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
