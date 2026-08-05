"""Attack vectors — the shared *mechanism* a family of attack methods is built on.

An attack **vector** is the delivery channel and the toolkit that goes with it (e.g. LiDAR
spoofing: how to add or remove points/objects/detections in the ego's view, plus the
geometry and the feasibility constraints). An attack **method** is a concrete goal realized
through a vector — object *spoofing* (a false positive) and object *removal* (a false
negative) are two methods that share the one LiDAR-spoofing vector.

Categorizing attacks by vector (rather than by seam or by ad-hoc class) means methods that
share a delivery channel share code *and* a single, consistent set of
:class:`~avsectester.core.binding.BindingSpec` — the seams the vector can operate at. A
method composes a vector, declares ``bindings = <Vector>.bindings``, and dispatches on the
resolved seam to the vector primitive that fits that seam's payload.
"""

from __future__ import annotations

from ..core.binding import BindingSpec


class AttackVector:
    """Base class for a shared attack mechanism.

    Subclasses set :attr:`bindings` (the seams this vector can realize an attack at, ranked
    by fidelity) and provide primitive operations (add/remove/perturb) keyed to those seams'
    payloads. Vectors are intended to be **stateless toolkits** — per-run state (caches,
    counters) lives on the attack method that composes the vector, so one vector instance is
    safe to share.
    """

    #: Seams this vector can operate at, ranked by fidelity. Methods inherit this verbatim.
    bindings: tuple[BindingSpec, ...] = ()

    @property
    def name(self) -> str:
        return type(self).__name__
