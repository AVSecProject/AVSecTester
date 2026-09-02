"""The security-plugin contract shared by attacks and defenses.

Every attack/defense is a :class:`SecurityPlugin`: a hook-shaped component
(``apply(payload, *, ego_state, ctx) -> payload``) that declares the **list of seams it hooks
into** (:attr:`seams`). The framework attaches it at every declared seam the backend exposes;
each attachment calls :meth:`apply` with ``ctx.seam`` set to the seam currently firing, so one
plugin can act at several seams and dispatch on which one it is. How a plugin uses its seams is
entirely up to the subclass — the base only fixes the attachment contract. If a declared seam
isn't exposed by the stack, :meth:`check` fails loudly.

Lifecycle hooks (:meth:`reset`, :meth:`validate`) default to no-ops; :meth:`describe` returns
inventory metadata.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .binding import check_support

if TYPE_CHECKING:
    from .experiment import ExperimentSpec


class SecurityPlugin(ABC):
    """Base class for attacks and defenses.

    Subclasses set :attr:`seams` (the seams they hook into) and implement :meth:`apply`. To act
    differently at different seams, branch on :meth:`current_seam` (``ctx.seam``).
    """

    #: Human/inventory category tag (e.g. "sensor", "perception", "input_sanitize").
    category: str = "generic"
    #: The seams this plugin hooks into. Attached at every one the backend exposes.
    seams: tuple[str, ...] = ()

    # -- identity / inventory -------------------------------------------------
    @property
    def name(self) -> str:
        return type(self).__name__

    def describe(self) -> dict[str, Any]:
        """Registry/inventory metadata for this plugin (extended by subclasses)."""
        return {
            "name": self.name,
            "kind": self._kind(),
            "category": self.category,
            "seams": list(self.seams),
        }

    def _kind(self) -> str:
        return "plugin"

    # -- attachment -----------------------------------------------------------
    def check(self, exposed: frozenset[str]) -> None:
        """Raise :class:`~avsectester.core.binding.IncompatiblePlugin` if the stack does not
        expose every declared seam. ``exposed`` = ``Backend.supported_seams()``."""
        check_support(self.seams, exposed)

    def current_seam(self, ctx: Any) -> str | None:
        """The seam currently firing (``ctx.seam``), or the sole declared seam for a direct
        call with no context. Subclasses dispatch on this."""
        if ctx is not None and getattr(ctx, "seam", ""):
            return ctx.seam
        return self.seams[0] if self.seams else None

    # -- lifecycle (default no-ops) -------------------------------------------
    def validate(self, spec: ExperimentSpec) -> None:
        """Raise if the experiment would violate this plugin's assumptions (optional)."""

    def reset(self) -> None:
        """Clear per-run state between passes (optional)."""

    # -- the hook -------------------------------------------------------------
    @abstractmethod
    def apply(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        """Transform module I/O at the firing seam (hook contract). Return the payload."""

    def __call__(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        return self.apply(data, *args, **kwargs)
