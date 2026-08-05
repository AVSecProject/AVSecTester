"""The security-plugin contract shared by attacks and defenses.

Every attack/defense is a :class:`SecurityPlugin`: a hook-shaped component
(``apply(payload, *, ego_state, ctx) -> payload``) that declares *how it can bind* to a
stack (a tuple of :class:`~avsectester.core.binding.BindingSpec`) rather than a single fixed
seam. The framework resolves those bindings against the backend's
:class:`~avsectester.core.capability.StackProfile` (:meth:`resolve_binding`) and attaches the
plugin at whatever concrete avstack hook the resolved seam maps to.

Lifecycle hooks (:meth:`setup`, :meth:`reset`, :meth:`validate`, :meth:`teardown`) all
default to no-ops so a concrete plugin overrides only what it needs. :meth:`describe`
returns registry/inventory metadata (seeds the vulnerability database + leaderboards).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .binding import BindingSpec, resolve

if TYPE_CHECKING:
    from .capability import StackProfile
    from .experiment import ExperimentSpec


class SecurityPlugin(ABC):
    """Base class for attacks and defenses.

    Subclasses set :attr:`bindings` (one or more :class:`BindingSpec`, ranked by fidelity)
    and implement :meth:`apply`. When a plugin can bind at more than one seam, dispatch on
    the resolved binding via :meth:`bound_seam` / a per-seam handler.
    """

    #: Human/inventory category tag (e.g. "sensor", "perception", "input_sanitize").
    category: str = "generic"
    #: Ranked ways this plugin can attach; the framework resolves against the stack profile.
    bindings: tuple[BindingSpec, ...] = ()
    #: The binding chosen by the last :meth:`resolve_binding` (class-level default so a
    #: subclass need not call ``super().__init__()``).
    _binding: BindingSpec | None = None

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
            "bindings": [
                {
                    "seam": b.seam,
                    "payload": b.payload,
                    "fidelity": b.fidelity,
                    "requires": sorted(c.value for c in b.requires),
                }
                for b in self.bindings
            ],
        }

    def _kind(self) -> str:
        return "plugin"

    # -- binding resolution ---------------------------------------------------
    def resolve_binding(self, profile: StackProfile) -> BindingSpec:
        """Pick and remember the binding to use on a stack with this ``profile``.

        Raises :class:`~avsectester.core.binding.IncompatiblePlugin` if unsupported.
        """
        self._binding = resolve(self.bindings, profile)
        return self._binding

    @property
    def binding(self) -> BindingSpec | None:
        """The binding chosen by the last :meth:`resolve_binding` (``None`` until resolved)."""
        return self._binding

    @property
    def bound_seam(self) -> str | None:
        """The seam name of the resolved binding, or ``None`` if not yet resolved."""
        return self._binding.seam if self._binding else None

    @property
    def primary_binding(self) -> BindingSpec | None:
        """The highest-fidelity declared binding (the plugin's preferred realization)."""
        return max(self.bindings, key=lambda b: b.fidelity) if self.bindings else None

    @property
    def seam(self) -> str | None:
        """Convenience: the resolved seam if bound, else the primary binding's seam.

        Lets callers read a plugin's default seam without a stack profile (e.g. scripts that
        attach directly); the engine still resolves against the live profile before attaching.
        """
        if self._binding is not None:
            return self._binding.seam
        pb = self.primary_binding
        return pb.seam if pb else None

    # -- lifecycle (default no-ops) -------------------------------------------
    def setup(self, spec: ExperimentSpec) -> None:
        """Prepare before a run (optional)."""

    def validate(self, spec: ExperimentSpec) -> None:
        """Raise if the experiment would violate this plugin's assumptions (optional)."""

    def reset(self) -> None:
        """Clear per-run state between passes (optional)."""

    def teardown(self) -> None:
        """Release resources after a run (optional)."""

    # -- the hook -------------------------------------------------------------
    @abstractmethod
    def apply(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        """Transform module I/O at the bound seam (hook contract). Return the payload."""

    def __call__(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        return self.apply(data, *args, **kwargs)
