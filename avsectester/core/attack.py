"""The attack and defense contracts.

Every attack has two parts:

- **Offline** ``prepare(data) -> artifact``: consume a stream of :class:`Frame`\\ s (a common
  AD data format) and return a serializable, attack-specific artifact (an adversarial patch, a
  point set, tuned parameters). ``load`` adopts a prepared artifact at runtime. Attacks that
  need no optimization skip ``prepare``.
- **Runtime** ``apply(payload, ctx)``: the *hook* — inject/modify the payload at the seam that
  is firing (``ctx.seam``), using the loaded artifact. The framework attaches ``apply`` at
  every seam in :attr:`seams`; there is no separate hook object.

A :class:`Defense` is the same shape without the offline half.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from .context import Context
from .frame import Frame
from .seam import Seam


class Attack(ABC):
    #: the seams this attack injects at (attached at every one the system exposes)
    seams: tuple[Seam, ...] = ()
    #: optional adversary specification (see core.threat_model.ThreatModel)
    threat_model: Any = None
    #: the prepared artifact, adopted via load(); None until then
    artifact: Any = None

    # -- offline --------------------------------------------------------------
    def prepare(self, data: Iterable[Frame]) -> Any:
        """Offline: consume data, return the attack artifact. Default: no artifact."""
        return None

    def load(self, artifact: Any) -> None:
        """Runtime: adopt a prepared artifact."""
        self.artifact = artifact

    # -- runtime --------------------------------------------------------------
    def reset(self) -> None:
        """Clear per-run state between passes (optional)."""

    @abstractmethod
    def apply(self, payload: Any, ctx: Context) -> Any:
        """Inject/modify ``payload`` at ``ctx.seam``; return the payload."""


class Defense(ABC):
    """A defense/mitigation — the runtime half of the attack shape (no artifact)."""

    seams: tuple[Seam, ...] = ()

    def reset(self) -> None:
        """Clear per-run state between passes (optional)."""

    @abstractmethod
    def apply(self, payload: Any, ctx: Context) -> Any:
        """Sanitize/mitigate ``payload`` at ``ctx.seam``; return the payload."""
