"""System — the AV pipeline under test, with attacks/defenses fired at its seams.

A ``System`` consumes a :class:`Frame`, runs the AV stack (perception -> ... -> control),
and returns an :class:`Outcome` (the control to feed back + a per-frame record for metrics).
Attacks/defenses are attached at seams; the system calls their ``apply`` directly at the
matching point in its pipeline (:meth:`fire`) — so the plugin's ``apply`` *is* the hook.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .context import Context
from .frame import Frame
from .seam import Seam


@dataclass
class Outcome:
    control: Any = None                                   # fed back to Environment.step
    record: dict[str, Any] = field(default_factory=dict)  # per-frame metrics record


class System(ABC):
    #: the seams this system exposes (a subset of Seam)
    seams: tuple[Seam, ...] = ()

    def __init__(self) -> None:
        self._attached: dict[Seam, list[Any]] = {}

    def attach(self, plugin: Any, seam: Seam | str | None = None) -> None:
        """Attach ``plugin`` at ``seam`` (or, if None, at each seam in ``plugin.seams``)."""
        targets = (Seam(seam),) if seam is not None else tuple(Seam(s) for s in plugin.seams)
        for s in targets:
            if s not in self.seams:
                exposed = [x.value for x in self.seams]
                raise ValueError(
                    f"{type(self).__name__} does not expose seam {s.value!r} (exposes {exposed})"
                )
            self._attached.setdefault(s, []).append(plugin)

    def fire(self, seam: Seam, payload: Any, frame: Frame) -> Any:
        """Run every plugin attached at ``seam`` over ``payload`` and return the result."""
        seam = Seam(seam)
        for plugin in self._attached.get(seam, ()):
            payload = plugin.apply(payload, Context(frame, seam))
        return payload

    @abstractmethod
    def process(self, frame: Frame) -> Outcome:
        """Run the AV pipeline on ``frame``; return control + record."""

    def close(self) -> None:
        """Release resources (optional)."""
