"""The online phase: one AttackRuntime per run and the attack actions it returns.

Actions are classified by where they act in the closed loop:

* ``WorldAttackAction`` changes the simulated world. The simulator applies it at setup or
  before a world step, and it is visible from the next capture.
* ``SensorAttackAction`` changes one captured sensor's data before ObservationPipeline, using the
  simulator's sensor view. It applies to the current decision.
* ``AVStackAttackAction`` changes a driving-system value from ``localization.pre`` through
  ``command`` in the current decision.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from avsectester.data_types import Observation, WorldSnapshot

if TYPE_CHECKING:
    from avsectester.closed_loop.simulators.base import SensorView, SimulatorRuntime


@dataclass(frozen=True)
class ResultSubscription:
    """Declare a closed-loop result read by an attack runtime."""

    kind: Literal["observation", "physical_state"]

    def __post_init__(self) -> None:
        if self.kind not in {"observation", "physical_state"}:
            raise ValueError(f"Unknown attack result kind: {self.kind!r}")


@dataclass(frozen=True)
class OnlineContext:
    """Independent snapshots of subscribed results, with a read-only physical clock.

    The framework creates it for every online-phase callback. It deliberately carries no
    simulator, driving system, native SDK or time-advancing service. Nested data can be
    modified locally without changing another callback.
    """

    step: int
    stage: str
    sim_time: float
    frame: int
    observation: Observation | None = None
    physical_state: WorldSnapshot | None = None
    _output_recorder: Callable[[OnlineContext, str, Any], None] | None = field(
        default=None, repr=False, compare=False,
    )

    def record_output(self, name: str, value: Any) -> None:
        """Save a named copy of ``value``, such as an attack decision, in this run's log."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Output name must be a nonempty string")
        if self._output_recorder is None:
            raise RuntimeError("Outputs can only be recorded during a run")
        self._output_recorder(self, name, value)


class AttackAction:
    """Base of everything the online phase applies. Subclasses state where it acts."""

    def receipt_details(self) -> Mapping[str, Any]:
        """Data-only evidence from the most recent application, included in its receipt."""
        return {}


class WorldAttackAction(AttackAction, ABC):
    """Change the simulated world. The simulator applies it. It is visible from the next capture.

    ``setup_ticks`` is the number of ``simulator.setup_tick()`` calls that ``apply`` makes when
    the action is returned by ``AttackRuntime.setup``. Ticks are unavailable after setup.
    Register the removal of created world objects on ``resources``; it runs when the run ends.
    """

    setup_ticks: int = 0
    'Simulator setup ticks requested by this participant before decisions begin.'

    @abstractmethod
    def apply(self, simulator_runtime: SimulatorRuntime, resources: ExitStack) -> None:
        """Change the world through the simulator's native access."""


class SensorAttackAction(AttackAction, ABC):
    """Change one captured sensor's data before ObservationPipeline, in the current decision.

    ``sensor`` names the simulator sensor. ``apply`` receives the current data, which earlier
    actions on the same sensor may have changed, and the capture's view, whose ``data`` is
    the unchanged capture. It returns the changed data in the same representation.
    """

    sensor: str
    'Name of the sensor whose captured data this action modifies.'

    @abstractmethod
    def apply(self, data: Any, view: SensorView) -> Any:
        """Return the changed sensor data."""


@dataclass(frozen=True)
class AVStackAttackAction(AttackAction):
    """Change the value at a driving-system stage in the current decision.

    ``apply(value, context)`` returns the value used downstream. It may be a bound method of
    the runtime, so a decision that depends on a component output keeps its state there.
    """

    stage: str
    apply: Callable[[Any, OnlineContext], Any]

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("AVStackAttackAction.stage must be a nonempty string")
        if not callable(self.apply):
            raise TypeError("AVStackAttackAction.apply must be callable")


class AttackRuntime:
    """One online-phase instance per run. Every callback shares this object's state.

    ``setup`` runs after the simulator reset and before the first capture, and returns only
    world attack actions. ``step`` runs once per decision before ObservationPipeline. ``stop`` runs
    at the end of the run, also after failures and with zero decisions. ``setup_ticks`` is the
    simulator tick budget used by the world actions that ``setup`` returns.
    """

    subscriptions: Sequence[ResultSubscription] = ()
    'Result subscriptions delivered to the runtime during execution.'
    setup_ticks: int = 0
    'Simulator setup ticks requested by this participant before decisions begin.'

    def setup(self, context: OnlineContext) -> Sequence[WorldAttackAction]:
        """Initialize online-phase state and return actions applied before the first decision."""
        return ()

    def step(self, context: OnlineContext) -> Sequence[AttackAction]:
        """Return attack actions for the current decision."""
        return ()

    def stop(self, context: OnlineContext) -> None:
        """Release online-phase resources. The default implementation has no effect."""
        pass


__all__ = [
    'ResultSubscription',
    'OnlineContext',
    'AttackAction',
    'WorldAttackAction',
    'SensorAttackAction',
    'AVStackAttackAction',
    'AttackRuntime',
]
