"""The driving-system interface: an Observation in, a Control out.

``AVStack`` knows nothing about how the system works: a modular perception->tracking->planning
->control pipeline and a single end-to-end model both satisfy it. Implementations live in
:mod:`avsectester.closed_loop.avstacks`. This module imports nothing from avstack, avcarla or
carla; the implementations carry those dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from avsectester.closed_loop.datatypes import Control, Detection, Observation
from avsectester.closed_loop.hooks import _StageAdapter

if TYPE_CHECKING:
    from avsectester.closed_loop.hooks import HookSet
    from avsectester.closed_loop.simulator import SimulatorRuntime
    from avsectester.closed_loop.vehicle_io import VehicleIO


class AVStack(_StageAdapter, ABC):
    """The AV 'box': an Observation in, a Control out.

    The interface knows nothing about how the box works — a modular perception->tracking->planning
    ->control pipeline and a single end-to-end model both satisfy it. Implementations adapt their
    own machinery (e.g. an avstack ``ModularDrivingPipeline``, or a camera end-to-end policy).
    """

    provides_detections: bool | None = None

    @property
    def vehicle_io(self) -> VehicleIO:
        """Driving-system input processing, including its localizer lifecycle."""
        from avsectester.closed_loop.vehicle_io import VehicleIO

        if "_vehicle_io" not in self.__dict__:
            self._vehicle_io = VehicleIO()
        return self._vehicle_io

    @vehicle_io.setter
    def vehicle_io(self, value: VehicleIO) -> None:
        from avsectester.closed_loop.vehicle_io import VehicleIO

        if not isinstance(value, VehicleIO):
            raise TypeError("vehicle_io must be VehicleIO")
        self._vehicle_io = value

    def reset(self, observation: Observation) -> None:
        """Optional warmup from the first Observation (e.g. seed a planner). Default no-op."""

    @property
    def command_modes(self) -> frozenset[str] | None:
        """Possible output command representations, or None for an undeclared adapter."""
        return None

    @property
    def models(self) -> Mapping[str, Any]:
        """Loaded native models available to user handlers. Adapters expose their own names."""
        return {}

    def component_log(self) -> Mapping[str, Any] | None:
        """Already-produced component outputs for the current decision, or None.

        Recording must never run extra inference to obtain outputs. Implementations may
        enable their own instrumentation before the first call.
        """
        return None

    def detections(self) -> Sequence[Detection] | None:
        """Current decision's ego-frame detections; None means no detection output.

        An empty sequence means a detector ran and found no objects. Calling this method
        must not run inference or modify the driving system's state.
        """
        return None

    def close(self) -> None:
        """Release driving-system resources. The default implementation owns none."""

    @abstractmethod
    def __call__(self, observation: Observation) -> Control:
        """Decide the control command for this Observation."""


class CommandCompatibilityError(ValueError):
    """Declared AVStack commands cannot be consumed by the selected simulator."""


def validate_command_compatibility(
    simulator_runtime: SimulatorRuntime, avstack: AVStack, hooks: HookSet | None = None,
) -> None:
    """Check declared commands before world setup, allowing explicit hook conversion.

    A command-stage handler may translate the AVStack output into a different representation.
    In that case only the final command is checked, after all handlers have run.
    """
    accepted, produced = simulator_runtime.command_modes, avstack.command_modes
    if accepted is None or produced is None:
        return
    if hooks is not None and any(hook.stage == "command" for hook in hooks.hooks):
        return
    missing = produced - accepted
    if missing:
        raise CommandCompatibilityError(
            f"{type(avstack).__name__} produces {sorted(missing)} commands unsupported by "
            f"{type(simulator_runtime).__name__}. Select compatible dynamics or add a command conversion hook."
        )
