"""The two interfaces the data plane connects — and the loop that drives them.

    WorldBackend : reset() / step(control) -> Observation      (CARLA, AlpaSim, dataset replay)
    AVStack      : (observation) -> Control                     (modular pipeline OR end-to-end model)

Both are deliberately ignorant of each other's internals. The backend owns world state and the
shared vehicle dynamics; the AV box owns the driving decision. Neither knows whether the other is
CARLA vs a neural-reconstruction world model, or a modular perception->planning->control pipeline
vs a single end-to-end model. Like :mod:`avsectester.plane`, this module imports nothing from
avstack/avcarla/carla — the backend and stack *implementations* carry those dependencies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from .plane import Control, FrameRecord, Observation, Trace


class WorldBackend(ABC):
    """A world that renders sensor data and applies shared physical control.

    Owns the world state and the ego dynamics; the AV stack never sees world state, only the
    :class:`~avsectester.plane.Observation`. ``step`` applies a :class:`~avsectester.plane.Control`
    through the shared physics, advances the world + traffic, renders the next sensors, and returns
    the next Observation. The causal invariant (so the loop serializes over gRPC): a control at
    frame *t* affects frame *t+1*, never the already-emitted sensors of frame *t*.
    """

    @abstractmethod
    def reset(self) -> Observation:
        """Build the world, spawn the ego + traffic, and return the first Observation."""

    @abstractmethod
    def step(self, control: Control) -> Observation:
        """Apply ``control`` via the shared dynamics, advance one frame, return the Observation."""

    def close(self) -> None:
        """Release simulator resources. Optional."""


class AVStack(ABC):
    """The AV 'box': an Observation in, a Control out.

    The interface knows nothing about how the box works — a modular perception->tracking->planning
    ->control pipeline and a single end-to-end model both satisfy it. Implementations adapt their
    own machinery (e.g. an avstack ``ModularDrivingPipeline``, or a camera end-to-end policy).
    """

    def reset(self, observation: Observation) -> None:
        """Optional warmup from the first Observation (e.g. seed a planner). Default no-op."""

    @abstractmethod
    def __call__(self, observation: Observation) -> Control:
        """Decide the control command for this Observation."""


def run(
    backend: WorldBackend,
    stack: AVStack,
    frames: int,
    perturb: Callable[[Observation], Observation] | None = None,
) -> Trace:
    """Drive ``stack`` in ``backend`` for ``frames`` steps and return the driving :class:`Trace`.

    ``perturb`` (optional) transforms each Observation before the box sees it — this is the single
    attack seam, and it is not part of the sim<->stack contract. The Trace always records the
    backend's **true** ego state, never the perturbed observation, so scoring is honest even when the
    box is being fed a spoofed view.
    """
    obs = backend.reset()
    stack.reset(obs)
    trace = Trace()
    for i in range(frames):
        seen = perturb(obs) if perturb is not None else obs
        control = stack(seen)
        obs = backend.step(control)
        trace.records.append(
            FrameRecord(
                frame=i,
                t=obs.t,
                speed=obs.ego_speed,
                throttle=control.throttle,
                brake=control.brake,
                steer=control.steer,
            )
        )
    return trace
