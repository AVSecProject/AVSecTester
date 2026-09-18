"""ModularAVStack — an AVStack backed by an avstack ``ModularDrivingPipeline``.

The modular (white-box) AV stack: perception -> tracking -> planning -> control, built from config.
Heavy avstack imports are lazy (done when the pipeline is built), so this module — and the
``avsectester.stacks`` package — imports without avstack, matching :class:`AlpamayoAVStack`.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from avsectester.backend import AVStack
from avsectester.plane import Control, Observation


class _DetectionCounter:
    """A trivial avstack post-hook that records how many detections perception emitted."""

    def __init__(self) -> None:
        self.last = 0

    def __call__(self, detections: Any) -> tuple[Any]:
        self.last = len(detections)
        return (detections,)


def _register_avstack_modules() -> None:
    """Import the avstack modules so perception/tracking/planning/control + the attack hooks register."""
    import avstack.modules.control.vehicle
    import avstack.modules.perception.object3d
    import avstack.modules.pipeline
    import avstack.modules.planning.vehicle
    import avstack.modules.tracking.tracker3d  # noqa: F401  (BasicBoxTracker3D)

    import avsectester.attacks.phantom  # noqa: F401  (registers PhantomInjection in avstack HOOKS)


class ModularAVStack(AVStack):
    """An :class:`~avsectester.backend.AVStack` backed by an avstack ``ModularDrivingPipeline``.

    Observation -> the pipeline (perception -> tracking -> planning -> control) -> a Control command.
    White-box modular attacks attach as avstack hooks on a stage via :meth:`attach` (e.g.
    ``PhantomInjection`` on ``perception``); universal sensor/world attacks instead use ``run``'s
    ``perturb`` seam. A detection counter (attached last, after any attack) records per-frame
    detection counts for telemetry.
    """

    def __init__(self, pipeline_cfg: dict) -> None:
        _register_avstack_modules()
        from avstack.config import PIPELINE

        self.pipeline = PIPELINE.build(deepcopy(pipeline_cfg))
        self._counter: _DetectionCounter | None = None
        self.detection_counts: list[int] = []

    def attach(self, stage: str, hook_cfg: dict) -> None:
        from avstack.config import HOOKS

        getattr(self.pipeline, stage).register_post_hook(HOOKS.build(deepcopy(hook_cfg)))

    def attach_counter(self) -> None:
        """Register the detection counter last, so it counts any attack-injected detections too."""
        self._counter = _DetectionCounter()
        self.pipeline.perception.register_post_hook(self._counter)

    def __call__(self, observation: Observation) -> Control:
        ctrl = self.pipeline(observation.sensor_data, observation.vehicle_state)
        if self._counter is not None:
            self.detection_counts.append(self._counter.last)
        return Control(
            throttle=float(ctrl.throttle), steer=float(ctrl.steer), brake=float(ctrl.brake)
        )
