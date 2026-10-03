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


class _StageCapture:
    """A generic avstack post-hook that keeps a pipeline stage's most recent output (pass-through).

    One capture mechanism for every per-stage need — telemetry (how many detections perception emitted)
    and in-system analysis (each stage's output) alike: it stores the stage's output so the stack can
    report per-layer results, and returns it unchanged so the pipeline is unaffected."""

    def __init__(self) -> None:
        self.last: Any = None

    def __call__(self, output: Any, *args: Any, **kwargs: Any) -> tuple[Any]:
        self.last = output
        return (output,)


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
    ``perturb`` seam. :meth:`instrument` captures per-stage outputs (the common
    :class:`~avsectester.evaluation.component_log.InstrumentedStack` interface, read via
    :meth:`component_log`) — the single source of both per-frame telemetry (n_detections) and in-system
    analysis.
    """

    def __init__(self, pipeline_cfg: dict) -> None:
        _register_avstack_modules()
        from avstack.config import PIPELINE

        self.pipeline = PIPELINE.build(deepcopy(pipeline_cfg))
        self._captures: dict[str, _StageCapture] = {}

    def attach(self, stage: str, hook_cfg: dict) -> None:
        from avstack.config import HOOKS

        getattr(self.pipeline, stage).register_post_hook(HOOKS.build(deepcopy(hook_cfg)))

    def instrument(self, stages: tuple[str, ...] = ("perception", "tracking", "planning", "control")) -> None:
        """Capture each named stage's per-frame output, reusing avstack's post-hook mechanism (a
        :class:`_StageCapture` attached **last** on each stage, so it observes the *attacked* output).
        Enables :meth:`component_log`. Instrument only ``perception`` for plain detection telemetry, or all
        stages for in-system propagation analysis."""
        self._captures = {s: _StageCapture() for s in stages}
        for stage, capture in self._captures.items():
            getattr(self.pipeline, stage).register_post_hook(capture)

    def component_log(self) -> dict[str, Any] | None:
        """The most recent per-stage outputs ``{stage: output}`` (needs :meth:`instrument` first), or
        None if the stack is not instrumented. The caller (an evaluation logger) derives counts/metrics
        from the raw avstack outputs — reusing e.g. ``avstack.metrics.get_instantaneous_metrics``."""
        return {stage: cap.last for stage, cap in self._captures.items()} if self._captures else None

    def __call__(self, observation: Observation) -> Control:
        ctrl = self.pipeline(observation.sensor_data, observation.vehicle_state)
        return Control(
            throttle=float(ctrl.throttle), steer=float(ctrl.steer), brake=float(ctrl.brake)
        )
