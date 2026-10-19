"""ModularAVStack — an AVStack backed by an avstack ``ModularDrivingPipeline``.

The modular (white-box) AV stack: perception -> tracking -> planning -> control, built from config.
Heavy avstack imports are lazy (done when the pipeline is built), so this module — and the
``avsectester.closed_loop.avstacks`` package — imports without avstack, matching :class:`AlpamayoAVStack`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.datatypes import Control, Observation

if TYPE_CHECKING:
    from avsectester.closed_loop.hooks import StageEmitter

def _remove_hook(hooks: list, target: Any) -> None:
    """Remove our own hook without disturbing hooks installed by the caller."""
    hooks[:] = [hook for hook in hooks if hook is not target]


class _StagePreHook:
    def __init__(self, avstack: ModularAVStack, stage: str) -> None:
        self.avstack = avstack
        self.stage = stage

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        from avsectester.closed_loop.hooks import CallInputs

        inputs = self.avstack._emit(self.stage, CallInputs(args=args, kwargs=kwargs))
        if not isinstance(inputs, CallInputs):
            raise TypeError(f"{self.stage} must return CallInputs")
        return inputs.args, inputs.kwargs


class _StagePostHook:
    def __init__(self, avstack: ModularAVStack, stage: str) -> None:
        self.avstack = avstack
        self.stage = stage

    def __call__(self, output: Any) -> tuple[Any]:
        # avstack post-hooks return a tuple of positional outputs.
        return (self.avstack._emit(self.stage, output),)


class _StageCapture:
    """A generic avstack post-hook that keeps a pipeline stage's most recent output (pass-through).

    One capture mechanism for every per-stage need — telemetry (how many detections perception emitted)
    and in-system analysis (each stage's output) alike: it snapshots the stage's output before
    downstream modules can mutate it, and returns the original so the pipeline is unaffected."""

    def __init__(self) -> None:
        from avsectester.runlog.codecs import default_codecs
        from avsectester.runlog import Recorder

        # Reuse the canonical snapshot path without creating a writer or retaining a run.
        self._recorder = Recorder(codecs=default_codecs())
        self.last: Any = None

    def __call__(self, output: Any, *args: Any, **kwargs: Any) -> tuple[Any]:
        self.last = self._recorder.capture(output)
        return (output,)


def _register_avstack_modules() -> None:
    """Import the avstack perception, tracking, planning and control modules."""
    import avstack.modules.control.vehicle
    import avstack.modules.perception.object3d
    import avstack.modules.pipeline
    import avstack.modules.planning.vehicle
    import avstack.modules.tracking.tracker3d  # noqa: F401  (BasicBoxTracker3D)



class ModularAVStack(AVStack):
    """An :class:`~avsectester.closed_loop.avstack.AVStack` backed by an avstack ``ModularDrivingPipeline``.

    Observation -> the pipeline (perception -> tracking -> planning -> control) -> a Control command.
    CARLA-trained MMDetection3D checkpoint configurations register their dataset metadata
    automatically before the pipeline is built.
    Runtime transforms replace each available stage's inputs and outputs. Use NativeHook
    to adapt registered avstack attacks to this same runtime chain.
    :meth:`instrument` captures per-stage outputs (the common
    :class:`~avsectester.evaluation.component_log.InstrumentedAVStack` interface, read via
    :meth:`component_log`) — the single source of both per-frame telemetry (n_detections) and in-system
    analysis.
    """

    def __init__(self, pipeline_cfg: dict) -> None:
        _register_avstack_modules()
        from avstack.config import PIPELINE

        perception = pipeline_cfg.get("perception", {})
        if (
            isinstance(perception, Mapping)
            and perception.get("type") == "MMDetObjectDetector3D"
            and perception.get("dataset", "kitti").lower().startswith("carla-")
            and not perception.get("deploy", False)
        ):
            # PointPillars inference also needs its custom dataset metadata.
            from .mmdet3d import register_carla_dataset

            register_carla_dataset()
        self.pipeline = PIPELINE.build(deepcopy(pipeline_cfg))
        self._captures: dict[str, _StageCapture] = {}

    def reset(self, observation: Observation) -> None:
        """Clear telemetry. Use a fresh stack for independent pipeline histories.

        avstack modules do not share a complete reset contract for neural model,
        tracker, planner and controller state. A custom adapter can override reset
        when its modules provide one. Constructing a fresh ModularAVStack is the
        default way to begin an independent experiment.
        """
        for capture in self._captures.values():
            capture.last = None

    def _stage_modules(self) -> dict[str, Any]:
        """Stages whose effective call actually dispatches avstack hooks."""
        stages = {}
        for name in ("perception", "tracking", "planning", "control"):
            module = getattr(self.pipeline, name, None)
            # BaseModule owns hook lists even when a subclass bypasses apply_hooks.
            # Only effective calls marked by the hook dispatcher advertise support.
            call = getattr(type(module), "__call__", None)
            if (
                getattr(call, "__avstack_hooks__", False)
                and isinstance(getattr(module, "pre_hooks", None), list)
                and isinstance(getattr(module, "post_hooks", None), list)
            ):
                stages[name] = module
        return stages

    @property
    def command_modes(self) -> frozenset[str]:
        return frozenset({"actuator"})

    @property
    def supported_stages(self) -> frozenset[str]:
        return frozenset(
            f"{name}.{phase}"
            for name in self._stage_modules()
            for phase in ("pre", "post")
        )

    @property
    def native(self) -> Mapping[str, Any]:
        """The pipeline and its actual modules, including non-hookable modules."""
        return {
            "pipeline": self.pipeline,
            **{
                name: module
                for name in ("perception", "tracking", "planning", "control")
                if (module := getattr(self.pipeline, name, None)) is not None
            },
        }

    @property
    def models(self) -> Mapping[str, Any]:
        """Public handles for the built components and their loaded models."""
        return {
            name: getattr(module, "model", module)
            for name, module in self.native.items()
            if name != "pipeline"
        }

    @contextmanager
    def bind_hooks(self, dispatcher: StageEmitter) -> Iterator[None]:
        """Bridge hook callbacks for one run, preserving native hook order."""
        installed = []
        with super().bind_hooks(dispatcher):
            try:
                for name, module in self._stage_modules().items():
                    pre = _StagePreHook(self, f"{name}.pre")
                    post = _StagePostHook(self, f"{name}.post")
                    module.pre_hooks.append(pre)
                    # Instrumentation must observe the final, modified output.
                    index = next(
                        (i for i, hook in enumerate(module.post_hooks) if isinstance(hook, _StageCapture)),
                        len(module.post_hooks),
                    )
                    module.post_hooks.insert(index, post)
                    installed.extend(((module.pre_hooks, pre), (module.post_hooks, post)))
                yield
            finally:
                for hooks, hook in installed:
                    _remove_hook(hooks, hook)

    def instrument(self, stages: tuple[str, ...] = ("perception", "tracking", "planning", "control")) -> None:
        """Capture each named stage's per-frame output, reusing avstack's post-hook mechanism (a
        :class:`_StageCapture` attached **last** on each stage, so it observes the *attacked* output).
        Enables :meth:`component_log`. Instrument only ``perception`` for plain detection telemetry, or all
        stages for in-system propagation analysis."""
        modules = self._stage_modules()
        missing = set(stages) - modules.keys()
        if missing:
            raise ValueError(f"Cannot instrument unavailable stages: {sorted(missing)}")
        for stage, capture in self._captures.items():
            module = modules.get(stage)
            if module is not None:
                _remove_hook(module.post_hooks, capture)
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
