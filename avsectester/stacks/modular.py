"""ModularAVStack — an AVStack backed by an avstack ``ModularDrivingPipeline``.

The modular (white-box) AV stack: perception -> tracking -> planning -> control, built from config.
Heavy avstack imports are lazy (done when the pipeline is built), so this module — and the
``avsectester.stacks`` package — imports without avstack, matching :class:`AlpamayoAVStack`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from avsectester.backend import AVStack
from avsectester.plane import Control, Observation

if TYPE_CHECKING:
    from avsectester.runtime import RuntimeSession

def _remove_hook(hooks: list, target: Any) -> None:
    """Remove our own hook without disturbing hooks installed by the caller."""
    hooks[:] = [hook for hook in hooks if hook is not target]


class _RuntimePreHook:
    def __init__(self, stack: ModularAVStack, stage: str) -> None:
        self.stack = stack
        self.stage = stage

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[tuple, dict]:
        from avsectester.runtime import CallInputs

        inputs = self.stack._emit(self.stage, CallInputs(args=args, kwargs=kwargs))
        if not isinstance(inputs, CallInputs):
            raise TypeError(f"{self.stage} must return CallInputs")
        return inputs.args, inputs.kwargs


class _RuntimePostHook:
    def __init__(self, stack: ModularAVStack, stage: str) -> None:
        self.stack = stack
        self.stage = stage

    def __call__(self, output: Any) -> tuple[Any]:
        # avstack post-hooks return a tuple of positional outputs.
        return (self.stack._emit(self.stage, output),)


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

    import avsectester.attacks.pipeline.phantom  # noqa: F401  (registers PhantomInjection in avstack HOOKS)


class ModularAVStack(AVStack):
    """An :class:`~avsectester.backend.AVStack` backed by an avstack ``ModularDrivingPipeline``.

    Observation -> the pipeline (perception -> tracking -> planning -> control) -> a Control command.
    CARLA-trained MMDetection3D checkpoint configurations register their dataset metadata
    automatically before the pipeline is built.
    Runtime transforms can replace each available stage's inputs and outputs. Native avstack
    hooks remain supported via :meth:`attach` (e.g. ``PhantomInjection`` on ``perception``).
    :meth:`instrument` captures per-stage outputs (the common
    :class:`~avsectester.evaluation.component_log.InstrumentedStack` interface, read via
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
            # The installed avstack decorator has no capability marker, so inspect
            # the effective call wrapper rather than advertising inactive lists.
            call = getattr(type(module), "__call__", None)
            if (
                getattr(call, "__name__", None) == "_apply_hooks"
                and isinstance(getattr(module, "pre_hooks", None), list)
                and isinstance(getattr(module, "post_hooks", None), list)
            ):
                stages[name] = module
        return stages

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
    def runtime_context(self, session: RuntimeSession) -> Iterator[None]:
        """Bridge runtime callbacks for one run, preserving native hook order."""
        installed = []
        with super().runtime_context(session):
            try:
                for name, module in self._stage_modules().items():
                    pre = _RuntimePreHook(self, f"{name}.pre")
                    post = _RuntimePostHook(self, f"{name}.post")
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

    def attach(self, stage: str, hook_cfg: dict) -> None:
        from avstack.config import HOOKS

        modules = self._stage_modules()
        if stage not in modules:
            raise ValueError(f"No hookable {stage!r} stage in this pipeline")
        hooks = modules[stage].post_hooks
        # A hook installed during reset still precedes runtime processing and logging.
        index = next(
            (
                i for i, hook in enumerate(hooks)
                if isinstance(hook, (_RuntimePostHook, _StageCapture))
            ),
            len(hooks),
        )
        hooks.insert(index, HOOKS.build(deepcopy(hook_cfg)))

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
