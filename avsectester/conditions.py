"""Environment settings shared by the treatments of a prepared scenario.

World settings are configured on the backend before reset. Sensor settings produce fresh
runtime handlers for every treatment, with random streams independent of attack handlers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from importlib import import_module
from typing import Any, TYPE_CHECKING

import numpy as np

from avsectester.plane import Observation
from avsectester.runtime import Hook, Plugin, Runtime

if TYPE_CHECKING:
    from avsectester.backend import WorldBackend
    from avsectester.runtime import RunContext, StageContext
    from avsectester.scenarios.context import FilterContext


class ConditionConfigurationError(ValueError):
    """A backend rejects a configured world condition before execution."""


def _seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Condition seed must be a nonnegative integer")
    return value


class _ConditionHandler(Plugin):
    """Give a condition its own lifecycle and RNG at each relative driving step."""

    def __init__(self, handler: Any, seed: int) -> None:
        if not callable(getattr(handler, "transform", handler)):
            raise TypeError("Sensor condition factories must return callable handlers or plugins")
        self.handler, self.seed = handler, seed
        self._owner = getattr(handler, "lifecycle_owner", handler)

    def reset(self, context: RunContext) -> None:
        reset = getattr(self._owner, "reset", None)
        if reset is not None:
            reset(replace(context, rng=np.random.default_rng(self.seed)))

    def _context(self, context: StageContext, phase: int = 0) -> StageContext:
        rng = np.random.default_rng((self.seed, context.step, phase))
        return replace(context, run=replace(context.run, rng=rng))

    def transform(self, value: Any, context: StageContext) -> Any:
        return getattr(self.handler, "transform", self.handler)(value, self._context(context))

    def after_step(self, context: StageContext) -> None:
        after = getattr(self._owner, "after_step", None)
        if after is not None:
            after(self._context(context, 1))

    def close(self) -> None:
        close = getattr(self._owner, "close", None)
        if close is not None:
            close()


@dataclass(frozen=True)
class SensorCondition:
    """A fresh sensor handler factory and its position relative to digital interventions.

    The handler has the usual ``handler(observation, context)`` or Plugin interface. Factories
    return independent instances. ``order='after'`` preserves attack-then-corruption behavior.
    ``seed=None`` inherits the scenario-condition seed, independently of Runtime.seed.
    """

    factory: Callable[[], Any]
    stage: str = "observation"
    order: str = "after"
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not callable(self.factory):
            raise TypeError("Sensor condition factory must be callable")
        if self.stage not in {"sensors.post", "observation"}:
            raise ValueError("Sensor conditions use 'sensors.post' or 'observation'")
        if self.order not in {"before", "after"}:
            raise ValueError("Sensor condition order must be 'before' or 'after'")
        if self.seed is not None:
            _seed(self.seed)

    @classmethod
    def image(
        cls,
        pipeline: Any,
        *,
        camera: str | None = None,
        stage: str = "observation",
        order: str = "after",
    ) -> SensorCondition:
        """Wrap an existing AugmentationPipeline without sharing its mutable operators."""
        from avsectester.perturbations.image import AugmentationPipeline

        if not isinstance(pipeline, AugmentationPipeline):
            raise TypeError("Image conditions require AugmentationPipeline")
        pipeline = deepcopy(pipeline)
        metadata = {
            "type": "image",
            "camera": camera,
            "corruptions": [
                {
                    "type": type(c).__name__, "severity": c.severity,
                    "options": {key: deepcopy(value) for key, value in vars(c).items()
                                if key != "severity" and not key.startswith("_") and not callable(value)},
                } for c in pipeline.corruptions
            ],
        }
        return cls(
            lambda: _ImageCondition(deepcopy(pipeline), camera),
            stage, order, pipeline.seed, metadata,
        )


class _ImageCondition:
    def __init__(self, pipeline: Any, camera: str | None) -> None:
        self.pipeline, self.camera = pipeline, camera

    def __call__(self, observation: Observation, context: StageContext) -> Observation:
        from avsectester.perturbations.image import read_camera_rgb, write_camera_rgb

        if self.camera is not None and self.camera not in observation.sensor_data:
            raise ValueError(f"Sensor condition camera {self.camera!r} is absent")
        image = read_camera_rgb(observation, self.camera)
        if image is None and self.camera is None:
            image = next((
                item for key in observation.sensor_data
                if (item := read_camera_rgb(observation, key)) is not None
            ), None)
        if image is None:
            raise ValueError("Image condition requires a readable RGB camera")
        key, rgb = image
        return write_camera_rgb(observation, key, self.pipeline.apply(rgb, context.step))


def _sensor_from_config(config: Mapping[str, Any], default_seed: int) -> SensorCondition:
    if not isinstance(config, Mapping):
        raise ValueError("Each sensor condition must be a mapping")
    spec = deepcopy(dict(config))
    stage, order = spec.pop("stage", "observation"), spec.pop("order", "after")
    seed = _seed(spec.pop("seed", default_seed))
    if "factory" in spec:
        unknown = set(spec) - {"factory", "options"}
        if unknown:
            raise ValueError(f"Unknown sensor condition options: {sorted(unknown)}")
        path, options = spec["factory"], spec.get("options", {})
        if not isinstance(path, str) or ":" not in path or not isinstance(options, Mapping):
            raise ValueError("Condition factory requires 'module:callable' and mapping options")
        module, name = path.split(":", 1)
        try:
            constructor = getattr(import_module(module), name)
        except (ImportError, AttributeError) as error:
            raise ValueError(f"Cannot import condition factory {path!r}") from error
        if not callable(constructor):
            raise ValueError(f"Condition factory {path!r} is not callable")
        factory = lambda: constructor(**deepcopy(dict(options)))
        return SensorCondition(factory, stage, order, seed, spec)
    from avsectester.perturbations.image import AugmentationPipeline, CORRUPTIONS

    allowed = {"type", "severity", "camera", "options"}
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"Unknown image condition options: {sorted(unknown)}")
    kind = spec.get("type")
    if not isinstance(kind, str) or kind not in CORRUPTIONS:
        raise ValueError(f"Unknown image condition {kind!r}. Available: {sorted(CORRUPTIONS)}")
    options = spec.get("options", {})
    if not isinstance(options, Mapping):
        raise ValueError("Image condition options must be a mapping")
    severity = spec.get("severity", 0.5)
    if isinstance(severity, bool) or not isinstance(severity, (int, float)) or not 0 <= severity <= 1:
        raise ValueError("Image condition severity must be in [0, 1]")
    pipeline = AugmentationPipeline([CORRUPTIONS[kind](severity, **options)], seed=seed)
    camera = spec.get("camera")
    if camera is not None and (not isinstance(camera, str) or not camera):
        raise ValueError("Condition camera must be a nonempty string")
    return replace(
        SensorCondition.image(pipeline, camera=camera, stage=stage, order=order),
        metadata=spec,
    )


@dataclass(frozen=True)
class ScenarioConditions:
    """Operating environment applied to every treatment of one scenario.

    Conditions are shared experimental settings, rather than attack or defense variables.
    World properties are applied before reset. Sensor factories build independent handlers
    using the same configured seeds and relative-step indexing for each treatment.
    """

    world: Mapping[str, Any] = field(default_factory=dict)
    sensors: tuple[SensorCondition, ...] = ()
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.world, Mapping):
            raise ValueError("World conditions must be a mapping")
        _seed(self.seed)
        if any(not isinstance(condition, SensorCondition) for condition in self.sensors):
            raise TypeError("Sensor conditions must be SensorCondition instances")
        object.__setattr__(self, "world", deepcopy(dict(self.world)))
        object.__setattr__(self, "sensors", tuple(self.sensors))

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> ScenarioConditions:
        if config is None:
            return cls()
        if not isinstance(config, Mapping):
            raise ValueError("conditions must be a mapping")
        unknown = set(config) - {"world", "sensors", "seed"}
        if unknown:
            raise ValueError(f"Unknown conditions options: {sorted(unknown)}")
        seed = _seed(config.get("seed", 0))
        sensors = config.get("sensors", ())
        if not isinstance(sensors, Sequence) or isinstance(sensors, (str, bytes)):
            raise ValueError("conditions.sensors must be a sequence")
        return cls(config.get("world", {}), tuple(_sensor_from_config(s, seed) for s in sensors), seed)

    def configure_backend(self, backend: WorldBackend) -> None:
        """Require the backend to accept every configured world property before reset."""
        if self.world:
            try:
                backend.configure_conditions(deepcopy(dict(self.world)))
            except ValueError as error:
                raise ConditionConfigurationError(str(error)) from error

    def apply_runtime(self, runtime: Runtime) -> Runtime:
        """Construct independent condition handlers around the existing ordered hooks."""
        before, after = [], []
        for condition in self.sensors:
            seed = self.seed if condition.seed is None else condition.seed
            hook = Hook(condition.stage, _ConditionHandler(condition.factory(), seed))
            (before if condition.order == "before" else after).append(hook)
        return runtime.with_hooks(before, prepend=True).with_hooks(after)

    def provenance(self) -> dict[str, Any]:
        """Return configuration metadata without retaining handlers or native resources."""
        return {
            "world": deepcopy(dict(self.world)),
            "seed": self.seed,
            "sensors": [
                {
                    **deepcopy(dict(c.metadata)), "stage": c.stage, "order": c.order,
                    "seed": self.seed if c.seed is None else c.seed,
                }
                for c in self.sensors
            ],
        }

    def selection_scope(self):
        """Own fresh condition state for one prescribed selection-frame sequence.

        The returned context manager is callable with ``(context, frame_index)``. Providers
        supply ``native['observation']`` at each scene timestamp. Conditioned measurements
        are exposed separately from annotations and raw provider observations.
        """
        return _SelectionScope(self)

    def selection_context(self, context: FilterContext) -> FilterContext:
        """Apply conditions to one selection frame with an independent handler lifecycle."""
        with self.selection_scope() as prepare:
            return prepare(context, context.frame_index)

    @staticmethod
    def selection_observation(context: FilterContext) -> Observation:
        """Get conditioned preview measurements or explain the missing provider capability."""
        observation = context.native.get("conditioned_observation")
        if observation is None:
            raise ValueError("This selection provider supplies no observation for sensor conditions")
        return observation


class _SelectionScope:
    """One condition session for a prescribed window, without advancing a driving policy."""

    def __init__(self, conditions: ScenarioConditions):
        self.conditions = conditions
        self.session = None
        self.activation = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.activation is not None:
            return self.activation.__exit__(*exc)

    def __call__(self, context: FilterContext, frame_index: int = 0) -> FilterContext:
        native = {**context.native, "conditions": self.conditions.provenance()}
        self.current_context = context
        observation = native.get("observation")
        if observation is None:
            return context.derive(native=native)
        if not isinstance(observation, Observation):
            raise TypeError("Selection observation must be Observation")
        if observation.frame != context.scene.frame or not np.isclose(observation.t, context.scene.t):
            raise ValueError("Selection observation and ground truth must describe the same frame")
        if not self.conditions.sensors:
            # No processing is needed. Retain the provider's capture without constructing a
            # session or attempting to copy borrowed SDK objects.
            native["conditioned_observation"] = observation
            return context.derive(native=native)
        if self.session is None:
            from avsectester.backend import AVStack, WorldBackend
            from avsectester.plane import Control

            scope = self

            class SnapshotBackend(WorldBackend):
                def reset(self):
                    raise RuntimeError("Selection conditions do not reset the world")

                def step(self, control):
                    raise RuntimeError("Selection conditions do not advance the world")

                def copy_observation(self, observation):
                    provider = scope.current_context.backend
                    copier = scope.current_context.native.get("copy_observation")
                    if copier is None and provider is not None:
                        copier = getattr(provider, "copy_observation", None)
                    if copier is not None:
                        return copier(observation)
                    return super().copy_observation(observation)

                @property
                def native(self):
                    return {**scope.current_context.native, "source_backend": scope.current_context.backend}

            class UnusedStack(AVStack):
                def __call__(self, seen):
                    return Control()

            self.backend = SnapshotBackend()
            self.backend._remember_ground_truth(observation)
            self.session = self.conditions.apply_runtime(Runtime()).session(self.backend, UnusedStack())
            self.activation = self.session.activate()
            self.activation.__enter__()
        self.backend._remember_ground_truth(observation)
        self.session.step = frame_index
        native["conditioned_observation"] = self.session.observe(observation)
        return context.derive(native=native)
