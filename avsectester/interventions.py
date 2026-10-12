"""Configuration and signature adapters for the shared runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from importlib import import_module
from typing import Any

from avsectester.plane import Observation
from avsectester.runtime import CallInputs, Hook, Plugin, RunContext, Runtime, StageContext


class ObservationTransform:
    """Adapt a one-argument observation function without losing its lifecycle."""

    def __init__(self, transform: Callable[[Observation], Observation]) -> None:
        self.function = transform

    @property
    def lifecycle_owner(self) -> Any:
        return self.function

    def transform(self, observation: Observation, context: StageContext) -> Observation:
        return self.function(observation)


class NativeHook(Plugin):
    """Adapt an avstack hook's tuple convention to a runtime stage value.

    The native hook is built afresh for every run. Its algorithm and configured arguments
    remain unchanged. Use a runtime plugin directly when contextual feedback is required.
    """

    def __init__(self, config: Mapping[str, Any], *, phase: str = "post") -> None:
        if phase not in {"pre", "post"}:
            raise ValueError("NativeHook phase must be 'pre' or 'post'")
        self.config = deepcopy(dict(config))
        self.phase = phase
        self.native: Any = None

    def reset(self, context: RunContext) -> None:
        from avstack.config import HOOKS

        self.native = HOOKS.build(deepcopy(self.config))

    def transform(self, value: Any, context: StageContext) -> Any:
        if self.phase == "pre":
            if not isinstance(value, CallInputs):
                raise TypeError("Native pre-hooks require CallInputs")
            args, kwargs = self.native(*value.args, **value.kwargs)
            return CallInputs(args=args, kwargs=kwargs)
        result = self.native(value)
        if not isinstance(result, tuple) or len(result) != 1:
            raise TypeError("Native post-hooks must return one output in a tuple")
        return result[0]

    def close(self) -> None:
        native, self.native = self.native, None
        close = getattr(native, "close", None)
        if close is not None:
            close()


def _handler(spec: Mapping[str, Any], *, phase: str = "post") -> Any:
    allowed = {"factory", "options", "avstack"}
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"Unknown handler options: {sorted(unknown)}")
    if ("factory" in spec) == ("avstack" in spec):
        raise ValueError("A handler requires exactly one of 'factory' or 'avstack'")
    if "avstack" in spec:
        if "options" in spec or not isinstance(spec["avstack"], Mapping):
            raise ValueError("avstack requires its native configuration mapping")
        return NativeHook(spec["avstack"], phase=phase)
    factory = spec["factory"]
    if not isinstance(factory, str) or ":" not in factory:
        raise ValueError("factory must be an import path 'package.module:factory'")
    module_name, name = factory.split(":", 1)
    options = spec.get("options", {})
    if not isinstance(options, Mapping):
        raise ValueError("Handler options must be a mapping")
    try:
        constructor = getattr(import_module(module_name), name)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"Cannot import handler factory {factory!r}") from error
    if not callable(constructor):
        raise ValueError(f"Handler factory {factory!r} is not callable")
    return constructor(**deepcopy(dict(options)))


def runtime_from_config(config: Mapping[str, Any] | None, *, case: Any = None) -> Runtime:
    """Build ordered hooks from configuration, including user-provided factories."""
    if config is None:
        return Runtime(case=case)
    if not isinstance(config, Mapping):
        raise ValueError("runtime must be a mapping")
    unknown = set(config) - {"hooks", "localizer", "seed"}
    if unknown:
        raise ValueError(f"Unknown runtime options: {sorted(unknown)}")
    entries = config.get("hooks", [])
    if not isinstance(entries, list):
        raise ValueError("runtime.hooks must be a list")
    hooks = []
    for entry in entries:
        if not isinstance(entry, Mapping) or "stage" not in entry:
            raise ValueError("Each runtime hook requires a stage and handler configuration")
        stage = entry["stage"]
        if not isinstance(stage, str) or not stage:
            raise ValueError("Hook stage must be a nonempty string")
        spec = {key: value for key, value in entry.items() if key != "stage"}
        hooks.append(Hook(stage, _handler(spec, phase="pre" if stage.endswith(".pre") else "post")))
    localizer = config.get("localizer")
    if localizer is not None:
        if not isinstance(localizer, Mapping) or "avstack" in localizer:
            raise ValueError("runtime.localizer requires a factory and optional options")
        localizer = _handler(localizer)
    return Runtime(hooks, localizer=localizer, case=case, seed=config.get("seed", 0))


def configured_runtime(scenario: Mapping[str, Any], *, case: Any = None) -> Runtime:
    """Translate CARLA convenience fields into the same ordered hook configuration."""
    runtime = Runtime.from_config(scenario.get("runtime"), case=case)
    hooks = []
    attacks = scenario.get("attacks", [])
    patches = scenario.get("patches", [])
    if not isinstance(attacks, list) or not isinstance(patches, list):
        raise ValueError("attacks and patches must be lists")
    for attack in attacks:
        if not isinstance(attack, Mapping) or set(attack) != {"stage", "hook"}:
            raise ValueError("Each attacks entry requires exactly 'stage' and 'hook'")
        stage = attack["stage"]
        if stage not in {"perception", "tracking", "planning", "control"}:
            raise ValueError(f"Invalid convenience attack stage: {stage!r}")
        if not isinstance(attack["hook"], Mapping):
            raise ValueError("An avstack hook configuration must be a mapping")
        hooks.append(Hook(f"{stage}.post", NativeHook(attack["hook"])))
    if patches:
        from avsectester.attacks.patch.physical_patch import PhysicalPatchPlugin

        hooks.append(Hook("world.setup", PhysicalPatchPlugin(patches)))
    return runtime.with_hooks(hooks, prepend=True)


def configured_runtime_pair(
    scenario: Mapping[str, Any],
    *,
    clean_runtime: Runtime | None = None,
    attacked_runtime: Runtime | None = None,
) -> tuple[Runtime, Runtime]:
    """Build CARLA treatments, with explicit Python runtimes overriding YAML defaults.

    YAML localizer factories build independent instances for each treatment. An explicit
    attacked runtime shares its localizer with the default clean runtime, so a stateful
    localizer must reset itself or the caller must supply an independent clean runtime.
    """
    explicit_attacked_runtime = attacked_runtime is not None
    if attacked_runtime is None:
        attacked_runtime = configured_runtime(scenario)
    if clean_runtime is None:
        if "clean_runtime" in scenario or not explicit_attacked_runtime:
            common = scenario.get("runtime") or {}
            baseline = scenario.get("clean_runtime", {
                key: value for key, value in common.items() if key in {"localizer", "seed"}
            })
            clean_runtime = Runtime.from_config(baseline)
        else:
            clean_runtime = Runtime(
                localizer=attacked_runtime.localizer, case=attacked_runtime.case,
                seed=attacked_runtime.seed,
            )
    return clean_runtime, attacked_runtime
