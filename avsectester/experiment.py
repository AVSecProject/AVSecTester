"""Assemble configured scenarios and interventions through the shared execution interfaces."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from enum import Enum
from typing import Any, TYPE_CHECKING

from avsectester.backend import AVStack, WorldBackend
from avsectester.interventions import (
    _handler, configured_clean_runtime, configured_runtime, configured_runtime_pair,
    prepare_from_config,
)
from avsectester.plane import Trace
from avsectester.runner import run_prepared
from avsectester.runtime import Runtime
from avsectester.scenarios.setup import scenario_from_config

if TYPE_CHECKING:
    from avsectester.scenarios.source import ScenarioInstance


class ConfigurationError(ValueError):
    """An experiment configuration could not be prepared for execution."""


class ExecutionMode(str, Enum):
    SINGLE = "single"
    PAIR = "pair"


class TreatmentMode(str, Enum):
    CLEAN = "clean"
    ATTACK = "attack"
    DEFENSE = "defense"
    COMBINED = "attack-defense"

    @property
    def kinds(self) -> tuple[str, ...]:
        return {
            self.CLEAN: (), self.ATTACK: ("attack",),
            self.DEFENSE: ("defense",), self.COMBINED: ("attack", "defense"),
        }[self]


def configured_stack(config: Mapping[str, Any]) -> AVStack:
    """Build a supplied driving adapter or a user's importable stack factory."""
    try:
        return _build_stack(config)
    except (ValueError, TypeError, KeyError, ImportError) as error:
        raise ConfigurationError(str(error)) from error


def _build_stack(config: Mapping[str, Any]) -> AVStack:
    spec = config.get("stack")
    if spec is None:
        from avsectester.stacks.modular import ModularAVStack

        stack = ModularAVStack(config["ego"]["pipeline"])
        stack.instrument(stages=("perception",))
        return stack
    if not isinstance(spec, Mapping):
        raise ValueError("stack must be a mapping")
    if "factory" in spec:
        stack = _handler(spec)
    else:
        if set(spec) - {"type", "options"} or spec.get("type") != "AlpamayoAVStack":
            raise ValueError("Use stack.factory/options or type: AlpamayoAVStack")
        options = spec.get("options", {})
        if not isinstance(options, Mapping):
            raise ValueError("Stack options must be a mapping")
        from avsectester.stacks.alpamayo import AlpamayoAVStack

        stack = AlpamayoAVStack(**deepcopy(dict(options)))
    if not isinstance(stack, AVStack):
        raise TypeError("Stack factories must return AVStack implementations")
    return stack


def run_configured(
    config: Mapping[str, Any],
    frames: int,
    *,
    mode: ExecutionMode | str | None = None,
    treatment: TreatmentMode | str | None = None,
    backend_factory: Callable[[dict[str, Any]], WorldBackend] | None = None,
    stack_factory: Callable[[], AVStack] | None = None,
    clean_runtime: Runtime | None = None,
    treated_runtime: Runtime | None = None,
    run_fn: Callable[..., Trace] | None = None,
    provenance: Mapping[str, Any] | None = None,
    scenario: ScenarioInstance | None = None,
) -> dict[str, Trace]:
    """Prepare a scenario, prepare enabled interventions, and run named treatments.

    Single mode executes one treatment. Pair mode executes a clean baseline and the selected
    treatment from one origin. Preparation runs once, while execution handlers and sensor
    conditions are constructed independently. Supplying a treated Runtime replaces configured
    interventions, as with the existing CARLA pair entry point. Explicit mode/treatment keywords
    override the mapping's settings, which otherwise default to pair/attack. A default empty
    attack pair compares baselines. Other non-clean modes require the selected method kinds.
    """
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 0:
        raise ValueError("frames must be a nonnegative integer")
    config = deepcopy(dict(config))
    try:
        mode = ExecutionMode(mode if mode is not None else config.get("mode", "pair"))
        treatment = TreatmentMode(treatment if treatment is not None else config.get("treatment", "attack"))
        instance = scenario if scenario is not None else scenario_from_config(
            config, backend_factory=backend_factory, provenance=provenance,
        )
        prepared = None
        if treated_runtime is None:
            prepared = prepare_from_config(config, instance, kinds=treatment.kinds)
        if treatment is TreatmentMode.CLEAN:
            selected = clean_runtime or configured_clean_runtime(config, case=instance.match)
            runtimes = {"clean": selected}
            if mode is ExecutionMode.PAIR:
                # YAML factories are independent even when both treatments are baselines.
                runtimes["baseline"] = clean_runtime or configured_clean_runtime(config, case=instance.match)
        elif mode is ExecutionMode.SINGLE:
            selected = treated_runtime or configured_runtime(
                config, instance=instance, prepared=prepared, kinds=treatment.kinds,
            )
            runtimes = {treatment.value: selected}
        else:
            baseline, selected = configured_runtime_pair(
                config, instance=instance, prepared=prepared, kinds=treatment.kinds,
                clean_runtime=clean_runtime, attacked_runtime=treated_runtime,
            )
            runtimes = {"clean": baseline, treatment.value: selected}
        if treatment is not TreatmentMode.CLEAN and treated_runtime is None:
            configured_kinds = {item.kind for item in prepared.interventions}
            if selected.hooks:
                # Raw runtime hooks and convenience fields are classified as attacks.
                if (config.get("runtime") or {}).get("hooks"):
                    configured_kinds.add("attack")
            required = set(treatment.kinds)
            # Preserve the default empty pair as an explicit baseline comparison.
            if not selected.hooks and mode is ExecutionMode.PAIR and treatment is TreatmentMode.ATTACK:
                required.clear()
            if required - configured_kinds:
                missing = required - configured_kinds
                message = f"No enabled intervention configured for {sorted(missing)}"
                if "defense" in missing and (config.get("runtime") or {}).get("hooks"):
                    message += (
                        ". Raw runtime.hooks are attack handlers in configured treatments. "
                        "Declare defenses under interventions with kind: defense."
                    )
                raise ValueError(message)
    except (ValueError, TypeError, KeyError, ImportError) as error:
        raise ConfigurationError(str(error)) from error
    results = run_prepared(
        instance, stack_factory or (lambda: configured_stack(config)), frames,
        runtimes=runtimes, run_fn=run_fn,
    )
    for name, trace in results.items():
        trace.provenance["interventions"] = [
            {"name": item.name, "kind": item.kind, "provenance": deepcopy(item.provenance)}
            for item in (() if prepared is None or name == "clean" else prepared.interventions)
        ]
    return results
