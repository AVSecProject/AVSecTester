"""Configuration and signature adapters for the shared runtime."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from importlib import import_module
from typing import Any, TYPE_CHECKING

from avsectester.closed_loop.avstacks.native_hook import NativeHook as _NativeHook
from avsectester.preparation import Intervention, Treatment, prepare_intervention

# Deferred concrete attacks use the legacy observation adapter import; implementation is generic.
from avsectester.closed_loop.hooks import (
    Hook, ObservationTransform as ObservationTransform, RunContext, Runtime,
)

if TYPE_CHECKING:
    from avsectester.scenarios.source import ScenarioInstance


class LegacyNativeHook(_NativeHook):
    """Retained for legacy runtime configurations with explicit registration modules.

    Concrete attacks still use the configured-runner path until their migration. Native
    registration is explicit there, just as it is in the experiment API's ``hooks[].imports``.
    """

    def __init__(self, config: Mapping[str, Any], *, phase: str = "post",
                 imports: Sequence[str] = ()) -> None:
        super().__init__(config, phase=phase)
        if not isinstance(imports, Sequence) or isinstance(imports, (str, bytes)) or any(
            not isinstance(module, str) or not module for module in imports
        ):
            raise ValueError("Native hook imports must be module names")
        self.imports = tuple(imports)

    def reset(self, context: RunContext) -> None:
        for module in self.imports:
            import_module(module)
        super().reset(context)


def _factory_definition(factory: Any, options: Any) -> tuple[str, str, dict[str, Any]]:
    """Check the factory specification without importing or constructing user code."""
    if not isinstance(factory, str) or ":" not in factory:
        raise ValueError("factory must be an import path 'package.module:factory'")
    module_name, name = factory.split(":", 1)
    if not module_name or not name:
        raise ValueError("factory must be an import path 'package.module:factory'")
    if not isinstance(options, Mapping):
        raise ValueError("Handler options must be a mapping")
    return module_name, name, dict(options)


def _handler(spec: Mapping[str, Any], *, phase: str = "post") -> Any:
    allowed = {"factory", "options", "avstack", "imports"}
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"Unknown handler options: {sorted(unknown)}")
    if ("factory" in spec) == ("avstack" in spec):
        raise ValueError("A handler requires exactly one of 'factory' or 'avstack'")
    if "avstack" in spec:
        if "options" in spec or not isinstance(spec["avstack"], Mapping):
            raise ValueError("avstack requires its native configuration mapping")
        return LegacyNativeHook(spec["avstack"], phase=phase, imports=spec.get("imports", ()))
    if "imports" in spec:
        raise ValueError("imports requires an avstack native hook configuration")
    factory = spec["factory"]
    module_name, name, options = _factory_definition(factory, spec.get("options", {}))
    try:
        constructor = getattr(import_module(module_name), name)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"Cannot import handler factory {factory!r}") from error
    if not callable(constructor):
        raise ValueError(f"Handler factory {factory!r} is not callable")
    return constructor(**deepcopy(options))


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


class NativeHookIntervention(Intervention):
    """Preparation adapter for an existing avstack post-hook configuration."""

    def __init__(self, stage: str, config: Mapping[str, Any], *, imports: Sequence[str] = ()) -> None:
        self.stage = stage
        self.config = deepcopy(dict(config))
        # Validate at configuration time; defer registration to the hook's lifecycle.
        self.imports = LegacyNativeHook(config, imports=imports).imports

    def prepare(self, scenario: ScenarioInstance) -> dict[str, Any]:
        return deepcopy(self.config)

    def hooks(self, artifacts: Any) -> Sequence[Hook]:
        return (Hook(self.stage, LegacyNativeHook(artifacts, imports=self.imports)),)


def prepare_from_config(
    config: Mapping[str, Any], scenario: ScenarioInstance, *,
    kinds: Iterable[str] | None = None,
) -> Treatment:
    """Prepare convenience and user-defined interventions once for a prepared scenario.

    Legacy convenience methods precede explicit ``runtime.hooks``. New ``interventions``
    entries follow those hooks in their definition order, regardless of attack/defense kind.
    Factory entries construct a method with ``factory(**options)``. An ``artifacts`` entry
    supplies already exported data and skips the method's preparation step.
    """
    selected = None if kinds is None else frozenset(kinds)
    if selected is not None and selected - {"attack", "defense"}:
        raise ValueError("Intervention kinds must be 'attack' or 'defense'")
    prepared = []
    if "defenses" in config:
        raise ValueError("Use interventions with kind: defense instead of a top-level defenses field")
    attacks = config.get("attacks", [])
    patches = config.get("patches", [])
    entries = config.get("interventions", [])
    if not isinstance(attacks, list) or not isinstance(patches, list):
        raise ValueError("attacks and patches must be lists")
    if not isinstance(entries, list):
        raise ValueError("interventions must be a list")
    for attack in attacks:
        if (not isinstance(attack, Mapping) or not {"stage", "hook"} <= set(attack)
                or set(attack) - {"stage", "hook", "imports"}):
            raise ValueError("Each attacks entry requires 'stage', 'hook' and optional 'imports'")
        stage = attack["stage"]
        if stage not in {"perception", "tracking", "planning", "control"}:
            raise ValueError(f"Invalid convenience attack stage: {stage!r}")
        if not isinstance(attack["hook"], Mapping):
            raise ValueError("An avstack hook configuration must be a mapping")
        if selected is None or "attack" in selected:
            hook_type = attack["hook"].get("type", f"{stage}_attack")
            name = hook_type if isinstance(hook_type, str) else getattr(hook_type, "__name__", None)
            prepared.append(prepare_intervention(
                NativeHookIntervention(f"{stage}.post", attack["hook"],
                                       imports=attack.get("imports", ())), scenario,
                name=name, position="before",
            ))
    if patches:
        if any(not isinstance(spec, dict) for spec in patches):
            raise ValueError("Physical patches must be configuration mappings")
        if selected is None or "attack" in selected:
            from avsectester.attacks.patch.physical_patch import PhysicalPatchIntervention

            prepared.append(prepare_intervention(
                PhysicalPatchIntervention(patches), scenario, position="before",
            ))
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("Each intervention must be a configuration mapping")
        unknown = set(entry) - {"factory", "options", "kind", "name", "artifacts"}
        if unknown:
            raise ValueError(f"Unknown intervention options: {sorted(unknown)}")
        if "factory" not in entry:
            raise ValueError("Each intervention requires a factory")
        kind = entry.get("kind", "attack")
        if kind not in {"attack", "defense"}:
            raise ValueError("Intervention kind must be 'attack' or 'defense'")
        name = entry.get("name")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("Intervention name must be a nonempty string")
        _factory_definition(entry["factory"], entry.get("options", {}))
        if selected is not None and kind not in selected:
            continue
        method = _handler({key: entry[key] for key in ("factory", "options") if key in entry})
        options = {key: entry[key] for key in ("kind", "name") if key in entry}
        if "artifacts" in entry:
            restore = getattr(method, "import_artifacts", lambda value: value)
            options["artifacts"] = restore(deepcopy(entry["artifacts"]))
        prepared.append(prepare_intervention(method, scenario, **options))
    return Treatment(tuple(prepared))


def configured_runtime(
    scenario: Mapping[str, Any], *, case: Any = None, instance: ScenarioInstance | None = None,
    prepared: Treatment | None = None, kinds: Iterable[str] | None = None,
) -> Runtime:
    """Adapt configuration to Runtime while retaining the existing Python entry point.

    High-level runners pass their ScenarioInstance as ``instance``. Direct legacy callers
    still get a preparation context containing the supplied match and configuration.
    """
    if instance is None:
        from avsectester.scenarios.source import ScenarioInstance

        def make_backend():
            raise RuntimeError("Supply a ScenarioInstance to access its backend factory")

        instance = ScenarioInstance(make_backend, case, {"configuration": deepcopy(dict(scenario))})
    if case is None:
        case = instance.match
    selected = None if kinds is None else frozenset(kinds)
    if selected is not None and selected - {"attack", "defense"}:
        raise ValueError("Intervention kinds must be 'attack' or 'defense'")
    runtime_config = scenario.get("runtime")
    if selected is not None and "attack" not in selected:
        if runtime_config is not None:
            if not isinstance(runtime_config, Mapping):
                raise ValueError("runtime must be a mapping")
            unknown = set(runtime_config) - {"hooks", "localizer", "seed"}
            if unknown:
                raise ValueError(f"Unknown runtime options: {sorted(unknown)}")
            runtime_config = {
                key: value for key, value in runtime_config.items() if key in {"localizer", "seed"}
            }
    runtime = Runtime.from_config(runtime_config, case=case)
    treatment = prepared if prepared is not None else prepare_from_config(
        scenario, instance, kinds=selected,
    )
    return treatment.runtime(runtime, case, kinds=selected)


def configured_clean_runtime(
    config: Mapping[str, Any], *, case: Any = None, attacked_runtime: Runtime | None = None,
) -> Runtime:
    """Construct only the baseline, without preparing attacks or their handler factories.

    An explicit Python attacked runtime supplies the default baseline's localizer, case and
    seed. Stateful shared localizers must reset or have an explicitly independent baseline.
    A YAML clean_runtime overrides this default and constructs its own configured localizer.
    """
    if "clean_runtime" in config:
        return Runtime.from_config(config["clean_runtime"], case=case)
    if attacked_runtime is not None:
        return Runtime(localizer=attacked_runtime.localizer, seed=attacked_runtime.seed,
                       case=attacked_runtime.case if attacked_runtime.case is not None else case)
    common = config.get("runtime")
    common = {} if common is None else common
    if not isinstance(common, Mapping):
        raise ValueError("runtime must be a mapping")
    unknown = set(common) - {"hooks", "localizer", "seed"}
    if unknown:
        raise ValueError(f"Unknown runtime options: {sorted(unknown)}")
    return Runtime.from_config({
        key: value for key, value in common.items() if key in {"localizer", "seed"}
    }, case=case)


def configured_runtime_pair(
    scenario: Mapping[str, Any],
    *,
    clean_runtime: Runtime | None = None,
    attacked_runtime: Runtime | None = None,
    instance: ScenarioInstance | None = None,
    prepared: Treatment | None = None,
    kinds: Iterable[str] | None = None,
) -> tuple[Runtime, Runtime]:
    """Build two treatments, preserving explicit Python runtime overrides.

    YAML localizer factories build independent instances. An explicit attacked runtime shares
    its localizer with the default clean runtime. See configured_clean_runtime for that rule.
    """
    if clean_runtime is None:
        clean_runtime = configured_clean_runtime(
            scenario, case=instance.match if instance is not None else None,
            attacked_runtime=attacked_runtime,
        )
    if attacked_runtime is None:
        attacked_runtime = configured_runtime(scenario, instance=instance, prepared=prepared,
                                             kinds=kinds)
    return clean_runtime, attacked_runtime
