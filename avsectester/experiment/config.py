"""Construct experiment objects from Python mappings or YAML files.

An experiment declares ``dataset``, ``simulator``, ``avstack``, named ``attacks`` and
``runs`` settings. Attack and driving specifications can be inline mappings or YAML paths.
Every constructed object is selected by one key, ``type``, with keyword arguments in
``options``: a value without a colon names a built-in class (``type: NuRecSimulator``), and
``package.module:name`` names user code (``type: my_project.driving:MyAVStack``). Known
filesystem arguments annotated with Path or os.PathLike are resolved relative to the file
containing them. Other constructor
arguments remain user-owned.

``avstack.type`` names an AVStack class or a callable returning an AVStack. It is called with
``options`` for every run and service operation, so each run has a fresh driving history.
Optional ``avstack.bindings`` exposes driving-system components and scorers to closed-loop
services.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import partial
from importlib import import_module
import inspect
from os import PathLike
from pathlib import Path
from types import SimpleNamespace, UnionType
from typing import Annotated, Any, TypeAlias, Union, get_args, get_origin, get_type_hints
import warnings

import yaml  # type: ignore[import-untyped]

from avsectester.validation import require_positive_integer
from avsectester.attack import Attack
from avsectester.closed_loop.avstacks.base import AVStack
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.closed_loop.avstacks.modular import NativeHook
from avsectester.execution.runner import RecordingPolicy
from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter
from avsectester.closed_loop.simulators.base import Simulator
from avsectester.closed_loop.avstacks.base import ObservationPipeline
from avsectester.datasets import Dataset
from avsectester.evaluation.report import EvaluationConfiguration
from avsectester.evaluation.metric import Metric
from . import configuration_summary, type_registry


HookConfiguration: TypeAlias = HookSet | Callable[[int], HookSet] | None


@dataclass(frozen=True)
class TreatmentConfiguration:
    """One attack, or a clean run, with explicit environment overrides."""

    name: str
    attack_name: str | None = None
    environment: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _name(self.name, "Treatment name")
        if self.attack_name is not None:
            _name(self.attack_name, "Treatment attack")
        object.__setattr__(self, "environment", _environment_override(self.environment))


class _ConfiguredAttacks(dict[str, Attack]):
    """An attack mapping carrying loader declarations through configuration copies."""

    def __init__(self, attacks: Mapping[str, Attack],
                 specifications: Mapping[str, tuple[Attack, Mapping[str, Any]]]) -> None:
        super().__init__(attacks)
        self.specifications = {name: (attack, deepcopy(parameters))
                               for name, (attack, parameters) in specifications.items()}


@dataclass(frozen=True)
class RunsConfiguration:
    """What to run on the configured components, and how to record and handle those runs."""

    treatments: tuple[TreatmentConfiguration, ...] = ()
    seeds: tuple[int, ...] = (0,)
    decision_steps: int | None = None
    environment: Mapping[str, Any] = field(default_factory=dict)
    recording: RecordingPolicy = field(default_factory=RecordingPolicy)
    output: Path | None = None
    strict: bool = False

    def __post_init__(self) -> None:
        from avsectester.closed_loop.environment import normalize_environment

        treatments = tuple(self.treatments)
        if any(not isinstance(item, TreatmentConfiguration) for item in treatments):
            raise TypeError("treatments must contain TreatmentConfiguration values")
        if len({item.name for item in treatments}) != len(treatments):
            raise ValueError("Treatment names must be unique")
        if not self.seeds or any(isinstance(seed, bool) or not isinstance(seed, int)
                                 or seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be a nonempty sequence of nonnegative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must not contain duplicates")
        if self.decision_steps is not None:
            require_positive_integer(self.decision_steps, "decision_steps")
        if not isinstance(self.recording, RecordingPolicy):
            raise TypeError("recording must be RecordingPolicy")
        if not isinstance(self.strict, bool):
            raise TypeError("strict must be boolean")
        object.__setattr__(self, "treatments", treatments)
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "environment", normalize_environment(self.environment))
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))


@dataclass(frozen=True)
class ExperimentConfiguration:
    """Reusable components and run settings; every run owns fresh driving and simulator state."""

    dataset: Dataset
    simulator: Simulator
    avstack: DrivingAdapter
    attacks: Mapping[str, Attack] = field(default_factory=dict)
    hooks: HookConfiguration = None
    evaluation: EvaluationConfiguration = field(default_factory=EvaluationConfiguration)
    runs: RunsConfiguration = field(default_factory=RunsConfiguration)
    # Loader-owned constructor descriptions. Keeping the source object alongside its options
    # prevents dataclasses.replace(attacks=...) from reusing another attack's description.
    _attack_specifications: Mapping[str, tuple[Attack, Mapping[str, Any]]] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        for name, expected in (("dataset", Dataset), ("simulator", Simulator),
                               ("avstack", DrivingAdapter), ("runs", RunsConfiguration),
                               ("evaluation", EvaluationConfiguration)):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        specifications = self.attacks.specifications if isinstance(self.attacks, _ConfiguredAttacks) else {}
        attacks = dict(self.attacks)
        for name, attack in attacks.items():
            _name(name, "Attack name")
            if not isinstance(attack, Attack):
                raise TypeError("An attack configuration must construct Attack")
        treatments = self.runs.treatments
        if not treatments:
            treatments = (TreatmentConfiguration("clean"),) + tuple(
                TreatmentConfiguration(name, name) for name in attacks
            )
            object.__setattr__(self, "runs", replace(self.runs, treatments=treatments))
        for item in treatments:
            if item.attack_name is not None and item.attack_name not in attacks:
                raise ValueError(f"Treatment {item.name!r} references unknown attack {item.attack_name!r}")
        if self.hooks is not None and not isinstance(self.hooks, HookSet) and not callable(self.hooks):
            raise TypeError("hooks must be HookSet or a seed-to-HookSet factory")
        configured_attacks = _ConfiguredAttacks(attacks, specifications)
        object.__setattr__(self, "attacks", configured_attacks)
        object.__setattr__(self, "_attack_specifications", configured_attacks.specifications)

    @property
    def configuration_summary(self) -> dict[str, Any]:
        """Describe compatibility on demand, when saving or loading prepared results.

        Constructing or running a configuration neither enumerates scenarios nor asks attacks
        to describe their parameters. Loader declarations retain their original options;
        directly supplied Python attacks describe their configured inputs on demand.
        """
        return self.describe_configuration()

    def describe_configuration(self, *, scenarios: Sequence[str] | None = None) -> dict[str, Any]:
        """Describe preparation inputs without requiring dataset access.

        Supplied scenario IDs avoid enumerating the dataset. Run settings are intentionally
        excluded from this compatibility summary.
        """
        descriptions = {}
        for name, attack in self.attacks.items():
            supplied = self._attack_specifications.get(name)
            try:
                parameters = supplied[1] if supplied is not None and supplied[0] is attack else (
                    attack.configuration_parameters()
                )
                if not isinstance(parameters, Mapping):
                    raise TypeError("Attack.configuration_parameters() must return a mapping")
                parameters = configuration_summary.summary_value(parameters)
            except (TypeError, ValueError) as error:
                warnings.warn(
                    f"Attack {name!r} parameters cannot be described: {error}. "
                    "Saving only its type and environment; parameter compatibility is unverified. "
                    "Override configuration_parameters() to enable validated loading.",
                    RuntimeWarning, stacklevel=2,
                )
                parameters = None
            description = {"type": configuration_summary.type_name(attack), "parameters": parameters,
                           "environment": configuration_summary.summary_value(attack.environment)}
            if parameters is None:
                description["parameters_verified"] = False
            descriptions[name] = description
        return {
            "dataset": {"type": configuration_summary.type_name(self.dataset),
                        "scenarios": list(scenarios) if scenarios is not None else
                        [scenario.id for scenario in self.dataset.scenarios()]},
            "simulator": configuration_summary.summary_value(self.simulator.identity),
            "attacks": descriptions,
        }


def load_configuration(
    config: ExperimentConfiguration | Mapping[str, Any] | str | Path,
) -> ExperimentConfiguration:
    """Load one experiment; file references are relative to their containing YAML file.

    Python mappings may supply Dataset, Simulator, DrivingAdapter, Attack and HookSet
    objects directly. User types are imported Python code and must be trusted.
    """
    if isinstance(config, ExperimentConfiguration):
        _attack_count(config.attacks)
        _validate_metric_configuration(config)
        return config
    spec, directory = _mapping(config, Path.cwd(), "Experiment")
    _keys(spec, {
        "dataset", "simulator", "avstack", "attacks", "hooks", "evaluation", "runs",
    }, "experiment")
    for key in ("dataset", "simulator", "avstack"):
        if key not in spec:
            raise ValueError(f"Experiment requires {key!r}")
    attack_specs = spec.get("attacks", {})
    if not isinstance(attack_specs, Mapping):
        raise TypeError("attacks must map names to Attack objects or constructor specifications")
    _attack_count(attack_specs)
    constructed_attacks = {name: _attack(value, directory) for name, value in attack_specs.items()}
    attacks = {name: pair[0] for name, pair in constructed_attacks.items()}
    attack_specifications = {
        name: (attack, parameters) for name, (attack, parameters) in constructed_attacks.items()
        if parameters is not None
    }
    runs = _runs(spec.get("runs", {}), directory)
    configuration = ExperimentConfiguration(
        dataset=_construct(spec["dataset"], directory, Dataset),
        simulator=_construct(spec["simulator"], directory, Simulator),
        avstack=_avstack(spec["avstack"], directory),
        attacks=_ConfiguredAttacks(attacks, attack_specifications),
        hooks=_hooks(spec.get("hooks"), directory),
        evaluation=_evaluation(spec.get("evaluation"), directory),
        runs=runs,
    )
    _validate_metric_configuration(configuration)
    return configuration


def _runs(value: Any, directory: Path) -> RunsConfiguration:
    if isinstance(value, RunsConfiguration):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("runs must be a mapping or RunsConfiguration")
    _keys(value, {"treatments", "seeds", "decision_steps", "environment", "recording", "output", "strict"},
          "runs")
    recording = value.get("recording", RecordingPolicy())
    if isinstance(recording, Mapping):
        _keys(recording, {"sensor_ids", "representations", "every_n_steps"}, "runs.recording")
        recording = RecordingPolicy(
            sensor_ids=_sequence(recording.get("sensor_ids", ()), "runs.recording.sensor_ids"),
            representations=_sequence(recording.get("representations", ("model_input",)),
                                      "runs.recording.representations"),
            every_n_steps=recording.get("every_n_steps", 1),
        )
    output = value.get("output")
    if output is not None:
        output = _path(output, directory)
    return RunsConfiguration(
        treatments=_treatments(value.get("treatments"), directory),
        seeds=_sequence(value.get("seeds", (0,)), "runs.seeds"),
        decision_steps=value.get("decision_steps"),
        environment=_configured_environment(value.get("environment", {}), directory),
        recording=recording, output=output, strict=value.get("strict", False),
    )


def _attack_count(attacks: Mapping[str, Any]) -> None:
    if len(attacks) != 1:
        raise ValueError("An experiment requires exactly one attack. Comparing multiple attacks "
                         "in one experiment is a future feature; to apply multiple actions "
                         "together, implement a composite Attack.")


def _validate_metric_configuration(configuration: ExperimentConfiguration) -> None:
    from avsectester.evaluation.metric import DetectionMetrics

    metrics = configuration.evaluation.metrics
    if metrics is None:
        return
    constructor = getattr(configuration.avstack.factory, "avstack_type", None)
    if (isinstance(constructor, type) and issubclass(constructor, AVStack)
            and constructor.provides_detections is False
            and any(isinstance(metric, DetectionMetrics) for metric in metrics)):
        raise ValueError(f"DetectionMetrics requires detections, but {constructor.__name__} "
                         "does not provide detection output")


def _evaluation(value: Any, directory: Path) -> EvaluationConfiguration:
    if value is None:
        return EvaluationConfiguration()
    if isinstance(value, EvaluationConfiguration):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("evaluation must be a mapping")
    _keys(value, {"metrics"}, "evaluation")
    metrics = value.get("metrics")
    return EvaluationConfiguration(
        metrics=None if metrics is None else tuple(
            _construct(item, directory, Metric) for item in _sequence(metrics, "evaluation.metrics")
        ),
    )


def _treatments(value: Any, directory: Path) -> tuple[TreatmentConfiguration, ...]:
    if value is None:
        return ()
    entries = ({"name": name, **dict(options)} for name, options in value.items()) if (
        isinstance(value, Mapping)
    ) else _sequence(value, "treatments")
    result = []
    for entry in entries:
        if isinstance(entry, TreatmentConfiguration):
            result.append(entry)
            continue
        if not isinstance(entry, Mapping):
            raise TypeError("Each treatment must be a mapping")
        _keys(entry, {"name", "attack_name", "environment"}, "treatment")
        options = dict(entry)
        options["environment"] = _configured_environment(options.get("environment", {}), directory)
        result.append(TreatmentConfiguration(**options))
    return tuple(result)


def _attack(value: Any, directory: Path) -> tuple[Attack, Mapping[str, Any] | None]:
    if isinstance(value, Attack):
        return value, None
    spec, directory = _mapping(value, directory, "Attack")
    _keys(spec, {"type", "options", "environment"}, "Attack")
    constructor, options = _constructor(spec, directory)
    attack = constructor(**options)
    if not isinstance(attack, Attack):
        raise TypeError("Attack constructor must return Attack")
    if "environment" in spec:
        attack.environment = _configured_environment(spec["environment"], directory)
    return attack, _options(spec.get("options", {}), directory, constructor, construct=False)


def _construct(value: Any, directory: Path, expected: type) -> Any:
    if isinstance(value, expected):
        return value
    spec, directory = _mapping(value, directory, expected.__name__)
    _keys(spec, {"type", "options"}, expected.__name__)
    constructor, options = _constructor(spec, directory)
    result = constructor(**options)
    if not isinstance(result, expected):
        raise TypeError(f"Constructor must return {expected.__name__}, got {type(result).__name__}")
    return result


def _avstack(value: Any, directory: Path) -> DrivingAdapter:
    """Build the per-run AVStack factory and its optional service bindings."""
    if isinstance(value, DrivingAdapter):
        return value
    spec, directory = _mapping(value, directory, "avstack")
    _keys(spec, {"type", "options", "observation_pipeline", "command_modes", "bindings"}, "avstack")
    constructor, options = _constructor(spec, directory)
    if isinstance(constructor, type) and not issubclass(constructor, AVStack):
        raise TypeError(f"avstack.type must name an AVStack class or a callable returning AVStack, "
                        f"got {constructor.__name__}")
    io_spec = spec.get("observation_pipeline")
    pipeline_constructor = None
    pipeline_options = {}
    pipeline_directory = directory
    if io_spec is not None and not isinstance(io_spec, ObservationPipeline):
        pipeline_spec, pipeline_directory = _mapping(io_spec, directory, "ObservationPipeline")
        _keys(pipeline_spec, {"type", "options"}, "ObservationPipeline")
        if "type" not in pipeline_spec:
            raise ValueError("A constructor requires type")
        pipeline_constructor = type_registry.resolve_type(pipeline_spec["type"])
        pipeline_options = _options(pipeline_spec.get("options", {}), pipeline_directory,
                                    pipeline_constructor, construct=False)
    modes = spec.get("command_modes")
    if modes is not None:
        modes = frozenset(_sequence(modes, "command_modes"))
        if not modes or modes - {"actuator", "trajectory"}:
            raise ValueError("command_modes must contain actuator and/or trajectory")

    def factory(seed: int) -> AVStack:
        # AVStack constructors receive only their configured options; the run seed controls
        # hooks, sensor environments and the simulator.
        avstack = constructor(**deepcopy(options))
        if not isinstance(avstack, AVStack):
            raise TypeError("avstack.type must construct an AVStack")
        try:
            if io_spec is not None:
                pipeline = (io_spec if pipeline_constructor is None else
                            pipeline_constructor(**_options(pipeline_options, pipeline_directory,
                                                            pipeline_constructor)))
                if not isinstance(pipeline, ObservationPipeline):
                    raise TypeError("Constructor must return ObservationPipeline, "
                                    f"got {type(pipeline).__name__}")
                avstack.observation_pipeline = pipeline
            if modes is not None and avstack.command_modes != modes:
                raise ValueError("Configured command_modes differs from the AVStack declaration")
            return avstack
        except BaseException as error:
            try:
                avstack.close()
            except BaseException as cleanup_error:
                error.add_note(f"AVStack cleanup failed: {cleanup_error}")
            raise

    # Static metadata enables validation without constructing a model or calling user code.
    factory.avstack_type = constructor  # type: ignore[attr-defined]
    bindings = spec.get("bindings")
    if bindings is None:
        return DrivingAdapter(factory)
    if not isinstance(bindings, Mapping):
        raise TypeError("avstack.bindings must be a mapping with type and options")
    _keys(bindings, {"type", "options"}, "avstack.bindings")
    bind, arguments = _constructor(bindings, directory)
    adapter = bind(factory, **arguments)
    if not isinstance(adapter, DrivingAdapter):
        raise TypeError("avstack.bindings must return DrivingAdapter")
    return adapter


def _hooks(value: Any, directory: Path) -> HookConfiguration:
    if value is None or isinstance(value, HookSet) or callable(value):
        return value
    entries = _sequence(value, "hooks")
    constructors: list[Callable[[], Hook]] = []
    for entry in entries:
        if isinstance(entry, Hook):
            def supplied_hook(entry: Hook = entry) -> Hook:
                return entry

            constructors.append(supplied_hook)
            continue
        if not isinstance(entry, Mapping):
            raise TypeError("Each hook must be Hook or a mapping")
        _keys(entry, {"stage", "type", "options", "native_hook", "imports"}, "hook")
        stage = _name(entry.get("stage"), "Hook stage")
        imports = _sequence(entry.get("imports", ()), "hook imports")
        for module in imports:
            if not isinstance(module, str) or not module:
                raise ValueError("Hook imports must be module names")
            import_module(module)
        if ("type" in entry) == ("native_hook" in entry):
            raise ValueError("A hook requires exactly one of type and native_hook")
        if "native_hook" in entry:
            if "options" in entry or not isinstance(entry["native_hook"], Mapping):
                raise ValueError("native_hook takes the avstack hook configuration, without options")
            phase = stage.rsplit(".", 1)[-1]
            if phase not in {"pre", "post"}:
                raise ValueError("Native hooks require a .pre or .post stage")
            native_spec = deepcopy(entry["native_hook"])
            def native_hook(
                stage: str = stage, spec: Mapping[str, Any] = native_spec, phase: str = phase,
            ) -> Hook:
                return Hook(stage, NativeHook(spec, phase=phase))

            constructors.append(native_hook)
        else:
            constructor, options = _constructor(entry, directory)
            def user_hook(
                stage: str = stage, constructor: Callable[..., Any] = constructor,
                options: dict[str, Any] = options,
            ) -> Hook:
                return Hook(stage, constructor(**deepcopy(options)))

            constructors.append(user_hook)

    def factory(seed: int) -> HookSet:
        return HookSet((constructor() for constructor in constructors), seed=seed)

    return factory


def _mapping(value: Any, directory: Path, label: str) -> tuple[dict[str, Any], Path]:
    if isinstance(value, (str, Path)):
        path = _path(value, directory)
        with path.open(encoding="utf-8") as stream:
            value = yaml.safe_load(stream)
        directory = path.parent
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping or YAML path")
    return dict(value), directory


def _constructor(spec: Mapping[str, Any], directory: Path) -> tuple[Callable[..., Any], dict]:
    if "type" not in spec:
        raise ValueError("A constructor requires type")
    constructor = type_registry.resolve_type(spec["type"])
    return constructor, _options(spec.get("options", {}), directory, constructor)


def _parameter_annotations(constructor: Callable[..., Any]) -> dict[str, Any]:
    """Resolve each accepted parameter independently, without requiring unrelated hints."""
    signature_source = getattr(constructor, "__init__") if isinstance(constructor, type) else constructor
    try:
        parameters = inspect.signature(signature_source).parameters
    except (TypeError, ValueError):
        return {}
    source: Any = constructor
    while isinstance(source, partial):
        source = source.func
    if isinstance(source, type):
        source = getattr(source, "__init__")
    elif not inspect.isroutine(source):
        source = getattr(source, "__call__")
    source = inspect.unwrap(source)
    globalns = getattr(source, "__globals__", {})
    localns = dict(vars(constructor)) if isinstance(constructor, type) else {}
    if isinstance(constructor, type):
        localns[constructor.__name__] = constructor
    annotations = {}
    for name, parameter in parameters.items():
        annotation = parameter.annotation
        if (annotation is inspect.Parameter.empty or
                parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}):
            continue
        try:
            annotations[name] = get_type_hints(
                SimpleNamespace(__annotations__={name: annotation}),
                globalns=globalns, localns=localns, include_extras=True,
            )[name]
        except Exception as error:
            if "Path" in str(annotation):
                raise ValueError(
                    f"Cannot resolve path annotation for {constructor!r}.{name}: {annotation}. "
                    "Import Path or os.PathLike at runtime in the defining module."
                ) from error
    return annotations


def _resolve_path_argument(value: Any, annotation: Any, directory: Path) -> Any:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Annotated:
        return _resolve_path_argument(value, arguments[0], directory)
    if origin in {Union, UnionType}:
        for alternative in arguments:
            value = _resolve_path_argument(value, alternative, directory)
        return value
    path_type = origin or annotation
    if isinstance(path_type, type) and issubclass(path_type, PathLike):
        if isinstance(value, (str, Path)):
            if isinstance(value, str) and "://" in value:
                return value
            return str(_path(value, directory))
        return value
    if origin in {Sequence, list, tuple} and isinstance(value, (list, tuple)) and arguments:
        annotations = (arguments if origin is tuple and arguments[-1] is not Ellipsis else
                       (arguments[0],) * len(value))
        values = [_resolve_path_argument(item, annotation, directory)
                  for item, annotation in zip(value, annotations)]
        values.extend(deepcopy(value[len(values):]))
        return tuple(values) if isinstance(value, tuple) else values
    return value


def _options(value: Any, directory: Path, constructor: Callable[..., Any], *,
             construct: bool = True) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Constructor options must be a mapping")
    annotations = _parameter_annotations(constructor)

    def resolve(item: Any) -> Any:
        if isinstance(item, Mapping):
            # A nested mapping is a constructor when its type is a built-in class name or user
            # code. Native configurations whose type is another registry's name pass through.
            kind = item.get("type")
            if isinstance(kind, str) and (type_registry.is_builtin_type_name(kind) or ":" in kind):
                _keys(item, {"type", "options"}, "nested constructor")
                nested_constructor = type_registry.resolve_type(kind)
                arguments = _options(item.get("options", {}), directory, nested_constructor,
                                     construct=construct)
                if not construct:
                    return {"type": item["type"], "options": arguments}
                return nested_constructor(**arguments)
            return {name: resolve(entry) for name, entry in item.items()}
        if isinstance(item, (list, tuple)):
            values = [resolve(entry) for entry in item]
            return tuple(values) if isinstance(item, tuple) else values
        return deepcopy(item)

    return {key: _resolve_path_argument(resolve(item), annotations.get(key), directory)
            for key, item in value.items()}


def _path(value: str | PathLike[str], directory: Path) -> Path:
    path = Path(value).expanduser()
    return (directory / path).resolve() if not path.is_absolute() else path


def _environment_override(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a layer without turning an omitted sensors key into an explicit removal."""
    from avsectester.closed_loop.environment import normalize_environment

    normalized = normalize_environment(value)
    return {key: normalized[key] for key in value}


def _configured_environment(value: Mapping[str, Any] | None, directory: Path) -> dict[str, Any]:
    """Resolve corruption constructor arguments while retaining the environment declaration."""
    from avsectester.closed_loop.sensor_corruptions import CORRUPTIONS

    result = _environment_override({} if value is None else value)
    if "sensors" in result:
        for specification in result["sensors"]:
            kind = specification["type"]
            constructor = (type_registry.resolve_type(kind) if ":" in kind else CORRUPTIONS[kind])
            specification["options"] = _options(specification["options"], directory, constructor)
    return result


def _keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {label} keys: {sorted(unknown)}")


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"{label} must be a nonempty name without path separators")
    return value


def _sequence(value: Any, label: str) -> tuple:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a sequence")
    return tuple(value)
