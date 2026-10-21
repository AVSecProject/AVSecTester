"""Construct experiment objects from Python mappings or YAML files.

An experiment declares ``dataset``, ``simulator``, ``avstack``, named ``attacks`` and
``treatments``. Attack and driving specifications can be inline mappings or YAML paths.
Every constructed object is selected by one key, ``type``, with keyword arguments in
``options``: a value without a colon names a built-in class (``type: NuRecSimulator``), and
``package.module:name`` names user code (``type: my_project.driving:MyAVStack``). Known
filesystem arguments are resolved relative to the file containing them. Other constructor
arguments remain user-owned.

``avstack.type`` names an AVStack class or a callable returning an AVStack. It is called with
``options`` for every run and service operation, so each run has a fresh driving history.
Optional ``avstack.bindings`` exposes driving-system components and scorers to closed-loop
services.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
from importlib import import_module
import math
import re
from pathlib import Path
from typing import Any, TypeAlias
import warnings

import yaml  # type: ignore[import-untyped]

from avsectester.attack import Attack
from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.closed_loop.avstacks.native_hook import NativeHook
from avsectester.closed_loop.runner import RecordingPolicy
from avsectester.closed_loop.services import DrivingAdapter
from avsectester.closed_loop.simulator import Simulator
from avsectester.closed_loop.vehicle_io import VehicleIO
from avsectester.datasets import Dataset
from avsectester.evaluation.report import EvaluationConfiguration
from avsectester.evaluation.metric import Metric


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
class ExperimentConfiguration:
    """Reusable configuration. Each run creates its own driving and simulator instances."""

    dataset: Dataset
    simulator: Simulator
    avstack: DrivingAdapter
    attacks: Mapping[str, Attack] = field(default_factory=dict)
    treatments: tuple[TreatmentConfiguration, ...] = ()
    seeds: tuple[int, ...] = (0,)
    decision_steps: int | None = None
    environment: Mapping[str, Any] = field(default_factory=dict)
    recording: RecordingPolicy = field(default_factory=RecordingPolicy)
    evaluation: EvaluationConfiguration = field(default_factory=EvaluationConfiguration)
    output: Path | None = None
    strict: bool = False
    hooks: HookConfiguration = None
    # Loader-owned constructor descriptions. Keeping the source object alongside its options
    # prevents dataclasses.replace(attacks=...) from reusing another attack's description.
    _attack_specifications: Mapping[str, tuple[Attack, Mapping[str, Any]]] = field(
        default_factory=dict, init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        from avsectester.closed_loop.environment import normalize_environment

        for name, expected in (("dataset", Dataset), ("simulator", Simulator),
                               ("avstack", DrivingAdapter), ("recording", RecordingPolicy),
                               ("evaluation", EvaluationConfiguration)):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        specifications = self.attacks.specifications if isinstance(self.attacks, _ConfiguredAttacks) else {}
        attacks = dict(self.attacks)
        for name, attack in attacks.items():
            _name(name, "Attack name")
            if not isinstance(attack, Attack):
                raise TypeError("An attack configuration must construct Attack")
        treatments = tuple(self.treatments)
        if not treatments:
            treatments = (TreatmentConfiguration("clean"),) + tuple(
                TreatmentConfiguration(name, name) for name in attacks
            )
        if any(not isinstance(item, TreatmentConfiguration) for item in treatments):
            raise TypeError("treatments must contain TreatmentConfiguration values")
        if len({item.name for item in treatments}) != len(treatments):
            raise ValueError("Treatment names must be unique")
        for item in treatments:
            if item.attack_name is not None and item.attack_name not in attacks:
                raise ValueError(f"Treatment {item.name!r} references unknown attack {item.attack_name!r}")
        if not self.seeds or any(isinstance(seed, bool) or not isinstance(seed, int)
                                 or seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be a nonempty sequence of nonnegative integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must not contain duplicates")
        if self.decision_steps is not None and (
            isinstance(self.decision_steps, bool) or not isinstance(self.decision_steps, int)
            or self.decision_steps < 1
        ):
            raise ValueError("decision_steps must be a positive integer or None")
        if not isinstance(self.strict, bool):
            raise TypeError("strict must be boolean")
        if self.hooks is not None and not isinstance(self.hooks, HookSet) and not callable(self.hooks):
            raise TypeError("hooks must be HookSet or a seed-to-HookSet factory")
        configured_attacks = _ConfiguredAttacks(attacks, specifications)
        object.__setattr__(self, "attacks", configured_attacks)
        object.__setattr__(self, "_attack_specifications", configured_attacks.specifications)
        object.__setattr__(self, "treatments", treatments)
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "environment", normalize_environment(self.environment))
        if self.output is not None:
            object.__setattr__(self, "output", Path(self.output))

    @property
    def configuration_summary(self) -> dict[str, Any]:
        """Describe compatibility on demand, when saving or loading prepared results.

        Constructing or running a configuration neither enumerates scenarios nor asks attacks
        to describe their parameters. Loader declarations retain their original options;
        directly supplied Python attacks describe their configured inputs on demand.
        """
        return self.describe_configuration()

    def describe_configuration(self, *, scenarios: Sequence[str] | None = None) -> dict[str, Any]:
        """Describe preparation inputs; supplied scenario IDs avoid accessing a dataset."""
        descriptions = {}
        for name, attack in self.attacks.items():
            supplied = self._attack_specifications.get(name)
            try:
                parameters = supplied[1] if supplied is not None and supplied[0] is attack else (
                    attack.configuration_parameters()
                )
                if not isinstance(parameters, Mapping):
                    raise TypeError("Attack.configuration_parameters() must return a mapping")
                parameters = _summary_value(parameters)
            except (TypeError, ValueError) as error:
                warnings.warn(
                    f"Attack {name!r} parameters cannot be described: {error}. "
                    "Saving only its type and environment; parameter compatibility is unverified. "
                    "Override configuration_parameters() to enable validated loading.",
                    RuntimeWarning, stacklevel=2,
                )
                parameters = None
            description = {"type": _type_name(attack), "parameters": parameters,
                           "environment": _summary_value(attack.environment)}
            if parameters is None:
                description["parameters_verified"] = False
            descriptions[name] = description
        return {
            "dataset": {"type": _type_name(self.dataset),
                        "scenarios": list(scenarios) if scenarios is not None else
                        [scenario.id for scenario in self.dataset.scenarios()]},
            "simulator": _summary_value(self.simulator.identity),
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
        "dataset", "simulator", "avstack", "attacks", "treatments", "seeds", "decision_steps",
        "environment", "recording", "evaluation", "output", "strict", "hooks",
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
    treatments = _treatments(spec.get("treatments"))
    recording = spec.get("recording", RecordingPolicy())
    if isinstance(recording, Mapping):
        _keys(recording, {"sensor_ids", "representations", "every_n_steps"}, "recording")
        recording = RecordingPolicy(
            sensor_ids=_sequence(recording.get("sensor_ids", ()), "recording.sensor_ids"),
            representations=_sequence(recording.get("representations", ("model_input",)),
                                      "recording.representations"),
            every_n_steps=recording.get("every_n_steps", 1),
        )
    output = spec.get("output")
    if output is not None:
        output = _path(output, directory)
    configuration = ExperimentConfiguration(
        dataset=_construct(spec["dataset"], directory, Dataset),
        simulator=_construct(spec["simulator"], directory, Simulator),
        avstack=_avstack(spec["avstack"], directory),
        attacks=_ConfiguredAttacks(attacks, attack_specifications), treatments=treatments,
        seeds=_sequence(spec.get("seeds", (0,)), "seeds"),
        decision_steps=spec.get("decision_steps"), environment=spec.get("environment", {}),
        recording=recording, evaluation=_evaluation(spec.get("evaluation"), directory),
        output=output, strict=spec.get("strict", False),
        hooks=_hooks(spec.get("hooks"), directory),
    )
    _validate_metric_configuration(configuration)
    return configuration


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


def _treatments(value: Any) -> tuple[TreatmentConfiguration, ...]:
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
        result.append(TreatmentConfiguration(**dict(entry)))
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
        attack.environment = _environment_override(spec["environment"])
    return attack, _options(spec.get("options", {}), directory, construct=False)


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
    _keys(spec, {"type", "options", "vehicle_io", "command_modes", "bindings"}, "avstack")
    constructor, options = _constructor(spec, directory)
    if isinstance(constructor, type) and not issubclass(constructor, AVStack):
        raise TypeError(f"avstack.type must name an AVStack class or a callable returning AVStack, "
                        f"got {constructor.__name__}")
    io_spec = spec.get("vehicle_io")
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
                avstack.vehicle_io = _construct(io_spec, directory, VehicleIO)
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
    return resolve_type(spec["type"]), _options(spec.get("options", {}), directory)


def resolve_type(value: Any) -> Callable[..., Any]:
    """Resolve a built-in class name, or user code written as ``package.module:name``."""
    if not isinstance(value, str) or not value:
        raise ValueError("type must be a built-in class name or 'package.module:name'")
    return _import(value) if ":" in value else _builtin(value)


def _import(value: str) -> Callable[..., Any]:
    if value.count(":") != 1:
        raise ValueError(f"type {value!r} must be 'package.module:name'")
    module, name = value.split(":")
    if not module or not name:
        raise ValueError(f"type {value!r} must be 'package.module:name'")
    result = import_module(module)
    for part in name.split("."):
        result = getattr(result, part)
    if not callable(result):
        raise TypeError(f"type {value!r} is not callable")
    return result


_BUILTINS = {
    "SignAsset": "avsectester.attack.implemented_attacks.object_insertion_attack.assets:SignAsset",
    "PersonAsset": "avsectester.attack.implemented_attacks.object_insertion_attack.assets:PersonAsset",
    "TrafficLightAsset": "avsectester.attack.implemented_attacks.object_insertion_attack.assets:TrafficLightAsset",
    "CheckerboardTexture": "avsectester.attack.insertion.assets:CheckerboardTexture",
    "ImageTexture": "avsectester.attack.insertion.assets:ImageTexture",
    "PatchCompositor": "avsectester.attack.insertion.compositor:PatchCompositor",
    "ClassicHarmonizer": "avsectester.attack.insertion.harmonizers:ClassicHarmonizer",
    "PCTNetHarmonizer": "avsectester.attack.insertion.harmonizers:PCTNetHarmonizer",
    "PatchOptimization": "avsectester.attack.implemented_attacks.patch_attack.offline:PatchOptimization",
    "PGD": "avsectester.attack.optimization.whitebox:PGD",
    "NES": "avsectester.attack.optimization.blackbox:NES",
    "AttachedPlacement": "avsectester.attack.placement:AttachedPlacement",
    "WorldPlacement": "avsectester.attack.placement:WorldPlacement",
    "Orientation": "avsectester.attack.placement:Orientation",
    "CameraForwardCollisionAVStack": "avsectester.closed_loop.avstacks.camera_forward_collision:CameraForwardCollisionAVStack",
    "DrivingMetrics": "avsectester.evaluation.metric:DrivingMetrics",
    "DetectionMetrics": "avsectester.evaluation.metric:DetectionMetrics",
    "TrajectoryDeviation": "avsectester.evaluation.metric:TrajectoryDeviation",
    "StopImpact": "avsectester.evaluation.metric:StopImpact",
    "PlaneAsset": "avsectester.attack.insertion:PlaneAsset",
    "PlaneSurface": "avsectester.attack.insertion:PlaneSurface",
    "Insertion": "avsectester.attack.insertion:Insertion",

    "CarlaDataset": "avsectester.datasets.carla:CarlaDataset",
    "CarlaRecipe": "avsectester.datasets.carla:CarlaRecipe",
    "PrescribedMotion": "avsectester.datasets.carla:PrescribedMotion",
    "NuRecDataset": "avsectester.datasets.nurec:NuRecDataset",
    "NuScenesDataset": "avsectester.datasets.nuscenes:NuScenesDataset",
    "CarlaSimulator": "avsectester.closed_loop.simulators.carla:CarlaSimulator",
    "NuRecSimulator": "avsectester.closed_loop.simulators.nurec:NuRecSimulator",
    "RecordedReplaySimulator": "avsectester.closed_loop.simulators.recorded:RecordedReplaySimulator",
    "StubRenderer": "avsectester.closed_loop.simulators.nurec:StubRenderer",
    "NuRecRenderer": "avsectester.closed_loop.simulators.nurec:NuRecRenderer",
    "KinematicBicycle": "avsectester.closed_loop.simulators.nurec:KinematicBicycle",
    "TrajectoryFollower": "avsectester.closed_loop.simulators.nurec:TrajectoryFollower",
    "ModularAVStack": "avsectester.closed_loop.avstacks.modular:ModularAVStack",
    "AlpamayoAVStack": "avsectester.closed_loop.avstacks.alpamayo:AlpamayoAVStack",
    "VehicleIO": "avsectester.closed_loop.vehicle_io:VehicleIO",
    "Control": "avsectester.closed_loop.datatypes:Control",
    "ActorState": "avsectester.datasets.base:ActorState",
}


def attack_directory_name(class_name: str) -> str:
    """Convert PascalCase to snake_case, preserving acronym groups (MyNNAttack -> my_nn_attack)."""
    words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", class_name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", words).lower()


def _attack_class_name(name: str) -> bool:
    return name.isidentifier() and name[0].isupper()


def _exported_attack(module: Any, name: str, folder: str) -> type[Attack] | None:
    """Apply the same name/class/export/folder rule to lookup and discovery."""
    attack = getattr(module, name, None)
    if (not _attack_class_name(name) or not isinstance(attack, type)
            or not issubclass(attack, Attack) or attack is Attack
            or attack.__name__ != name or attack_directory_name(name) != folder):
        return None
    return attack


def _implemented_attack(value: str) -> Callable[..., Any] | None:
    package = import_module("avsectester.attack.implemented_attacks")
    if not _attack_class_name(value):
        return None
    folder = attack_directory_name(value)
    if not any((Path(root) / folder / "__init__.py").is_file() for root in package.__path__):
        return None
    module = import_module(f"{package.__name__}.{folder}")
    attack = _exported_attack(module, value, folder)
    if attack is None:
        raise TypeError(f"{module.__name__} must export the Attack subclass {value}")
    return attack


def _discover_attacks() -> tuple[tuple[str, ...], dict[str, str]]:
    """Return accepted attack class names and the folders that failed to import."""
    package = import_module("avsectester.attack.implemented_attacks")
    names: set[str] = set()
    failures: dict[str, str] = {}
    for root in package.__path__:
        for source in Path(root).glob("*/__init__.py"):
            folder = source.parent.name
            try:
                module = import_module(f"{package.__name__}.{folder}")
            except Exception as error:
                failures[folder] = f"{type(error).__name__}: {error}"
                continue
            names.update(name for name in vars(module)
                         if _exported_attack(module, name, folder) is not None)
    return tuple(sorted(names)), failures


def available_attacks() -> tuple[str, ...]:
    """List exactly the exported Attack classes accepted by class-name lookup.

    Attack folders that fail to import are omitted; an unknown type's error message names them.
    """
    return _discover_attacks()[0]


def _builtin(value: str) -> Callable[..., Any]:
    if value in _BUILTINS:
        return _import(_BUILTINS[value])
    attack = _implemented_attack(value)
    if attack is not None:
        return attack
    names, failures = _discover_attacks()
    message = (f"Unknown built-in type {value!r}. Use an exact class name or "
               f"'package.module:name'. Available attacks: {', '.join(names) or 'none'}")
    if failures:
        message += ". Attack folders that failed to import: " + "; ".join(
            f"{folder} ({reason})" for folder, reason in sorted(failures.items()))
    raise ValueError(message)


_PATH_KEYS = {
    "face", "init_texture", "path", "directory", "checkpoint_path", "checkpoint", "dataroot", "usdz_path", "usdz_paths",
    "image", "texture", "asset_path", "image_path", "texture_path", "output_path", "model_path", "config_path",
}


def _options(value: Any, directory: Path, *, construct: bool = True) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Constructor options must be a mapping")

    def resolve(item: Any, key: str | None = None) -> Any:
        if isinstance(item, Mapping):
            # A nested mapping is a constructor when its type is a built-in class name or user
            # code. Native configurations whose type is another registry's name pass through.
            kind = item.get("type")
            if isinstance(kind, str) and (kind in _BUILTINS or ":" in kind):
                _keys(item, {"type", "options"}, "nested constructor")
                if not construct:
                    return {"type": item["type"],
                            "options": _options(item.get("options", {}), directory, construct=False)}
                constructor, arguments = _constructor(item, directory)
                return constructor(**arguments)
            return {name: resolve(entry, name) for name, entry in item.items()}
        if isinstance(item, (list, tuple)):
            values = [resolve(entry, key) for entry in item]
            return tuple(values) if isinstance(item, tuple) else values
        if key in _PATH_KEYS and isinstance(item, (str, Path)):
            # URI/model identifiers are not local filesystem references.
            if isinstance(item, str) and "://" in item:
                return item
            return str(_path(item, directory))
        return deepcopy(item)

    return {key: resolve(item, key) for key, item in value.items()}


def _path(value: str | Path, directory: Path) -> Path:
    path = Path(value).expanduser()
    return (directory / path).resolve() if not path.is_absolute() else path


def _environment_override(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a layer without turning an omitted sensors key into an explicit removal."""
    from avsectester.closed_loop.environment import normalize_environment

    normalized = normalize_environment(value)
    return {key: normalized[key] for key in value}


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


def _type_name(value: Any) -> str:
    return f"{type(value).__module__}:{type(value).__qualname__}"


def _summary_value(value: Any) -> Any:
    """Summarize configuration data; never inspect arbitrary object state or live handles."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Configuration summaries require finite numbers")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Configuration parameter mappings require string keys")
        return {key: _summary_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_summary_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {"type": _type_name(value), "parameters": {
            item.name: _summary_value(getattr(value, item.name)) for item in fields(value) if item.init
        }}
    import numpy as np

    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Configuration arrays must not contain Python objects")
        return {"array_dtype": value.dtype.str, "array_shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return _summary_value(value.item())
    raise TypeError(f"Cannot summarize configuration parameter {type(value).__name__}; "
                    "override Attack.configuration_parameters() with a data description")
