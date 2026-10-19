"""Prepare one configured scenario or enumerate cases through the selection interfaces.

The existing flat YAML layout is preserved. Backend settings and intervention settings are
separate views of that layout, so scene factories never consume attack/defense configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any, TYPE_CHECKING

from avsectester.conditions import ScenarioConditions
from avsectester.scenarios.source import (
    CarlaScenarioBuilder,
    DatasetFilter,
    ScenarioInstance,
    ScenarioSource,
    _check_limit,
)

if TYPE_CHECKING:
    from avsectester.closed_loop.simulator import WorldBackend
    from avsectester.scenarios.requirement import ScenarioRequirement
    from avsectester.scenarios.source import Dataset


# Scene adapters receive only their own settings. Model and intervention assembly retain
# the original mapping separately, including a CARLA pipeline stored under ego.pipeline.
_RUN_KEYS = frozenset({
    "frames", "stack", "attacks", "patches", "defenses", "interventions",
    "runtime", "clean_runtime", "conditions", "selection", "mode", "treatment", "output",
})


def scene_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached scene/backend view of the existing flat experiment configuration."""
    if not isinstance(config, Mapping):
        raise ValueError("Scenario configuration must be a mapping")
    return deepcopy({key: value for key, value in config.items() if key not in _RUN_KEYS})


def _configuration(config: Mapping[str, Any]) -> tuple[dict[str, Any], ScenarioConditions]:
    scene = scene_config(config)
    spec = deepcopy(config.get("conditions", {}))
    if not isinstance(spec, Mapping):
        raise ValueError("conditions must be a mapping")
    spec = dict(spec)
    legacy_weather = scene.pop("weather", None)
    if legacy_weather is not None:
        world = spec.get("world", {})
        if not isinstance(world, Mapping):
            raise ValueError("World conditions must be a mapping")
        if "weather" in world and world["weather"] != legacy_weather:
            raise ValueError("weather and conditions.world.weather disagree")
        spec["world"] = {**world, "weather": legacy_weather}
    return scene, ScenarioConditions.from_config(spec)


class ScenarioConfigurationError(ValueError):
    """The configured built-in backend cannot be constructed from its settings."""


def _backend_factory(config: dict[str, Any]) -> WorldBackend:
    try:
        return _build_backend(config)
    except (ValueError, TypeError, KeyError) as error:
        raise ScenarioConfigurationError(str(error)) from error


def _build_backend(config: dict[str, Any]) -> WorldBackend:
    spec = config.get("backend", {})
    if not isinstance(spec, Mapping):
        raise ValueError("backend must be a mapping")
    kind = spec.get("type", "CarlaSimulator")
    if kind == "CarlaSimulator":
        if set(spec) - {"type"}:
            raise ValueError("CARLA settings belong in client/ego/npcs/lead, not backend")
        from avsectester.closed_loop.simulators.carla import CarlaSimulator

        return CarlaSimulator(config).create_runtime()
    if kind != "NuRecSimulator":
        raise ValueError(
            f"Unknown backend type {kind!r}: expected CarlaSimulator or NuRecSimulator. "
            "Supply backend_factory for a custom adapter."
        )
    unknown = set(spec) - {"type", "scene", "renderer", "dynamics", "ego0", "dt", "camera_aliases"}
    if unknown:
        raise ValueError(f"Unknown NuRec backend options: {sorted(unknown)}")
    from avsectester.closed_loop.simulators.nurec import (
        KinematicBicycle,
        NuRecSimulator,
        NuRecRenderer,
        StubRenderer,
        TrajectoryFollower,
    )

    backend_config = {key: deepcopy(value) for key, value in config.items() if key != "backend"}
    backend_config.update({key: deepcopy(value) for key, value in spec.items() if key != "type"})
    renderer_spec = backend_config.pop("renderer", "StubRenderer")
    if isinstance(renderer_spec, str):
        renderer_spec = {"type": renderer_spec}
    if not isinstance(renderer_spec, Mapping):
        raise ValueError("NuRec renderer must be 'StubRenderer', 'NuRecRenderer', or a mapping")
    renderer_options = dict(renderer_spec)
    renderer_type = renderer_options.pop("type", "StubRenderer")
    if renderer_type == "StubRenderer":
        renderer = StubRenderer(**renderer_options)
    elif renderer_type == "NuRecRenderer":
        renderer = NuRecRenderer(**renderer_options)
    else:
        raise ValueError(
            f"Unknown NuRec renderer type {renderer_type!r}: expected StubRenderer or NuRecRenderer"
        )
    dynamics_options = dict(backend_config.pop("dynamics", {}))
    dynamics_type = dynamics_options.pop("type", "KinematicBicycle")
    try:
        if dynamics_type == "TrajectoryFollower":
            dynamics = TrajectoryFollower(**dynamics_options)
        elif dynamics_type == "KinematicBicycle":
            dynamics = KinematicBicycle(**dynamics_options)
        else:
            raise ValueError(
                f"Unknown NuRec dynamics type {dynamics_type!r}: "
                "expected KinematicBicycle or TrajectoryFollower"
            )
        return NuRecSimulator(backend_config, renderer=renderer, dynamics=dynamics).create_runtime()
    except BaseException:
        renderer.close()
        raise


class ConfiguredScenarioSource(ScenarioSource):
    """Yield one configured scenario, without preview, role inference or filter evaluation."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        backend_factory: Callable[[dict[str, Any]], WorldBackend] | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> None:
        self.config, self.conditions = _configuration(config)
        self.backend_factory = backend_factory or _backend_factory
        if not callable(self.backend_factory):
            raise TypeError("backend_factory must be callable")
        self.provenance = deepcopy(dict(provenance or {}))
        self.selection_log: list[dict[str, Any]] = []

    def scenarios(self, req: ScenarioRequirement | None = None, limit: int | None = None):
        _check_limit(limit)
        self.selection_log = []
        if req is not None:
            raise ValueError("ConfiguredScenarioSource does not evaluate filters. Use a selection source.")
        if limit == 0:
            return
        # No backend is constructed until a consumer invokes the case's factory.
        yield ScenarioInstance(
            lambda: self.backend_factory(deepcopy(self.config)),
            None, deepcopy(self.provenance), {},
            deepcopy(self.config), self.conditions,
        )


class _ConfiguredSelectionSource(ScenarioSource):
    def __init__(self, source, requirement, conditions, config, provenance, backend_factory=None):
        self.source, self.requirement = source, requirement
        self.conditions, self.config, self.provenance = conditions, config, provenance
        self.backend_factory = backend_factory
        self.selection_log = []

    def scenarios(self, req=None, limit=None):
        if req is not None and req is not self.requirement:
            raise ValueError("This source already has a configured selection requirement")
        try:
            for instance in self.source.scenarios(self.requirement, limit=limit):
                if self.backend_factory is not None:
                    prepared_config = deepcopy(instance.config or self.config)
                    instance = replace(instance, factory=lambda c=prepared_config: self.backend_factory(deepcopy(c)))
                yield replace(instance, provenance={
                    **instance.provenance, **deepcopy(self.provenance),
                })
        finally:
            self.selection_log = self.source.selection_log


def source_from_config(
    config: Mapping[str, Any],
    *,
    backend_factory: Callable[[dict[str, Any]], WorldBackend] | None = None,
    provenance: Mapping[str, Any] | None = None,
    candidate_provider: Callable | None = None,
    dataset: Dataset | None = None,
) -> ScenarioSource:
    """Create a pass-through or explicitly configured selection source.

    ``selection`` accepts ``requirement`` (a serialized ScenarioRequirement) and ``source``
    (CarlaScenarioBuilder options). Python datasets use DatasetFilter. A custom CARLA
    candidate provider must prepare the supplied weather setting before yielding its context.
    """
    scene, conditions = _configuration(config)
    selection = config.get("selection")
    if selection is None:
        return ConfiguredScenarioSource(config, backend_factory=backend_factory, provenance=provenance)
    if not isinstance(selection, Mapping) or set(selection) - {"requirement", "source"}:
        raise ValueError("selection requires 'requirement' and optional 'source' options")
    from avsectester.scenarios.requirement import ScenarioRequirement
    from avsectester.scenarios.serialize import requirement_from_dict

    requirement = selection.get("requirement")
    if isinstance(requirement, Mapping):
        requirement = requirement_from_dict(dict(requirement))
    if not isinstance(requirement, ScenarioRequirement):
        raise ValueError("selection.requirement must describe a ScenarioRequirement")
    source_options = selection.get("source", {})
    if not isinstance(source_options, Mapping):
        raise ValueError("selection.source must be a mapping")
    if dataset is not None:
        if source_options or backend_factory is not None:
            raise ValueError("Dataset selection uses its provider's backend factory and context")
        source = DatasetFilter(dataset, conditions=conditions)
    else:
        backend_kind = scene.get("backend", {}).get("type", "CarlaSimulator")
        if backend_kind != "CarlaSimulator":
            raise ValueError("Recorded NuRec selection requires a Dataset. Direct NuRec config needs no selection.")
        source = CarlaScenarioBuilder(
            base_scenario=scene,
            candidate_provider=candidate_provider,
            conditions=conditions,
            **dict(source_options),
        )
    return _ConfiguredSelectionSource(
        source, requirement, conditions, scene, dict(provenance or {}), backend_factory,
    )


def scenario_from_config(
    config: Mapping[str, Any],
    *,
    backend_factory: Callable[[dict[str, Any]], WorldBackend] | None = None,
    provenance: Mapping[str, Any] | None = None,
    candidate_provider: Callable | None = None,
    dataset: Dataset | None = None,
) -> ScenarioInstance:
    """Prepare one configured/selected case, releasing selection resources before returning."""
    source = source_from_config(
        config, backend_factory=backend_factory, provenance=provenance,
        candidate_provider=candidate_provider, dataset=dataset,
    )
    iterator = source.scenarios(limit=1)
    try:
        return next(iterator)
    except StopIteration as error:
        raise ValueError("No scenario satisfies the configured selection requirement") from error
    finally:
        iterator.close()
