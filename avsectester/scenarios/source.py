"""Prepare candidate initial states and select runnable cases with one filter pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .context import FilterContext
from .requirement import ScenarioMatch, ScenarioRequirement

if TYPE_CHECKING:
    from avsectester.backend import WorldBackend
    from .scene import SceneGT


@dataclass(init=False)
class ScenarioInstance:
    """A selected origin and its fixed role/insertion plan, plus a fresh-backend factory."""

    make_backend: Callable[[], WorldBackend]
    match: ScenarioMatch
    provenance: dict[str, Any] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        make_backend: Callable[[], WorldBackend],
        match: ScenarioMatch | None = None,
        provenance: dict[str, Any] | None = None,
        selection: dict[str, Any] | None = None,
        *,
        target: ScenarioMatch | None = None,
    ):
        # Keep both positional construction and the original target= keyword working.
        if match is not None and target is not None:
            raise TypeError("Provide match or its compatibility alias target, not both")
        if match is None:
            match = target
        if match is None:
            raise TypeError("ScenarioInstance requires a match")
        self.make_backend = make_backend
        self.match = match
        self.provenance = {} if provenance is None else provenance
        self.selection = {} if selection is None else selection

    @property
    def target(self) -> ScenarioMatch:
        """Compatibility alias for :attr:`match`."""
        return self.match

    @target.setter
    def target(self, value: ScenarioMatch) -> None:
        self.match = value


class ScenarioSource(ABC):
    """Initial preparation only. Filters are never installed in experiment run loops."""

    def _evaluate(self, requirement, context):
        result = requirement.evaluate(context)
        scene = context.scene
        self.selection_log.append(
            {
                "source": deepcopy(scene.source),
                "frame": scene.frame,
                "t": scene.t,
                **result.summary(),
            }
        )
        return result

    def save_selection(self, path) -> None:
        """Write accepted, rejected and unavailable cases without copying raw provider data."""
        import json
        from pathlib import Path

        Path(path).write_text(json.dumps(self.selection_log, indent=2) + "\n")

    @abstractmethod
    def scenarios(
        self, req: ScenarioRequirement, limit: int | None = None
    ) -> Iterator[ScenarioInstance]: ...


class Dataset(ABC):
    """Ground-truth records with overridable access to original data and prepared sequences."""

    @abstractmethod
    def scenes(self) -> Iterator[SceneGT]: ...

    @abstractmethod
    def make_backend(self, scene: SceneGT) -> WorldBackend: ...

    def context(self, scene: SceneGT) -> FilterContext:
        """Expose the adapter itself. Providers may also expose SDK clients and raw metadata."""
        return FilterContext(scene=scene, dataset=self)

    def initial_sequence(self, scene: SceneGT, frames: int) -> tuple[SceneGT, ...]:
        """Return available recorded/predefined states, beginning at ``scene``.

        A single-frame source cannot certify a longer window. It must not repeat its initial
        state or silently run a driving model to fabricate the remaining frames.
        """
        return (scene,)


def _check_limit(limit):
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 0):
        raise ValueError("limit must be a nonnegative integer")


class DatasetFilter(ScenarioSource):
    """Select recorded initial cases while preserving original provider API access.

    ``prepare_context`` augments a selection context, for example with a custom visibility
    provider. ``prepare_scene`` is a single-scene geometry adapter for existing callers.
    """

    def __init__(
        self,
        dataset: Dataset,
        prepare_scene: Callable | None = None,
        prepare_context: Callable[[FilterContext], FilterContext] | None = None,
    ):
        self.dataset = dataset
        self.prepare_scene = prepare_scene
        self.prepare_context = prepare_context
        self.selection_log: list[dict] = []

    def _context(self, scene, resources):
        context = resources.enter_context(self.dataset.context(scene))
        if self.prepare_context is not None:
            prepared = self.prepare_context(context)
            if not isinstance(prepared, FilterContext):
                raise TypeError("Context preparation must return FilterContext")
            if prepared is not context:
                resources.enter_context(prepared)
            context = prepared
        if not isinstance(context, FilterContext):
            raise TypeError("Context preparation must return FilterContext")
        return context

    def scenarios(self, req, limit=None):
        _check_limit(limit)
        self.selection_log = []
        if limit == 0:
            return
        accepted = 0
        for raw_scene in self.dataset.scenes():
            sequence = tuple(
                deepcopy(scene)
                for scene in self.dataset.initial_sequence(raw_scene, req.window.frames)
            )
            if not sequence:
                sequence = (deepcopy(raw_scene),)
            if (sequence[0].frame, sequence[0].t) != (raw_scene.frame, raw_scene.t):
                raise ValueError("Dataset initial_sequence must preserve the selected origin")
            if self.prepare_scene:
                sequence = tuple(self.prepare_scene(scene) for scene in sequence)
            scene = sequence[0]
            with ExitStack() as resources:

                def frame_context(frame):
                    return self._context(frame, resources)

                context = frame_context(scene).derive(
                    sequence=sequence, frame_context=frame_context
                )
                result = self._evaluate(req, context)
                if result.match is None:
                    continue
                factory = context.native.get("make_backend")
                if factory is None:
                    factory = lambda s=scene: self.dataset.make_backend(s)
            # Release preview clients before yielding a case, even if enumeration stops here.
            yield ScenarioInstance(factory, result.match, dict(scene.source), result.summary())
            accepted += 1
            if limit is not None and accepted >= limit:
                return


class CarlaScenarioBuilder(ScenarioSource):
    """Instantiate CARLA candidates before applying the common initial-selection pipeline.

    Fixed ``base_scenario`` positions are respected by default. ``sample_scene=True`` allows
    lead distance/lateral sampling within configured ranges. This changes candidate scene
    generation, never user-specified insertion placement. ``analytic_preview`` explicitly
    opts into approximate offline enumeration without asserting live scene validity.

    A candidate provider is a ``config -> context manager[FilterContext]`` callable. The
    default uses CARLA. Providers can prepare a predefined initial sequence and expose
    original world/client/sensor APIs, without running a driving policy.
    """

    def __init__(
        self,
        base_scenario: dict | None = None,
        samples: int = 1,
        lateral_range: tuple[float, float] = (0.0, 3.0),
        ego_speed: float = 0.0,
        min_gap: float = 6.0,
        seed: int = 0,
        prepare_scene: Callable | None = None,
        *,
        candidate_provider: Callable | None = None,
        sample_scene: bool = False,
        analytic_preview: bool = False,
        prepare_context: Callable[[FilterContext], FilterContext] | None = None,
    ):
        if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
            raise ValueError("samples must be a positive integer")
        if ego_speed != 0:
            raise ValueError(
                "The CARLA builder starts stationary. Nonzero initial speed is not supported."
            )
        if lateral_range[0] > lateral_range[1]:
            raise ValueError("lateral_range must be ordered")
        if analytic_preview and candidate_provider is not None:
            raise ValueError("analytic_preview cannot also use a live candidate_provider")
        self.base_scenario = base_scenario
        self.samples = samples
        self.lateral_range = lateral_range
        self.ego_speed = ego_speed
        self.min_gap = min_gap
        self.seed = seed
        self.prepare_scene = prepare_scene
        self.prepare_context = prepare_context
        self.candidate_provider = candidate_provider
        self.sample_scene = sample_scene
        self.analytic_preview = analytic_preview
        self.selection_log: list[dict] = []

    def _base(self):
        if self.base_scenario is not None:
            return deepcopy(self.base_scenario)
        from pathlib import Path
        import yaml

        cfg = Path(__file__).resolve().parents[2] / "configs" / "carla_patch_scenario.yaml"
        return yaml.safe_load(cfg.read_text())

    def _configs(self, req):
        base = self._base()
        base.pop("patches", None)
        if not self.sample_scene:
            yield base
            return
        import numpy as np
        from avsectester.scenarios.filters import DistanceRange

        dist = next((c for c in req.constraints if isinstance(c, DistanceRange)), None)
        lower, upper = (dist.min_m, dist.max_m) if dist else (self.min_gap, 20.0)
        lower = max(lower, self.min_gap)
        if lower > upper:
            return
        rng = np.random.RandomState(self.seed)
        for _ in range(self.samples):
            config = deepcopy(base)
            config.setdefault("lead", {}).update(
                gap=float(rng.uniform(lower, upper)),
                lateral=float(rng.uniform(*self.lateral_range)),
            )
            yield config

    def scenarios(self, req, limit=None):
        _check_limit(limit)
        self.selection_log = []
        if limit == 0:
            return
        provider = self.candidate_provider
        if provider is None and not self.analytic_preview:
            from .carla_provider import CarlaCandidateProvider

            provider = CarlaCandidateProvider(initial_frames=req.window.frames)
        accepted = 0
        for config in self._configs(req):
            if self.analytic_preview:
                from .carla_gt import predict_scene_gt

                lead = config.get("lead", {})
                scene = predict_scene_gt(config, lead.get("gap", 10.0), lead.get("lateral", 0.0))
                context_manager = nullcontext(FilterContext(scene))
            else:
                context_manager = provider(config)
            instance = None
            with ExitStack() as resources:
                context = resources.enter_context(context_manager)
                if not isinstance(context, FilterContext):
                    raise TypeError("CARLA candidate providers must yield FilterContext")
                resources.enter_context(context)
                if self.prepare_scene is not None:
                    context = context.derive(scene=self.prepare_scene(context.scene))
                if self.prepare_context is not None:
                    context = self.prepare_context(context)
                    if not isinstance(context, FilterContext):
                        raise TypeError("Context preparation must return FilterContext")
                    resources.enter_context(context)
                result = self._evaluate(req, context)
                if result.match is not None:
                    factory = context.native.get("make_backend")
                    if factory is None:
                        replay = (context.metadata or {}).get("replay_scenario", config)
                        factory = lambda c=deepcopy(replay): _carla_backend(c)
                    provenance = {
                        **context.scene.source,
                        "backend": "carla",
                        "validation": "analytic_preview" if self.analytic_preview else "prepared",
                    }
                    instance = ScenarioInstance(factory, result.match, provenance, result.summary())
            # Release the preparation world before a consumer starts the actual experiment.
            if instance is not None:
                yield instance
                accepted += 1
                if limit is not None and accepted >= limit:
                    return


def _carla_backend(config):
    from avsectester.simulators.carla import CarlaBackend

    return CarlaBackend(config)
