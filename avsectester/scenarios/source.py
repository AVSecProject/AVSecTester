"""Scenario sources — turn a requirement into runnable, qualifying test cases.

Both providers consume the *same* :class:`~avsectester.scenarios.requirement.ScenarioRequirement`:

  * :class:`DatasetFilter` — **select** real-data (Alpamayo/NuRec) frames whose annotations satisfy it;
  * :class:`CarlaScenarioBuilder` — **construct** a CARLA scene that satisfies it, then self-validate.

Each yields :class:`ScenarioInstance`s: a ``WorldBackend`` factory + the selected target + provenance,
which the evaluation harness runs clean-vs-attacked and scores. Backends are created lazily (a factory,
not a live object) so a source can enumerate thousands of candidates cheaply and the harness only
instantiates the ones it runs.

The two providers are **skeletons** here (phase 1 = interface). Their GT sourcing — CARLA world state vs.
dataset labels — and the backend construction are the phase 2-4 work described in ``DESIGN.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from avsectester.scenarios.requirement import ScenarioMatch, ScenarioRequirement

if TYPE_CHECKING:  # avoid importing the sim stack at module import time
    from avsectester.backend import WorldBackend
    from avsectester.scenarios.scene import SceneGT


@dataclass
class ScenarioInstance:
    """One runnable test case that satisfies a requirement.

    ``make_backend`` lazily builds the ``WorldBackend`` to run (a CARLA config, or a NuRec clip seeded at
    the qualifying frame); ``target`` is the object + camera the attack acts on; ``provenance`` records
    where it came from (dataset clip/frame, or builder parameters) for the report."""

    make_backend: Callable[[], WorldBackend]
    target: ScenarioMatch
    provenance: dict[str, Any] = field(default_factory=dict)


class ScenarioSource(ABC):
    """Yields :class:`ScenarioInstance`s satisfying a requirement. The one interface the eval harness
    depends on — swap a dataset filter for a CARLA builder without touching the harness."""

    @abstractmethod
    def scenarios(self, req: ScenarioRequirement, limit: int | None = None) -> Iterator[ScenarioInstance]:
        ...


class Dataset(ABC):
    """A real-data trace source that can expose ground truth per frame (phase 4).

    NuRec/Alpamayo: the nre-ga renderer produces *pixels* but exposes no actor boxes, so ground truth
    must come from the clip's **annotations** here — separate from the renderer used at run time."""

    @abstractmethod
    def scenes(self) -> Iterator[SceneGT]:
        """Iterate ground-truth scenes (from annotations) across the dataset's clips/frames."""

    @abstractmethod
    def make_backend(self, scene: SceneGT) -> WorldBackend:
        """Build the runtime backend (e.g. a ``NuRecBackend`` on that clip, seeded at that frame)."""


class DatasetFilter(ScenarioSource):
    """Select the subset of a :class:`Dataset` whose annotations satisfy the requirement.

    Only a fraction of a dataset satisfies any given attack's preconditions; this keeps those and drops
    the rest. Skeleton: the loop is trivial once ``Dataset`` exists — evaluate ``req.match`` on each
    scene's GT, and on a hit yield a :class:`ScenarioInstance` whose ``make_backend`` replays that clip."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def scenarios(self, req, limit=None):
        n = 0
        for scene in self.dataset.scenes():
            match = req.match(scene)
            if match is None:
                continue
            yield ScenarioInstance(
                make_backend=lambda s=scene: self.dataset.make_backend(s),
                target=match, provenance=dict(scene.source))
            n += 1
            if limit is not None and n >= limit:
                return


class CarlaScenarioBuilder(ScenarioSource):
    """Construct CARLA scenes that satisfy the requirement by placing a lead vehicle ahead of the ego.

    Enumeration is **analytic** (no CARLA): sample a lead placement — distance inside the requirement's
    ``DistanceRange``, a small lateral offset so ``ViewpointRear`` / ``ImageAreaFrac`` can hold —
    :func:`~avsectester.scenarios.carla_gt.predict_scene_gt` computes the SceneGT the front camera would
    see, and the case is kept only if ``req.match`` holds. The real ``CarlaBackend`` is built lazily in
    ``make_backend`` (so ``scenarios`` runs offline and CARLA is only touched for cases actually run);
    :func:`~avsectester.scenarios.carla_gt.carla_scene_gt` can re-validate the live scene at run time."""

    def __init__(self, base_scenario: dict | None = None, samples: int = 200,
                 lateral_range: tuple[float, float] = (0.0, 3.0), ego_speed: float = 5.0,
                 min_gap: float = 6.0, seed: int = 0) -> None:
        self.base_scenario = base_scenario
        self.samples = samples
        self.lateral_range = lateral_range
        self.ego_speed = ego_speed
        self.min_gap = min_gap  # two ~4.7 m cars can't be spawned closer without overlapping
        self.seed = seed

    def _base(self) -> dict:
        if self.base_scenario is not None:
            return self.base_scenario
        from pathlib import Path

        import yaml
        cfg = Path(__file__).resolve().parents[2] / "configs" / "carla_patch_scenario.yaml"
        return yaml.safe_load(cfg.read_text())

    def scenarios(self, req, limit=None):
        import numpy as np

        from avsectester.scenarios.carla_gt import predict_scene_gt
        from avsectester.scenarios.requirement import DistanceRange

        base = self._base()
        dist = next((c for c in req.constraints if isinstance(c, DistanceRange)), None)
        gmin, gmax = (dist.min_m, dist.max_m) if dist else (5.0, 20.0)
        gmin = max(gmin, self.min_gap)  # keep the lead physically spawnable (no ego overlap)
        rng = np.random.RandomState(self.seed)
        n = 0
        for _ in range(self.samples):
            gap = float(rng.uniform(gmin, gmax))
            lateral = float(rng.uniform(*self.lateral_range))
            scene = predict_scene_gt(base, gap, lateral, speed=self.ego_speed)
            match = req.match(scene)
            if match is None:
                continue
            config = deepcopy(base)
            config.setdefault("lead", {}).update(gap=gap, lateral=lateral)
            config.pop("patches", None)  # the attack is applied by the eval harness, not baked in
            yield ScenarioInstance(
                make_backend=lambda c=config: _carla_backend(c), target=match,
                provenance={"backend": "carla", "gap": gap, "lateral": lateral})
            n += 1
            if limit is not None and n >= limit:
                return


def _carla_backend(config: dict):
    """Lazily build a ``CarlaBackend`` (imports the CARLA stack only when a scenario is actually run)."""
    from avsectester.simulators.carla import CarlaBackend

    return CarlaBackend(config)
