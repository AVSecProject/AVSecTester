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
        raise NotImplementedError(
            "phase 4: for scene in self.dataset.scenes(): "
            "m = req.match(scene); if m: yield ScenarioInstance("
            "make_backend=lambda s=scene: self.dataset.make_backend(s), target=m, "
            "provenance=scene.source)  # honour `limit`"
        )


class CarlaScenarioBuilder(ScenarioSource):
    """Construct CARLA scenes that satisfy the requirement, then self-validate with the same predicate.

    Reads the constraints to *parameterize* construction — e.g. sample the lead-vehicle distance inside a
    ``DistanceRange``, a small lateral offset so ``ViewpointRear`` / ``ImageAreaFrac`` hold — emit a
    ``carla_patch_scenario``-style config, build ``SceneGT`` from the CARLA world (live actor boxes +
    sensor calibration, projected via ``simulators.patch_insertion``), and keep the case only if
    ``req.match`` holds. Skeleton: phases 2-3."""

    def __init__(self, base_scenario: dict | None = None, samples: int = 50, seed: int = 0) -> None:
        self.base_scenario = base_scenario
        self.samples = samples
        self.seed = seed

    def scenarios(self, req, limit=None):
        raise NotImplementedError(
            "phase 2-3: sample construction params from req.constraints -> build a carla scenario config"
            " -> carla_scene_gt(backend) -> if req.match(scene): yield ScenarioInstance(make_backend=..."
            ", target=match, provenance={'backend': 'carla', 'params': ...})"
        )
