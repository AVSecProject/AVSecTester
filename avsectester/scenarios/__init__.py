"""Scenario layer — obtain test cases that satisfy an attack's assumptions (see ``DESIGN.md``).

An attack is only meaningful on a scene that meets its preconditions (a patch-hides-a-vehicle attack
needs a vehicle actually visible + placeable). Each attack declares its preconditions once as a
:class:`~avsectester.scenarios.requirement.ScenarioRequirement` — a target selection + composable
:class:`~avsectester.scenarios.requirement.Constraint`s over ground-truth :class:`~avsectester.scenarios.scene.SceneGT`.

That single requirement drives two :class:`~avsectester.scenarios.source.ScenarioSource`s:
:class:`~avsectester.scenarios.source.DatasetFilter` (**select** the qualifying subset of real
Alpamayo/NuRec traces) and :class:`~avsectester.scenarios.source.CarlaScenarioBuilder` (**construct** a
CARLA scene that qualifies) — both yielding runnable ``ScenarioInstance``s for the evaluation harness.

Phase 1 (here): the ``SceneGT`` schema + the requirement DSL (implemented, tested) + the source
interface (skeletons). Later phases wire the CARLA builder, the dataset adapter, and the eval harness.
"""

from avsectester.scenarios.requirement import (
    Constraint,
    ScenarioMatch,
    ScenarioRequirement,
    TargetSpec,
)
from avsectester.scenarios.scene import CameraCalib, EgoState, ObjectGT, SceneGT
from avsectester.scenarios.source import (
    CarlaScenarioBuilder,
    Dataset,
    DatasetFilter,
    ScenarioInstance,
    ScenarioSource,
)

__all__ = [
    "CameraCalib",
    "CarlaScenarioBuilder",
    "Constraint",
    "Dataset",
    "DatasetFilter",
    "EgoState",
    "ObjectGT",
    "ScenarioInstance",
    "ScenarioMatch",
    "ScenarioRequirement",
    "ScenarioSource",
    "SceneGT",
    "TargetSpec",
]
