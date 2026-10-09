"""Initial scenario selection with composable filters and native provider access."""

from avsectester.rendering.types import Visibility
from avsectester.scenarios.context import FilterContext
from avsectester.scenarios.nl import interpret
from avsectester.scenarios.filters import (
    All,
    Any,
    Not,
    Constraint,
    FilterResult,
)
from avsectester.scenarios.requirement import (
    InitialWindow,
    RoleSpec,
    SelectionResult,
    ScenarioMatch,
    ScenarioRequirement,
    TargetSpec,
)
from avsectester.scenarios.scene import CameraCalib, EgoState, ObjectGT, PlacementCandidate, SceneGT
from avsectester.scenarios.serialize import (
    register_constraint,
    requirement_from_dict,
    requirement_to_dict,
)
from avsectester.scenarios.source import (
    CarlaScenarioBuilder,
    Dataset,
    DatasetFilter,
    ScenarioInstance,
    ScenarioSource,
)

__all__ = [
    "FilterContext",
    "All",
    "Any",
    "Not",
    "InitialWindow",
    "RoleSpec",
    "CameraCalib",
    "CarlaScenarioBuilder",
    "Constraint",
    "FilterResult",
    "SelectionResult",
    "PlacementCandidate",
    "Visibility",
    "register_constraint",
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
    "interpret",
    "requirement_from_dict",
    "requirement_to_dict",
]
