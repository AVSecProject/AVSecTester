"""CARLA recipes are source data. Materialization belongs to experiment preparation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field

from avsectester.plane import Control

from .base import (
    ActorState, Dataset, FrameRef, ScenarioDescriptor, SourceData, Unavailable, nonnegative,
)


@dataclass(frozen=True)
class PrescribedMotion:
    """Exactly one explicit control sequence (N) or world-pose sequence (N+1).

    Trajectories include velocity evidence. A stationary actor is represented explicitly by
    zero controls or repeated poses, never by an inferred driving policy.
    """

    controls: tuple[Control, ...] = ()
    trajectory: tuple[ActorState, ...] = ()

    def validate(self, decision_steps: int) -> None:
        if bool(self.controls) == bool(self.trajectory):
            raise ValueError("Specify exactly one explicit control or trajectory sequence")
        if self.controls:
            if len(self.controls) != decision_steps:
                raise ValueError("Control sequence must cover every recipe decision")
            for control in self.controls:
                if not isinstance(control, Control) or control.trajectory is not None:
                    raise ValueError("CARLA prescribed controls must be actuator commands")
                if not all(math.isfinite(value) for value in
                           (control.throttle, control.brake, control.steer)):
                    raise ValueError("Prescribed controls must be finite")
        else:
            if len(self.trajectory) != decision_steps + 1:
                raise ValueError("Trajectory must include the terminal reference state")
            if any(state.velocity is None for state in self.trajectory):
                raise ValueError("Prescribed trajectories require velocity evidence")


@dataclass(frozen=True)
class CarlaRecipe:
    scenario: dict
    decision_steps: int
    time_step: float
    motion: Mapping[str, PrescribedMotion]
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        nonnegative(self.decision_steps, "decision_steps")
        if self.decision_steps == 0:
            raise ValueError("A recipe must prescribe at least one decision")
        if not math.isfinite(self.time_step) or self.time_step <= 0:
            raise ValueError("time_step must be positive and finite")
        scenario = deepcopy(self.scenario)
        client = scenario["client"]
        if not client.get("synchronous", False):
            raise ValueError("Reference recipes require synchronous CARLA")
        if not math.isclose(1 / float(client["rate"]), self.time_step, abs_tol=1e-9):
            raise ValueError("Recipe time_step must match the CARLA client rate")
        npcs = scenario.get("npcs") or []
        if isinstance(npcs, dict):
            raise ValueError("Expand NPCs into explicit spawn/motion specifications for a recipe")
        actors = {"ego", *(f"npc:{i}" for i in range(len(npcs)))}
        if scenario.get("lead") is not None:
            actors.add("lead")
        if set(self.motion) != actors:
            raise ValueError(f"Explicit motion required for exactly these actors: {sorted(actors)}")
        if scenario["ego"].get("autopilot") or any(npc.get("autopilot", True) for npc in npcs):
            raise ValueError("Traffic Manager motion is unsupported for reference recipes")
        for motion in self.motion.values():
            motion.validate(self.decision_steps)
        object.__setattr__(self, "scenario", scenario)
        object.__setattr__(self, "motion", deepcopy(dict(self.motion)))
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))


class CarlaDataset(Dataset):
    def __init__(self, recipes: Mapping[str, CarlaRecipe]) -> None:
        self._recipes = deepcopy(dict(recipes))

    def scenarios(self):
        for identity in self._recipes:
            yield ScenarioDescriptor(identity, "carla", {"recipe_only": True})

    def frames(self, scenario_id):
        recipe = self._recipes[scenario_id]
        return tuple(FrameRef(scenario_id, i, i, i * recipe.time_step)
                     for i in range(recipe.decision_steps + 1))

    def reference(self, scenario_id, frame):
        self.checked_frame(scenario_id, frame)
        return Unavailable("recipe_only")

    def native(self, scenario_id):
        return deepcopy(self._recipes[scenario_id])

    def source_data(self, scenario_id, start):
        self.checked_frame(scenario_id, start)
        return SourceData("carla", scenario_id, start, {"recipe": self.native(scenario_id)})
