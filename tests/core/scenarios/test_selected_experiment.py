"""Selected identities and initial state reach the paired evaluation without runtime filtering."""

from copy import deepcopy
from dataclasses import replace

import numpy as np

from avsectester.backend import AVStack, WorldBackend
from avsectester.evaluation.robustness import evaluate_robustness
from avsectester.plane import Control, Observation
from avsectester.scenarios import (
    Constraint,
    Dataset,
    DatasetFilter,
    FilterResult,
    InitialWindow,
    RoleSpec,
    ScenarioRequirement,
)


def test_selected_case_drives_a_pair_from_the_same_origin(scene, patch):
    origin = scene(frame=12)
    origin.ego.speed = 10
    origin.source = {"scene_id": "selected-clip"}
    filter_calls, resets, attacked_ids, backends, controls = [], [], [], [], []

    class Eligible(Constraint):
        def evaluate(self, context):
            assert not backends, "Initial filters must finish before the driving backend is built"
            actor = context.bindings["host"][0]
            filter_calls.append((context.scene.frame, actor.track_id))
            return FilterResult.from_bool(actor.distance >= 10, "host distance")

    class Backend(WorldBackend):
        def __init__(self, selected):
            self.selected = deepcopy(selected)
            self.prepared = self.closed = False

        def prepare_clean_attack_pair(self):
            self.prepared = True

        def observation(self):
            return Observation(
                self.t,
                self.frame,
                {"front": np.zeros((4, 4, 3), np.uint8)},
                vehicle_state=self.selected.ego.pose,
                ego_speed=self.speed,
            )

        def reset(self):
            assert self.prepared
            self.frame, self.t, self.speed = (
                self.selected.frame,
                self.selected.t,
                self.selected.ego.speed,
            )
            resets.append((self.frame, self.t, self.speed, deepcopy(self.selected.ego.pose)))
            return self.observation()

        def step(self, control):
            controls.append(control)
            self.speed = max(0, self.speed - control.brake * 5)
            self.frame += 1
            self.t += 0.1
            return self.observation()

        def close(self):
            self.closed = True

    class Cases(Dataset):
        def scenes(self):
            yield origin

        def initial_sequence(self, selected, frames):
            return tuple(
                replace(selected, frame=selected.frame + i, t=selected.t + i * 0.1)
                for i in range(frames)
            )

        def make_backend(self, selected):
            backend = Backend(selected)
            backends.append(backend)
            return backend

    class Stack(AVStack):
        def __call__(self, observation):
            return (
                Control(brake=1) if observation.sensor_data["front"].any() else Control(throttle=1)
            )

    def attack_for(match, backend):
        assert match.binding_ids == {"host": ("b",)}
        assert match.insertions[0].placement.host == "host"
        np.testing.assert_array_equal(match.scene.ego.pose, backend.selected.ego.pose)

        def perturb(observation):
            attacked_ids.append(match.binding_ids["host"])
            return replace(observation, sensor_data={"front": np.full((4, 4, 3), 255, np.uint8)})

        return perturb

    source = DatasetFilter(Cases())
    requirement = ScenarioRequirement(
        "initial",
        roles={"host": RoleSpec()},
        insertions=(patch(),),
        window=InitialWindow(2),
        constraints=[Eligible()],
    )
    report = evaluate_robustness(source, requirement, attack_for, Stack, frames=4, conditions=[])
    assert filter_calls == [(12, "a"), (12, "b"), (13, "a"), (13, "b")]
    assert len(backends) == 1 and backends[0].closed
    assert len(resets) == 2
    assert resets[0][:3] == resets[1][:3] == (origin.frame, origin.t, origin.ego.speed)
    np.testing.assert_array_equal(resets[0][3], resets[1][3])
    assert attacked_ids == [("b",)] * 4
    assert all(c.throttle == 1 for c in controls[:4])
    assert all(c.brake == 1 for c in controls[4:])
    assert len(report.rows) == 1 and report.rows[0].scenario_id == "selected-clip"
    assert report.per_condition["clean"].n == 1
