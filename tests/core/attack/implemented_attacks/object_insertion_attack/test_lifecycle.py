"""Portable attacks complete public preparation, execution and offline evaluation."""
from dataclasses import replace
import numpy as np
import pytest

from avsectester.attack.implemented_attacks.object_insertion_attack import ObjectInsertionAttack
from avsectester.attack.insertion import PlaneAsset
from avsectester.attack.placement import WorldPlacement
from avsectester.closed_loop.services import DrivingAdapter
from avsectester.datasets import FrameReference
from avsectester.evaluation import evaluate
from avsectester.experiment import Experiment, ExperimentConfiguration, TreatmentConfiguration
from avsectester.geometry.poses import ActorPose

def test_attack_prepares_executes_and_evaluates(memory_experiment):
    Data = memory_experiment.Data
    Driver = memory_experiment.Driver
    Sim = memory_experiment.Sim
    attack = ObjectInsertionAttack(decision_steps=2, max_segments=1,
        assets=PlaneAsset(np.full((8, 8, 4), 255, np.uint8), 2., 1.),
        placement=WorldPlacement((6., -2., 0.)))
    config = ExperimentConfiguration(Data(), Sim(), DrivingAdapter(lambda seed: Driver()),
        attacks={"test": attack}, treatments=(TreatmentConfiguration("clean"),
        TreatmentConfiguration("attack", "test")), strict=True)
    prepared = Experiment.prepare(config)
    assert len(prepared.metadata["test"]) == 1
    result = Experiment.execute(prepared)
    assert not result.failures
    assert len(result.runs) == 2
    assert all(len(run.log.steps) == 2 for run in result.runs)
    report = evaluate(result, config)
    assert len(report.groups) == 1
    clean, attacked = [run.log for run in result.runs]
    assert any(e.get("outcome") == "applied" for e in attacked.steps[0].effects)

def test_ego_origin_relative_object_is_resolved_once_and_runtime_uses_metadata(memory_experiment):
    CAM_FROM_WORLD = memory_experiment.CAM_FROM_WORLD
    Data = memory_experiment.Data
    Driver = memory_experiment.Driver
    Sim = memory_experiment.Sim
    World = memory_experiment.World
    class ShiftedWorld(World):
        def sensor_view(self, sensor):
            view = super().sensor_view(sensor)
            origin = np.array([[0., -1., 0., 100. + self.i], [1., 0., 0., 200.],
                               [0., 0., 1., 0.], [0., 0., 0., 1.]])
            return replace(view, ego=ActorPose(origin),
                           cam_from_world=CAM_FROM_WORLD @ np.linalg.inv(origin))

    class ShiftedData(Data):
        def frame_data(self, scenario_id, frame):
            original = super().frame_data(scenario_id, frame)
            pose = np.array([[0., -1., 0., 100.], [1., 0., 0., 200.],
                             [0., 0., 1., 0.], [0., 0., 0., 1.]])
            return replace(original, ego=replace(original.ego, pose=pose, box_center=(0., 0., .74)))

    class ShiftedSim(Sim):
        def create_runtime(self, seed=None):
            return ShiftedWorld(self)

    attack = ObjectInsertionAttack(decision_steps=1, max_segments=2,
        assets=PlaneAsset(np.full((8, 8, 4), 255, np.uint8), 2., 1.),
        placement=WorldPlacement((6., -2., 0.)), placement_relative_to_ego_origin=True)
    config = ExperimentConfiguration(ShiftedData(), ShiftedSim(), DrivingAdapter(lambda seed: Driver()),
                                     attacks={"object": attack}, strict=True)
    prepared = Experiment.prepare(config)
    assert len(prepared.metadata["object"]) == 2
    for segment in prepared.metadata["object"]:
        np.testing.assert_allclose(segment.placement.position_m, [102., 206., 0.])
        runtime = attack.create_runtime(segment, prepared.artifacts["object"])
        assert runtime.insertion.placement == segment.placement
    assert attack.placement.position_m == (6., -2., 0.)

def test_selection_does_not_hide_invalid_asset_geometry(memory_experiment):
    Data = memory_experiment.Data
    Driver = memory_experiment.Driver
    Sim = memory_experiment.Sim
    class InvalidAsset:
        def planes(self):
            raise ValueError("invalid asset geometry")

    attack = ObjectInsertionAttack(decision_steps=2, assets=InvalidAsset(),
                                    placement=WorldPlacement((8., 0., 0.)))
    config = ExperimentConfiguration(Data(), Sim(), DrivingAdapter(lambda seed: Driver()),
                                     attacks={"attack": attack}, strict=True)
    with pytest.raises(ValueError, match="invalid asset geometry"):
        Experiment.prepare(config)

def test_selection_keeps_earlier_segments_when_target_later_crosses_camera_plane(memory_experiment):
    Data = memory_experiment.Data
    Driver = memory_experiment.Driver
    Sim = memory_experiment.Sim
    World = memory_experiment.World
    attack = ObjectInsertionAttack(decision_steps=2,
        assets=PlaneAsset(np.full((8, 8, 4), 255, np.uint8), 2., 1.),
        placement=WorldPlacement((8., 0., 0.)))
    class LongData(Data):
        timeline = tuple(FrameReference("s", i, i, i * .1) for i in range(6))

    class CrossingWorld(World):
        def sensor_view(self, sensor):
            view = super().sensor_view(sensor)
            if self.i < 3:
                return view
            r = 1 / np.sqrt(2)
            camera_x = 8. if self.i == 3 else 20.
            camera = np.array([[r, -r, 0., -camera_x*r], [0., 0., -1., 0.],
                               [r, r, 0., -camera_x*r], [0., 0., 0., 1.]])
            return replace(view, cam_from_world=camera)

    class CrossingSim(Sim):
        def create_runtime(self, seed=None):
            return CrossingWorld(self)

    config = ExperimentConfiguration(LongData(), CrossingSim(), DrivingAdapter(lambda seed: Driver()),
                                     attacks={"attack": attack}, strict=True)
    prepared = Experiment.prepare(config)
    assert [(s.start.position, s.stop.position) for s in prepared.metadata["attack"]] == [(0, 2)]
    assert prepared.selections[0].status == "selected"
    # Assert the fixture exercises unknown camera-plane visibility, not merely a zero score.
    runtime = CrossingSim().create_runtime()
    runtime.i = 3
    from avsectester.attack.insertion.effect import InsertionAttackEffect
    evidence = InsertionAttackEffect("front", attack.insertion(attack.assets)).evidence(runtime.sensor_view("front"))
    assert evidence.visibility is None and "camera plane" in evidence.reason

def test_selection_does_not_hide_invalid_sensor_configuration(memory_experiment):
    Data = memory_experiment.Data
    Driver = memory_experiment.Driver
    Sim = memory_experiment.Sim
    attack = ObjectInsertionAttack(decision_steps=2, sensor="unknown-camera",
        placement=WorldPlacement((8., 0., 0.)))
    config = ExperimentConfiguration(Data(), Sim(), DrivingAdapter(lambda seed: Driver()),
                                     attacks={"attack": attack}, strict=True)
    with pytest.raises(KeyError, match="unknown-camera"):
        Experiment.prepare(config)
