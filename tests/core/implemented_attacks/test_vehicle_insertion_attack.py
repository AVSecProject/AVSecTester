"""Vehicle placement on navigation routes and its setup insertion."""
import json
import zipfile

import numpy as np
import pytest

from avsectester.attack import AttackSegment, OnlineContext
from avsectester.attack.insertion import InsertObject
from avsectester.closed_loop.simulators.nurec import NuRecRenderer, NuRecSimulator
from avsectester.datasets.nurec import NuRecDataset
from avsectester.execution.services import ClosedLoopServices
from avsectester.geometry.poses import rotation_matrix
from avsectester.implemented_attacks.vehicle_insertion_attack import VehicleInsertionAttack
from avsectester.rendering import ExposureHarmonizer, MeshAsset, MeshPart
from tests.fakes import MemoryDataset, MemorySimulator


def test_vehicle_placement_requires_navigation_setting():
    part = MeshPart(np.array([[0., -1., 0.], [0., 1., 0.], [0., 0., 2.]]), np.array([[0, 1, 2]]),
                    color=np.array([255] * 4, np.uint8))
    attack = VehicleInsertionAttack(MeshAsset((part,), (0., 2., 2.), "vehicle"), decision_steps=2)
    with ClosedLoopServices(MemoryDataset(), MemorySimulator()) as services:
        with pytest.raises(TypeError, match="route_generator"):
            attack.build_metadata(services.scenario("scene", seed=0))


def test_vehicle_placement(nurec_usdz_path, tmp_path):
    spacing = 80. / 19.
    positions = np.array([[0., 0., .3], [spacing, 0., .3]] + [
        [2 * spacing, index * spacing, .3] for index in range(29)])
    poses = np.repeat(np.eye(4)[None], len(positions), axis=0)
    poses[:, :3, 3] = positions
    poses[3, :3, :3] = rotation_matrix((7., -11., 90.))

    def recording_dataset(name, transforms):
        path = tmp_path / f"{name}.usdz"
        with zipfile.ZipFile(nurec_usdz_path) as source:
            metadata = json.loads(source.read("rig_trajectories.json"))
            camera = next(iter(metadata["camera_calibrations"]))
            recording = metadata["rig_trajectories"][0]
            recording["cameras_frame_timestamps_us"][camera] = [
                [index * 100000, index * 100000 + 1] for index in range(len(transforms))]
            recording["cameras_frame_T_rig_worlds"][camera] = [
                [pose.tolist(), pose.tolist()] for pose in transforms]
            with zipfile.ZipFile(path, "w") as target:
                for member in source.namelist():
                    target.writestr(member, json.dumps(metadata) if member == "rig_trajectories.json"
                                    else source.read(member))
        return NuRecDataset([path], time_step=.1)

    def simulator_for(dataset, kind):
        return NuRecSimulator(
            NuRecRenderer(cameras=[dataset.camera_id], asset_directory=tmp_path / "assets"),
            time_step=.1, depth="none", route_generator=kind, physics_update_mode="none",
            object_insertion="scene")

    dataset = recording_dataset("curved", poses)
    part = MeshPart(np.array([[0., -1., 0.], [0., 1., 0.], [0., 0., 2.]]), np.array([[0, 1, 2]]),
                    color=np.array([255] * 4, np.uint8))
    asset = MeshAsset((part,), (0., 2., 2.), "vehicle")
    harmonizer = ExposureHarmonizer()
    attack = VehicleInsertionAttack(asset, distance_m=12., time_ahead_s=0., lateral_offset_m=2.,
                                    decision_steps=2, max_segments=1, harmonizer=harmonizer)
    with ClosedLoopServices(dataset, simulator_for(dataset, "recorded")) as services:
        scenario = services.scenario("scene", seed=0)
        segments = attack.build_metadata(scenario)
        assert len(segments) == 1 and segments[0].start.position == 0
        far = VehicleInsertionAttack(asset, distance_m=81., time_ahead_s=0., decision_steps=2)
        assert far.build_metadata(scenario) == ()
        assert any("available navigation route" in reason for reason in scenario.reasons)
        near = VehicleInsertionAttack(asset, distance_m=6., time_ahead_s=.1,
                                      decision_steps=2, max_segments=1).build_metadata(scenario)
        fast = VehicleInsertionAttack(asset, distance_m=6., time_ahead_s=.25,
                                      decision_steps=2, max_segments=1).build_metadata(scenario)
    expected = poses[3].copy()
    expected[:3, 3] = (2 * spacing - 2., 12. - 2 * spacing, .3)
    np.testing.assert_allclose(segments[0].user_data["world_from_object"], expected, atol=1e-12)
    np.testing.assert_allclose(near[0].user_data["world_from_object"][:3, 3], [6., 0., .3])
    np.testing.assert_allclose(fast[0].user_data["world_from_object"][:3, 3],
                               [2 * spacing, .5 * spacing, .3])

    short = recording_dataset("short", poses[:4])
    with ClosedLoopServices(short, simulator_for(short, "recorded")) as services:
        scenario = services.scenario("scene", seed=0)
        assert VehicleInsertionAttack(asset, distance_m=13., time_ahead_s=0.,
                                      decision_steps=2).build_metadata(scenario) == ()
        assert any("available navigation route" in reason for reason in scenario.reasons)

    folded = np.repeat(np.eye(4)[None], 5, axis=0)
    folded[:, :3, 3] = [[0., 0., .3], [10., 0., .3], [20., 0., .3], [10., 0., .3], [0., 0., .3]]
    reversing = recording_dataset("folded", folded)
    with ClosedLoopServices(reversing, simulator_for(reversing, "recorded")) as services:
        scenario = services.scenario("scene", seed=0)
        assert VehicleInsertionAttack(asset, decision_steps=2).build_metadata(scenario) == ()
        assert any("navigation is unavailable" in reason for reason in scenario.reasons)

    straight = np.repeat(np.eye(4)[None], 5, axis=0)
    straight[:, :3, 3] = [[0., 0., .1], [5., 0., .2], [5., 5., .3], [0., 5., 1.2], [-5., 5., 2.]]
    straight[0, :3, :3] = rotation_matrix((0., 0., 90.))
    straight[3, :3, :3] = rotation_matrix((7., -11., 15.))
    no_route = recording_dataset("no-route", straight)
    with ClosedLoopServices(no_route, simulator_for(no_route, "none")) as services:
        selected = VehicleInsertionAttack(asset, distance_m=5., time_ahead_s=0., lateral_offset_m=2.,
                                          decision_steps=2, max_segments=1).build_metadata(
                                              services.scenario("scene", seed=0))
    expected = straight[3].copy()
    expected[:3, :3] = rotation_matrix((0., 0., 75.)) @ straight[3, :3, :3]
    expected[:3, 3] = [-2., 5., 1.2]
    np.testing.assert_allclose(selected[0].user_data["world_from_object"], expected, atol=1e-12)
    assert attack.configuration_parameters()["distance_m"] == 12.
    assert attack.configuration_parameters()["time_ahead_s"] == 0.
    runtime = attack.create_runtime(segments[0], None)
    action, = runtime.setup(OnlineContext(0, "setup", 0., 0))
    assert isinstance(action, InsertObject) and action.object_id == "inserted_vehicle"
    assert action.asset is asset and action.harmonizer is harmonizer
    np.testing.assert_array_equal(action.world_from_object,
                                  segments[0].user_data["world_from_object"])
    with pytest.raises(ValueError, match="world_from_object"):
        attack.validate_segment(AttackSegment("scene", segments[0].start, segments[0].stop))
    with pytest.raises(ValueError, match="distance_m"):
        VehicleInsertionAttack(asset, distance_m=0.)
    with pytest.raises(ValueError, match="time_ahead_s"):
        VehicleInsertionAttack(asset, time_ahead_s=-1.)
    with pytest.raises(ValueError):
        attack.validate_segment(AttackSegment("scene", segments[0].start, segments[0].stop,
                                             {"world_from_object": np.zeros((4, 4))}))
