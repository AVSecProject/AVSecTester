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


@pytest.fixture
def vehicle_asset():
    part = MeshPart(np.array([[0., -1., 0.], [0., 1., 0.], [0., 0., 2.]]), np.array([[0, 1, 2]]),
                    color=np.array([255] * 4, np.uint8))
    return MeshAsset((part,), (0., 2., 2.), "vehicle")


@pytest.fixture
def recorded_scenario(nurec_usdz_path, tmp_path):
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
    return recording_dataset, simulator_for


@pytest.fixture
def curved_route(recorded_scenario):
    recording, simulator = recorded_scenario
    spacing = 80. / 19.
    positions = np.array([[0., 0., .3], [spacing, 0., .3]] + [
        [2 * spacing, index * spacing, .3] for index in range(29)])
    poses = np.repeat(np.eye(4)[None], len(positions), axis=0)
    poses[:, :3, 3] = positions
    poses[3, :3, :3] = rotation_matrix((7., -11., 90.))
    return recording("curved", poses), simulator, poses, spacing


def test_placement_on_recorded_route(curved_route, vehicle_asset):
    dataset, simulator, poses, spacing = curved_route
    attack = VehicleInsertionAttack(vehicle_asset, distance_m=12., time_ahead_s=0.,
                                    lateral_offset_m=2., decision_steps=2, max_segments=1)
    with ClosedLoopServices(dataset, simulator(dataset, "recorded")) as services:
        view = services.scenario("scene", seed=0)
        segment, = attack.build_metadata(view)
        assert segment.start.position == 0
        far = VehicleInsertionAttack(vehicle_asset, distance_m=81., time_ahead_s=0.,
                                     decision_steps=2)
        assert far.build_metadata(view) == ()
        assert any("available navigation route" in reason for reason in view.reasons)
    expected = poses[3].copy()
    expected[:3, 3] = (2 * spacing - 2., 12. - 2 * spacing, .3)
    np.testing.assert_allclose(segment.user_data["world_from_object"], expected, atol=1e-12)


def test_placement_accounts_for_recorded_speed(curved_route, vehicle_asset):
    dataset, simulator, _, spacing = curved_route
    with ClosedLoopServices(dataset, simulator(dataset, "recorded")) as services:
        view = services.scenario("scene", seed=0)
        for seconds, expected in ((.1, [6., 0., .3]), (.25, [2 * spacing, .5 * spacing, .3])):
            attack = VehicleInsertionAttack(vehicle_asset, distance_m=6., time_ahead_s=seconds,
                                            decision_steps=2, max_segments=1)
            segment, = attack.build_metadata(view)
            np.testing.assert_allclose(segment.user_data["world_from_object"][:3, 3], expected)


def test_placement_rejects_short_or_unavailable_routes(recorded_scenario, vehicle_asset, curved_route):
    recording, simulator = recorded_scenario
    _, _, poses, _ = curved_route
    short = recording("short", poses[:4])
    with ClosedLoopServices(short, simulator(short, "recorded")) as services:
        view = services.scenario("scene", seed=0)
        attack = VehicleInsertionAttack(vehicle_asset, distance_m=13., time_ahead_s=0.,
                                        decision_steps=2)
        assert attack.build_metadata(view) == ()
        assert any("available navigation route" in reason for reason in view.reasons)

    folded = np.repeat(np.eye(4)[None], 5, axis=0)
    folded[:, :3, 3] = [[0., 0., .3], [10., 0., .3], [20., 0., .3], [10., 0., .3], [0., 0., .3]]
    reversing = recording("folded", folded)
    with ClosedLoopServices(reversing, simulator(reversing, "recorded")) as services:
        view = services.scenario("scene", seed=0)
        assert VehicleInsertionAttack(vehicle_asset, decision_steps=2).build_metadata(view) == ()
        assert any("navigation is unavailable" in reason for reason in view.reasons)

    with ClosedLoopServices(MemoryDataset(), MemorySimulator()) as services:
        with pytest.raises(TypeError, match="route_generator"):
            attack.build_metadata(services.scenario("scene", seed=0))


def test_placement_without_navigation(recorded_scenario, vehicle_asset):
    recording, simulator = recorded_scenario
    straight = np.repeat(np.eye(4)[None], 5, axis=0)
    straight[:, :3, 3] = [[0., 0., .1], [5., 0., .2], [5., 5., .3], [0., 5., 1.2], [-5., 5., 2.]]
    straight[0, :3, :3] = rotation_matrix((0., 0., 90.))
    straight[3, :3, :3] = rotation_matrix((7., -11., 15.))
    no_route = recording("no-route", straight)
    with ClosedLoopServices(no_route, simulator(no_route, "none")) as services:
        selected = VehicleInsertionAttack(vehicle_asset, distance_m=5., time_ahead_s=0., lateral_offset_m=2.,
                                          decision_steps=2, max_segments=1).build_metadata(
                                              services.scenario("scene", seed=0))
    expected = straight[3].copy()
    expected[:3, :3] = rotation_matrix((0., 0., 75.)) @ straight[3, :3, :3]
    expected[:3, 3] = [-2., 5., 1.2]
    np.testing.assert_allclose(selected[0].user_data["world_from_object"], expected, atol=1e-12)


def test_create_runtime(vehicle_asset):
    dataset = MemoryDataset()
    start, _, stop, *_ = dataset.frames("scene")
    placement = np.eye(4)
    placement[:3, 3] = [12., 2., 1.]
    segment = AttackSegment("scene", start, stop, {"world_from_object": placement})
    harmonizer = ExposureHarmonizer()
    attack = VehicleInsertionAttack(vehicle_asset, decision_steps=2, harmonizer=harmonizer)
    action, = attack.create_runtime(segment, None).setup(OnlineContext(0, "setup", 0., 0))
    assert isinstance(action, InsertObject) and action.object_id == "inserted_vehicle"
    assert action.asset is vehicle_asset and action.harmonizer is harmonizer
    np.testing.assert_array_equal(action.world_from_object, placement)
    with pytest.raises(ValueError, match="world_from_object"):
        attack.create_runtime(AttackSegment("scene", start, stop), None)
    with pytest.raises(ValueError, match="right-handed rigid pose"):
        attack.create_runtime(AttackSegment("scene", start, stop,
                                           {"world_from_object": np.zeros((4, 4))}), None)


def test_vehicle_placement_options(vehicle_asset):
    with pytest.raises(ValueError, match="distance_m"):
        VehicleInsertionAttack(vehicle_asset, distance_m=0.)
    with pytest.raises(ValueError, match="time_ahead_s"):
        VehicleInsertionAttack(vehicle_asset, time_ahead_s=-1.)
