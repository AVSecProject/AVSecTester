"""Vehicle placement along the recorded ego path and its setup insertion."""
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


def test_vehicle_placement(nurec_usdz_path, tmp_path):
    positions = np.array([
        [0., 0., 0.], [1., 0., .1], [1., 1., .2], [2., 1., .3], [3., 1., .4],
        [3., 2., .5], [4., 2., .6], [5., 2., .7], [6., 2., .8], [7., 2., .9],
    ])
    poses = np.repeat(np.eye(4)[None], len(positions), axis=0)
    poses[:, :3, 3] = positions
    poses[2, :3, :3] = rotation_matrix((7., -11., 90.))
    archive_path = tmp_path / "curved.usdz"
    with zipfile.ZipFile(nurec_usdz_path) as source:
        metadata = json.loads(source.read("rig_trajectories.json"))
        camera = next(iter(metadata["camera_calibrations"]))
        recording = metadata["rig_trajectories"][0]
        recording["cameras_frame_timestamps_us"][camera] = [
            [index * 100000, index * 100000 + 1] for index in range(len(poses))]
        recording["cameras_frame_T_rig_worlds"][camera] = [
            [pose.tolist(), pose.tolist()] for pose in poses]
        with zipfile.ZipFile(archive_path, "w") as target:
            for name in source.namelist():
                target.writestr(name, json.dumps(metadata) if name == "rig_trajectories.json"
                                else source.read(name))
    dataset = NuRecDataset([archive_path], time_step=.1)
    simulator = NuRecSimulator(
        NuRecRenderer(cameras=[dataset.camera_id], asset_directory=tmp_path / "assets"),
        time_step=.1, depth="none", route_generator="none", physics_update_mode="none",
        object_insertion="scene")
    part = MeshPart(np.array([[0., -1., 0.], [0., 1., 0.], [0., 0., 2.]]), np.array([[0, 1, 2]]),
                    color=np.array([255] * 4, np.uint8))
    asset = MeshAsset((part,), (0., 2., 2.), "vehicle")
    harmonizer = ExposureHarmonizer()
    attack = VehicleInsertionAttack(asset, distance_m=1.9, lateral_offset_m=2., decision_steps=2,
                                    harmonizer=harmonizer)
    with ClosedLoopServices(dataset, simulator) as services:
        scenario = services.scenario("scene", seed=0)
        segments = attack.build_metadata(scenario)
        far = VehicleInsertionAttack(asset, distance_m=100., decision_steps=2)
        assert far.build_metadata(scenario) == ()
        assert scenario.reasons
    # Only the start needs a placement pose. No recorded pose lies distance_m beyond frame 8.
    assert [(segment.start.position, segment.stop.position) for segment in segments] == [
        (0, 2), (2, 4), (4, 6), (6, 8)]
    # A straight-line distance would choose a later pose on this bent recording.
    assert np.linalg.norm(positions[2] - positions[0]) < attack.distance_m
    for segment in segments:
        expected = poses[segment.stop.position].copy()
        expected[:3, 3] += expected[:3, 1] * 2.
        np.testing.assert_allclose(segment.user_data["world_from_object"], expected, atol=1e-12)
    runtime = attack.create_runtime(segments[0], None)
    action, = runtime.setup(OnlineContext(0, "setup", 0., 0))
    assert isinstance(action, InsertObject) and action.object_id == "inserted_vehicle"
    assert action.asset is asset and action.harmonizer is harmonizer
    np.testing.assert_array_equal(action.world_from_object,
                                  segments[0].user_data["world_from_object"])
    with pytest.raises(ValueError, match="world_from_object"):
        attack.validate_segment(AttackSegment("scene", segments[0].start, segments[0].stop))
