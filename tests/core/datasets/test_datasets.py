"""Dataset components share one public timeline contract; interpolation stays numerical."""
from dataclasses import replace
import zipfile

import numpy as np
import pytest

from avsectester.data_types import Control
from avsectester.datasets import FrameData
from avsectester.data_types import Unavailable
from avsectester.datasets.carla import CarlaRecipe, PrescribedMotion
from avsectester.datasets.nurec import NuRecDataset, read_scene_mesh
from avsectester.geometry.poses import rotation_matrix
from avsectester.maps import ScenarioMap
from tests.core.datasets.scenario_map_contract import assert_scenario_map


@pytest.fixture(params=[("carla_dataset", Unavailable), ("nurec_dataset", ScenarioMap),
                        ("nuscenes_dataset", Unavailable)])
def dataset_case(request):
    name, map_type = request.param
    return request.getfixturevalue(name), map_type


def test_datasets(dataset_case):
    dataset, map_type = dataset_case
    scenario = next(dataset.scenarios())
    frames = dataset.frames(scenario.id)
    assert [frame.position for frame in frames] == list(range(len(frames)))
    assert all(right.timestamp > left.timestamp for left, right in zip(frames, frames[1:]))
    assert dataset.decision_window(scenario.id, frames[0], len(frames) - 1) == frames
    start = dataset.scenario_start(scenario.id, frames[1])
    assert start.frame == frames[1] and start.scenario_id == scenario.id
    assert isinstance(dataset.scenario_map(scenario.id), map_type)
    assert start.scenario_map == dataset.scenario_map(scenario.id)
    with pytest.raises(ValueError, match="terminal frame"):
        dataset.decision_window(scenario.id, frames[-1], 1)
    with pytest.raises(ValueError, match="scenario's frames"):
        dataset.scenario_start(scenario.id, replace(frames[1], timestamp=99.))


@pytest.mark.parametrize("dataset_fixture", ["nurec_dataset", "nuscenes_dataset"])
def test_recorded_frame_data(dataset_fixture, request):
    dataset = request.getfixturevalue(dataset_fixture)
    scenario = next(dataset.scenarios())
    frame = dataset.frames(scenario.id)[1]
    data = dataset.frame_data(scenario.id, frame)
    assert isinstance(data, FrameData)
    assert data.frame == frame and data.ego is not None
    assert "host" in data.actors and "ego" not in data.actors
    assert data.actors["host"].extent == (4., 2., 2.)
    frames = dataset.frames(scenario.id)
    start = dataset.scenario_start(scenario.id, frame)
    assert len(start.recorded_ego) == len(frames)
    for recorded, reference in zip(start.recorded_ego, frames):
        expected = dataset.frame_data(scenario.id, reference).ego
        np.testing.assert_array_equal(recorded.pose, expected.pose)
        assert recorded.velocity == expected.velocity
        assert recorded.extent == expected.extent and recorded.box_center == expected.box_center


@pytest.mark.parametrize("dataset_fixture", ["nurec_dataset"])
def test_scenario_map(dataset_fixture, request):
    dataset = request.getfixturevalue(dataset_fixture)
    scenario = next(dataset.scenarios())
    frame = dataset.frames(scenario.id)[0]
    assert_scenario_map(dataset.scenario_map(scenario.id), dataset.frame_data(scenario.id, frame).ego)


def test_carla_frame_data(carla_dataset):
    frame = carla_dataset.frames("scene")[1]
    data = carla_dataset.frame_data("scene", frame)
    assert isinstance(data, Unavailable) and data.reason == "recipe_only"


def test_nurec_interpolation(nurec_usdz_path):
    dataset = NuRecDataset([nurec_usdz_path], time_step=.05)
    frames = dataset.frames("scene")
    for position, timestamp in ((0, 0.), (5, .25), (10, .5)):
        frame = dataset.frame_data("scene", frames[position])
        np.testing.assert_allclose(frame.ego.pose[:3, 3], [2 * timestamp, 0., .3])
        np.testing.assert_allclose(frame.actors["host"].pose[:3, 3], [8 + 4 * timestamp, 0., 1.])
        np.testing.assert_allclose(frame.actors["host"].pose[:3, :3],
                                   rotation_matrix((0., 0., 360 * timestamp)), atol=1e-12)
        assert ("brief" in frame.actors) == (timestamp == .25)
    ego = dataset.frame_data("scene", frames[5]).ego
    assert ego.extent == (4., 2., 1.5) and ego.box_center == (.2, 0., .7)
    np.testing.assert_allclose(dataset.ego_state("scene", 250000).pose, ego.pose)
    np.testing.assert_allclose(dataset.actor_states("scene", 250000)["host"].pose,
                               dataset.frame_data("scene", frames[5]).actors["host"].pose)
    # Returned annotations and arrays remain independent of the immutable recording.
    data = dataset.frame_data("scene", frames[5])
    data.actors["host"].pose[:] = 0
    metadata = dataset.raw_metadata("scene")
    metadata["sequence_tracks.json"].clear()
    np.testing.assert_allclose(dataset.frame_data("scene", frames[5]).actors["host"].pose[0, 3], 9.)


def test_nurec_camera_geometry(nurec_dataset):
    camera, rig_from_camera = nurec_dataset.camera_geometry("scene")
    assert (camera.width, camera.height) == (16, 12)
    np.testing.assert_allclose(rig_from_camera,
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]])
    rig_from_camera[:] = 0.
    assert nurec_dataset.camera_geometry("scene")[1][2, 3] == 1.5
    with pytest.raises(KeyError, match="Unknown NuRec scenario"):
        nurec_dataset.camera_geometry("missing")


def test_nurec_camera_timestamps(nurec_dataset):
    timestamps = nurec_dataset.camera_timestamps("scene")
    np.testing.assert_allclose(timestamps, [0., .25, .5])
    timestamps[:] = -1.
    np.testing.assert_allclose(nurec_dataset.camera_timestamps("scene"), [0., .25, .5])


def test_nuscenes_sensor_references(nuscenes_dataset):
    frame = nuscenes_dataset.frames("scene")[0]
    data = nuscenes_dataset.frame_data("scene", frame)
    assert set(data.sensor_references) == {"CAM_FRONT"}
    assert data.sensor_references["CAM_FRONT"].sensor == "CAM_FRONT"


def test_nurec_usdz_path(nurec_dataset, nurec_usdz_path):
    assert nurec_dataset.usdz_path("scene") == nurec_usdz_path
    with pytest.raises(KeyError, match="Unknown NuRec scenario"):
        nurec_dataset.usdz_path("missing")


def test_nurec_scene_mesh(nurec_dataset, nurec_usdz_path, tmp_path):
    vertices, faces = nurec_dataset.scene_mesh("scene")
    assert vertices.shape == (12, 3) and faces.shape == (14, 3)
    np.testing.assert_allclose(vertices.min(axis=0), [-20., -20., 0.])
    np.testing.assert_allclose(vertices.max(axis=0), [20., 20., 2.])
    np.testing.assert_array_equal(faces[-2:], [[8, 9, 10], [8, 10, 11]])
    vertices[:] = 1000.
    np.testing.assert_allclose(nurec_dataset.scene_mesh("scene")[0].min(axis=0), [-20., -20., 0.])

    incomplete = tmp_path / "without-mesh.usdz"
    with zipfile.ZipFile(nurec_usdz_path) as source, zipfile.ZipFile(incomplete, "w") as destination:
        for member in source.infolist():
            if not member.filename.endswith(".ply"):
                destination.writestr(member.filename, source.read(member.filename))
    with pytest.raises(ValueError, match="has no mesh.ply"):
        NuRecDataset([str(incomplete)]).scene_mesh("scene")


def test_read_scene_mesh(nurec_usdz_path, tmp_path):
    mesh_only = tmp_path / "mesh-only.usdz"
    with zipfile.ZipFile(nurec_usdz_path) as source, zipfile.ZipFile(mesh_only, "w") as target:
        for name in ("mesh.ply", "mesh_ground.ply"):
            target.writestr(name, source.read(name))
    vertices, faces = read_scene_mesh(mesh_only)
    assert vertices.shape == (12, 3) and faces.shape == (14, 3)
    np.testing.assert_allclose(vertices.min(axis=0), [-20., -20., 0.])


def test_nurec_scenario_map(nurec_usdz_path, tmp_path):
    # A geographic OpenDRIVE projection alone does not establish alignment to the rig frame.
    fallback = tmp_path / "without-clipgt.usdz"
    with zipfile.ZipFile(nurec_usdz_path) as source, zipfile.ZipFile(fallback, "w") as destination:
        for member in source.infolist():
            if not member.filename.startswith("clipgt/"):
                destination.writestr(member.filename, source.read(member.filename))
    unavailable = NuRecDataset([str(fallback)]).scenario_map("scene")
    assert isinstance(unavailable, Unavailable) and "coordinate alignment" in unavailable.reason


def test_carla_recipe():
    recipe = CarlaRecipe({"ego": {}}, 2, .1,
        {"ego": PrescribedMotion(constant_control=Control(throttle=.4))})
    assert [command.throttle for command in recipe.motion["ego"].controls] == [.4, .4]
    recipe.motion["ego"].controls[0].throttle = .1
    assert recipe.motion["ego"].controls[1].throttle == .4
    with pytest.raises(ValueError, match="exactly these actors"):
        CarlaRecipe({"ego": {}, "lead": {"gap": 8}}, 2, .1, recipe.motion)
    with pytest.raises(ValueError, match="runtime settings"):
        CarlaRecipe({"ego": {}, "client": {"seed": 1}}, 2, .1, recipe.motion)
