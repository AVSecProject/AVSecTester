"""Dataset components share timeline and map contracts; adapters own resource details."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from avsectester.data_types import Control, Unavailable
from avsectester.datasets import FrameData, FrameReference, ScenarioStart
from avsectester.datasets.carla import CarlaDataset, CarlaRecipe, PrescribedMotion
from avsectester.datasets.nurec import NuRecDataset, read_scene_mesh
from avsectester.geometry.poses import rotation_matrix
from avsectester.maps import ScenarioMap
from tests.core.datasets.scenario_map_contract import assert_scenario_map
from tests.fakes import actor_state


@pytest.fixture(params=[
    pytest.param(("carla_dataset", Unavailable, Unavailable, "recipe_only", CarlaRecipe), id="carla"),
    pytest.param(("nurec_dataset", ScenarioMap, FrameData, None, dict), id="nurec"),
    pytest.param(("nuscenes_dataset", Unavailable, FrameData, None, dict), id="nuscenes"),
])
def dataset_case(request):
    name, map_type, frame_type, reason, native_type = request.param
    return SimpleNamespace(dataset=request.getfixturevalue(name), map_type=map_type,
                           frame_type=frame_type, reason=reason, native_type=native_type)


def test_dataset_contract(dataset_case):
    dataset = dataset_case.dataset
    scenario = next(dataset.scenarios())
    frames = dataset.frames(scenario.id)
    assert [frame.position for frame in frames] == list(range(len(frames)))
    assert all(right.timestamp > left.timestamp for left, right in zip(frames, frames[1:]))
    assert dataset.decision_window(scenario.id, frames[0], len(frames) - 1) == frames
    start = dataset.scenario_start(scenario.id, frames[1])
    assert start.frame == frames[1] and start.scenario_id == scenario.id
    assert isinstance(dataset.scenario_map(scenario.id), dataset_case.map_type)
    assert start.scenario_map == dataset.scenario_map(scenario.id)
    native = dataset.native(scenario.id)
    assert isinstance(native, dataset_case.native_type) and native
    data = dataset.frame_data(scenario.id, frames[1])
    assert isinstance(data, dataset_case.frame_type)
    assert getattr(data, "reason", None) == dataset_case.reason
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


@pytest.fixture
def carla_map_dataset(tmp_path):
    pytest.importorskip("carla")
    fixture = Path(__file__).resolve().parents[2] / "avstack" / "fixtures" / "straight_stop.xodr"
    (tmp_path / "StraightStop.xodr").write_text(fixture.read_text())
    initial_ego = actor_state((2., -1.75, 0.))
    recipe = CarlaRecipe({"client": {"map_name": "StraightStop"}, "ego": {}}, 1, .1,
                         {"ego": PrescribedMotion(trajectory=(initial_ego, initial_ego))})
    return CarlaDataset({"scene": recipe}, map_directory=tmp_path)


@pytest.fixture(params=[
    pytest.param(("nurec_dataset", lambda dataset: dataset.frame_data(
        "scene", dataset.frames("scene")[0]).ego), id="nurec"),
    pytest.param(("carla_map_dataset", lambda dataset: dataset.native(
        "scene").motion["ego"].trajectory[0]), id="carla"),
])
def map_case(request):
    name, ego = request.param
    dataset = request.getfixturevalue(name)
    return dataset, ego(dataset)


def test_scenario_map_contract(map_case):
    dataset, ego = map_case
    assert_scenario_map(dataset.scenario_map("scene"), ego)


def test_nurec_timeline(nurec_usdz_path, tmp_path):
    with pytest.raises(ValueError, match="positive and finite"):
        NuRecDataset([nurec_usdz_path], time_step=0.)
    with pytest.raises(ValueError, match="integer microseconds"):
        NuRecDataset([nurec_usdz_path], time_step=1e-8)
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

    path = tmp_path / "track-flags.usdz"
    with zipfile.ZipFile(nurec_usdz_path) as source, zipfile.ZipFile(path, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name == "sequence_tracks.json":
                tracks = json.loads(data)
                tracks["scene"]["tracks_data"]["tracks_flags"] = ["DYNAMIC|CONTROLLABLE", "NONE"]
                data = json.dumps(tracks).encode()
            target.writestr(name, data)
    dataset = NuRecDataset([path])
    tracks = dataset.track_trajectories("scene")
    assert set(tracks) == {"host", "brief"}
    host = tracks["host"]
    assert host["controllable"] is True
    assert host["label_class"] == "automobile" and host["category"] == "vehicle"
    assert host["extent"] == (4., 2., 2.)
    assert [time for time, pose in host["trajectory"]] == [0, 500000]
    np.testing.assert_allclose(host["trajectory"][0][1][:3, 3], [8., 0., 1.])
    np.testing.assert_allclose(host["trajectory"][-1][1][:3, :3], rotation_matrix((0., 0., 180.)), atol=1e-12)
    assert tracks["brief"]["category"] == "pedestrian" and tracks["brief"]["controllable"] is False
    host["trajectory"][0][1][:] = 0.
    np.testing.assert_allclose(dataset.track_trajectories("scene")["host"]["trajectory"][0][1][:3, 3], [8., 0., 1.])


def test_nurec_resources(nurec_dataset, nurec_usdz_path, tmp_path):
    assert nurec_dataset.usdz_path("scene") == nurec_usdz_path
    with pytest.raises(KeyError, match="Unknown NuRec scenario"):
        nurec_dataset.usdz_path("missing")

    # A dataroot lists every archive below it, ordered by path.
    root = Path(nurec_usdz_path).parent
    nested = root / "nested" / "scene.usdz"
    nested.parent.mkdir()
    shutil.copy(nurec_usdz_path, nested)
    with pytest.raises(ValueError, match="occurs in"):
        list(NuRecDataset(dataroot=root).scenarios())
    nested.unlink()
    from_root = NuRecDataset(dataroot=root)
    assert [item.id for item in from_root.scenarios()] == ["scene"]
    assert from_root.usdz_path("scene") == str(nurec_usdz_path)
    for options in ({}, {"usdz_paths": [nurec_usdz_path], "dataroot": root}):
        with pytest.raises(ValueError, match="exactly one of dataroot"):
            NuRecDataset(**options)
    # The directory is read on first use, so a configuration loads without data present.
    empty = NuRecDataset(dataroot=nested.parent)
    with pytest.raises(ValueError, match="contains no .usdz archives"):
        list(empty.scenarios())

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

    mesh_only = tmp_path / "mesh-only.usdz"
    with zipfile.ZipFile(nurec_usdz_path) as source, zipfile.ZipFile(mesh_only, "w") as target:
        for name in ("mesh.ply", "mesh_ground.ply"):
            target.writestr(name, source.read(name))
    vertices, faces = read_scene_mesh(mesh_only)
    assert vertices.shape == (12, 3) and faces.shape == (14, 3)
    np.testing.assert_allclose(vertices.min(axis=0), [-20., -20., 0.])

    # A geographic OpenDRIVE projection alone does not establish alignment to the rig frame.
    fallback = tmp_path / "without-clipgt.usdz"
    with zipfile.ZipFile(nurec_usdz_path) as source, zipfile.ZipFile(fallback, "w") as destination:
        for member in source.infolist():
            if not member.filename.startswith("clipgt/"):
                destination.writestr(member.filename, source.read(member.filename))
    unavailable = NuRecDataset([str(fallback)]).scenario_map("scene")
    assert isinstance(unavailable, Unavailable) and "coordinate alignment" in unavailable.reason


def test_nurec_camera_calibration(nurec_dataset, nurec_usdz_path, tmp_path):
    camera, rig_from_camera = nurec_dataset.camera_geometry("scene")
    assert (camera.width, camera.height) == (16, 12)
    np.testing.assert_allclose(rig_from_camera,
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]])
    rig_from_camera[:] = 0.
    assert nurec_dataset.camera_geometry("scene")[1][2, 3] == 1.5
    with pytest.raises(KeyError, match="Unknown NuRec scenario"):
        nurec_dataset.camera_geometry("missing")

    timestamps = nurec_dataset.camera_timestamps("scene")
    np.testing.assert_allclose(timestamps, [0., .25, .5])
    timestamps[:] = -1.
    np.testing.assert_allclose(nurec_dataset.camera_timestamps("scene"), [0., .25, .5])

    windshield = {"reference_poly": "FORWARD", "horizontal_poly": [.01, 1., 0., 0., 0., 0.],
                  "vertical_poly": [0., 0., 0., 1., 0., 0.], "horizontal_poly_inverse": [-.01, 1., 0., 0., 0., 0.],
                  "vertical_poly_inverse": [0., 0., 0., 1., 0., 0.]}
    distorted = tmp_path / "windshield" / "scene.usdz"
    distorted.parent.mkdir()
    with zipfile.ZipFile(nurec_usdz_path) as source, zipfile.ZipFile(distorted, "w") as target:
        for name in source.namelist():
            data = source.read(name)
            if name == "rig_trajectories.json":
                rig = json.loads(data)
                for calibration in rig["camera_calibrations"].values():
                    calibration["camera_model"]["parameters"]["external_distortion_parameters"] = windshield
                data = json.dumps(rig).encode()
            target.writestr(name, data)
    camera, _ = NuRecDataset([distorted], time_step=.1).camera_geometry("scene")
    assert camera.windshield is not None and camera.windshield.horizontal_poly == (.01, 1., 0., 0., 0., 0.)
    # The recorded camera keeps the archive's principal point; NRE's half-pixel offset is renderer-specific.
    assert (camera.cx, camera.cy) == (8., 6.)
    plain, _ = NuRecDataset([nurec_usdz_path], time_step=.1).camera_geometry("scene")
    assert plain.windshield is None

    archive_path = tmp_path / "side_only.usdz"
    with zipfile.ZipFile(nurec_usdz_path) as source:
        metadata = json.loads(source.read("rig_trajectories.json"))
        previous = next(iter(metadata["camera_calibrations"]))
        replacement = "side@scene"
        metadata["camera_calibrations"] = {
            replacement: metadata["camera_calibrations"][previous]}
        for trajectory in metadata["rig_trajectories"]:
            for field in ("cameras_frame_timestamps_us", "cameras_frame_T_rig_worlds"):
                trajectory[field] = {replacement: trajectory[field][previous]}
        with zipfile.ZipFile(archive_path, "w") as target:
            for name in source.namelist():
                target.writestr(name, json.dumps(metadata) if name == "rig_trajectories.json"
                                else source.read(name))
    side = NuRecDataset([archive_path], camera_id="side", time_step=.1)
    assert list(side.raw_metadata("scene")["rig_trajectories.json"][
        "camera_calibrations"]) == ["side@scene"]
    camera, _ = side.camera_geometry("scene")
    np.testing.assert_allclose(camera.unproject([[8., 6.]]), [[0., 0., 1.]])


def test_nuscenes_sensor_references(nuscenes_dataset):
    frame = nuscenes_dataset.frames("scene")[0]
    data = nuscenes_dataset.frame_data("scene", frame)
    assert set(data.sensor_references) == {"CAM_FRONT"}
    assert data.sensor_references["CAM_FRONT"].sensor == "CAM_FRONT"


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


def test_frame_reference():
    reference = FrameReference("scene", 2, "native-2", .2)
    assert reference.scenario_id == "scene" and reference.position == 2
    assert reference.frame_id == "native-2" and reference.timestamp == .2
    with pytest.raises(ValueError, match="position"):
        FrameReference("scene", -1, "native", .1)
    with pytest.raises(ValueError, match="finite"):
        FrameReference("scene", 0, 0, float("nan"))


def test_scenario_start():
    frame = FrameReference("scene", 0, "native", 0.)
    payload = {"nested": {"positions": [1., 2.]}}
    start = ScenarioStart("UserDataset", "scene", frame, payload)
    payload["nested"]["positions"].append(3.)
    assert start.frame is frame and start.payload == {"nested": {"positions": [1., 2.]}}
    with pytest.raises(ValueError, match="belong to the scenario"):
        ScenarioStart("UserDataset", "other", frame, {})
