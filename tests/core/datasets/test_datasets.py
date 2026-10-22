"""Dataset components share one public timeline contract; interpolation stays numerical."""
from dataclasses import replace
import zipfile

import numpy as np
import pytest

from avsectester.closed_loop.datatypes import Control
from avsectester.datasets import FrameData, Unavailable
from avsectester.datasets.carla import CarlaRecipe, PrescribedMotion
from avsectester.datasets.nurec import NuRecDataset, read_scene_mesh
from avsectester.geometry.poses import rotation_matrix
from avsectester.maps import ScenarioMap
from tests.fakes import assert_scenario_map


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
    # Returned annotations and arrays remain independent of the immutable recording.
    data = dataset.frame_data("scene", frames[5])
    data.actors["host"].pose[:] = 0
    metadata = dataset.raw_metadata(nurec_usdz_path)
    metadata["sequence_tracks.json"].clear()
    np.testing.assert_allclose(dataset.frame_data("scene", frames[5]).actors["host"].pose[0, 3], 9.)


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


def test_nurec_scenario_map(nurec_dataset, nurec_usdz_path, nurec_map_variants, tmp_path):
    scenario_map = nurec_dataset.scenario_map("scene")
    lane = scenario_map.lanes["lane"]
    np.testing.assert_allclose(lane.centerline, [[-2., 0., 0.], [4., 0., 0.]])
    assert lane.speed_limit_mps == pytest.approx(25 * .44704)
    assert lane.lane_type == "driving" and lane.right_neighbor is None
    assert scenario_map.lanes["bike-lane"].lane_type == "bike"
    assert scenario_map.lanes["bike-lane"].speed_limit_mps is None
    assert scenario_map.unavailable_layers["lanes.speed_limit_mps"]
    assert scenario_map.lanes["shoulder-lane"].lane_type == "shoulder"
    assert scenario_map.lanes["parking-lane"].lane_type == "parking"
    assert scenario_map.lanes["bus-lane"].lane_type == "other"
    sign = scenario_map.traffic_controls["stop"]
    assert sign.kind == "stop_sign" and sign.native_category == "TRAFFIC_SIGN_REGULATORY_R1_STOP"
    assert sign.extent == (.1, .8, 1.2) and sign.lane_ids == ("lane",)
    assert {control.pose_reference for control in scenario_map.traffic_controls.values()} == {"face_center"}
    assert sign.stop_line_id == "stop-line"
    np.testing.assert_allclose(sign.pose[:3, 3], [3., 2., 1.5])
    assert scenario_map.traffic_controls["no-stopping"].kind == "other"
    assert scenario_map.traffic_controls["speed-limit"].kind == "speed_limit"
    assert scenario_map.traffic_controls["yield"].kind == "yield_sign"
    line = scenario_map.stop_lines["stop-line"]
    assert line.kind == "stop_sign" and line.lane_ids == ("lane",) and line.control_id == "stop"
    assert line.native_category == "STOP"
    signal = scenario_map.stop_lines["signal-line"]
    assert signal.kind == "traffic_light" and signal.control_id == "light"
    assert scenario_map.stop_lines["yield-line"].kind == "yield_sign"
    ambiguous = scenario_map.stop_lines["ambiguous-line"]
    assert ambiguous.kind == "traffic_light" and ambiguous.control_id is None
    unknown = scenario_map.stop_lines["unknown-line"]
    assert unknown.kind == "unknown" and unknown.control_id is None
    unbound = scenario_map.stop_lines["unbound-line"]
    assert unbound.kind == "unknown" and unbound.lane_ids == () and unbound.native_category == "STOP"
    assert set(scenario_map.stop_lines) == {"stop-line", "signal-line", "yield-line",
                                          "ambiguous-line", "unknown-line", "unbound-line"}
    assert scenario_map.drivable_area == ()
    assert scenario_map.unavailable_layers["drivable_area"]
    assert scenario_map.unavailable_layers["crosswalks"]

    placeholder = NuRecDataset([nurec_map_variants["placeholder_lights"]]).scenario_map("scene")
    assert len(placeholder.traffic_controls) == 4
    assert placeholder.unavailable_layers["traffic_controls.traffic_light"]
    assert "traffic_controls" not in placeholder.unavailable_layers
    implicit = NuRecDataset([nurec_map_variants["implicit_only_lines"]]).scenario_map("scene")
    assert implicit.stop_lines == {} and "stop_lines" not in implicit.unavailable_layers
    assert all(control.stop_line_id is None for control in implicit.traffic_controls.values())

    unknown_units = tmp_path / "without-speed-unit.usdz"
    with zipfile.ZipFile(nurec_usdz_path) as source, zipfile.ZipFile(unknown_units, "w") as destination:
        for member in source.infolist():
            if member.filename != "map.xodr":
                destination.writestr(member.filename, source.read(member.filename))
    unscaled = NuRecDataset([str(unknown_units)]).scenario_map("scene")
    assert unscaled.lanes["lane"].speed_limit_mps is None
    assert unscaled.unavailable_layers["lanes.speed_limit_mps"]

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
