"""The real CARLA client parses an OpenDRIVE map without a running simulator."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("carla")

from avsectester.maps import ScenarioMap, read_opendrive


FIXTURES = Path(__file__).parent / "fixtures"


def test_read_opendrive():
    text = (FIXTURES / "straight_stop.xodr").read_text()
    scenario_map = read_opendrive(text, name="StraightStop")
    assert isinstance(scenario_map, ScenarioMap)
    assert set(scenario_map.lanes) == {"1:0:-1"}
    lane = scenario_map.lanes["1:0:-1"]
    np.testing.assert_allclose(lane.centerline[:, 1], -1.75)
    np.testing.assert_allclose(lane.left_boundary[:, 1], 0., atol=1e-6)
    np.testing.assert_allclose(lane.right_boundary[:, 1], -3.5, atol=1e-6)
    np.testing.assert_allclose(lane.centerline[[0, -1], 0], [0., 20.], atol=.001)
    assert lane.lane_type == "driving" and not lane.predecessors and not lane.successors
    assert len(scenario_map.drivable_area) == 1 and len(scenario_map.road_boundaries) == 1
    np.testing.assert_array_equal(scenario_map.road_boundaries[0], lane.right_boundary)
    control = scenario_map.traffic_controls["stop-1"]
    assert control.kind == "stop_sign" and control.native_category == "206"
    np.testing.assert_allclose(control.pose[:3, 3], [10., -3.5, 2.3])
    np.testing.assert_allclose(control.pose[:3, 0], [-1., 0., 0.], atol=1e-6)
    assert control.extent == (0., .6, .6) and control.lane_ids == (lane.id,)
    assert control.pose_reference == "face_center"
    assert control.stop_line_id == "stop-line:stop-1:1:-1"
    line = scenario_map.stop_lines[control.stop_line_id]
    assert line.control_id == control.id and line.kind == "stop_sign"
    assert line.native_category == "opendrive_signal_reference"
    np.testing.assert_allclose(line.points[:, 0], 10.)
    np.testing.assert_allclose(sorted(line.points[:, 1]), [-3.5, 0.], atol=1e-6)
    assert line.lane_ids == (lane.id,)
    assert "traffic_controls.extent" in scenario_map.unavailable_layers
    assert "traffic_controls" not in scenario_map.unavailable_layers
    assert "lanes.speed_limit_mps" in scenario_map.unavailable_layers


def test_signal_references():
    scenario_map = read_opendrive((FIXTURES / "referenced_controls.xodr").read_text(), name="Referenced")
    for identity, kind, reference, height in (
        ("stop-shared", "stop_sign", "face_center", 2.3),
        ("yield-shared", "yield_sign", "face_center", 2.3),
        ("light-shared", "traffic_light", "signal_position", 2.),
    ):
        control = scenario_map.traffic_controls[identity]
        assert control.kind == kind
        assert control.lane_ids == ("1:0:-1", "2:0:-1", "3:0:-1")
        assert control.stop_line_id is None
        assert control.pose_reference == reference
        assert control.pose[2, 3] == pytest.approx(height)
        assert control.extent == (0., .6, .6)
        np.testing.assert_allclose(control.pose[:3, 0], [-1., 0., 0.], atol=1e-6)
        for road, position in ((1, 6), (2, 12), (3, 10)):
            line = scenario_map.stop_lines[f"stop-line:{identity}:{road}:-1"]
            assert line.control_id == identity and line.lane_ids == (f"{road}:0:-1",)
            assert line.kind == kind and line.native_category == "opendrive_signal_reference"
            np.testing.assert_allclose(line.points[:, 0], position, atol=1e-6)
            assert np.linalg.norm(line.points[1] - line.points[0]) == pytest.approx(3.5)
    unbound = scenario_map.traffic_controls["unbound"]
    assert unbound.lane_ids == () and unbound.stop_line_id is None
    assert unbound.pose_reference == "face_center" and unbound.pose[2, 3] == pytest.approx(2.3)
    np.testing.assert_allclose(unbound.pose[:3, 0], [-np.cos(.4), -np.sin(.4), 0.], atol=1e-6)
    zero_height = scenario_map.traffic_controls["zero-height"]
    assert zero_height.pose_reference == "signal_position" and zero_height.pose[2, 3] == pytest.approx(2.)
    assert zero_height.extent == (0., .6, 0.)

    for identity, lane_ids in (("opposite", ("1:0:-1", "1:0:1")),
                               ("ambiguous", ("1:0:-1", "1:0:1", "3:0:-1"))):
        control = scenario_map.traffic_controls[identity]
        assert control.lane_ids == lane_ids
        # Balanced directions, and a 2:1 mixture with mean length 1/3, both use
        # the signal's source heading turned by 180 degrees rather than a false mean.
        np.testing.assert_allclose(control.pose[:3, 0], [-np.cos(.6), -np.sin(.6), 0.], atol=1e-6)


def test_lane_geometry():
    scenario_map = read_opendrive((FIXTURES / "curved_lanes.xodr").read_text(), name="Curved")
    assert len(scenario_map.lanes) == 12
    assert {lane.lane_type for lane in scenario_map.lanes.values()} == {"driving", "shoulder", "other"}
    for lane in scenario_map.lanes.values():
        for points in (lane.centerline, lane.left_boundary, lane.right_boundary):
            distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
            assert 0 < distances.min() and distances.max() <= 2.00001
        for neighbor in (lane.left_neighbor, lane.right_neighbor):
            assert neighbor is None or int(neighbor.split(":")[-1]) * int(lane.id.split(":")[-1]) > 0
    assert scenario_map.lanes["1:0:1"].left_neighbor is None
    assert scenario_map.lanes["1:0:-1"].left_neighbor is None
    assert len(scenario_map.road_boundaries) == 2
    for boundary, identity in zip(scenario_map.road_boundaries, ("1:0:-2", "1:0:2")):
        np.testing.assert_array_equal(boundary, scenario_map.lanes[identity].right_boundary)
    # One junction gives one intersection area: the convex hull of its driving lanes.
    (intersection,) = scenario_map.intersections
    edges = np.roll(intersection[:, :2], -1, axis=0) - intersection[:, :2]
    for identity in ("2:0:-1", "2:0:1"):
        lane = scenario_map.lanes[identity]
        for point in np.concatenate((lane.left_boundary, lane.right_boundary))[:, :2]:
            offsets = point - intersection[:, :2]
            assert np.all(edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0] >= -1e-6)

    # A true width/offset step has no intermediate native waypoint. Its boundary
    # connector remains the same line, with enough collinear vertices to bound spacing.
    stepped = read_opendrive((FIXTURES / "discontinuous_lanes.xodr").read_text(), name="Stepped")
    lane = stepped.lanes["1:0:1"]
    for points in (lane.centerline, lane.left_boundary, lane.right_boundary):
        assert np.linalg.norm(np.diff(points, axis=0), axis=1).max() <= 2.00001
    step = lane.left_boundary[np.abs(lane.left_boundary[:, 0] - 10.) < 1e-4]
    assert len(step) >= 3
    np.testing.assert_allclose([step[:, 1].min(), step[:, 1].max()], [-7., -4.135], atol=1e-5)
