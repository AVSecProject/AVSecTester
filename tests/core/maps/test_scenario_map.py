"""Semantic map records own their geometry in canonical world metres."""

import numpy as np
import pytest

from avsectester.maps import Lane, ScenarioMap, StopLine, TrafficControl


def test_lane():
    points = np.array([[0., 0., 0.], [4., 0., 0.]])
    lane = Lane("lane", points, points + (0., 1., 0.), points - (0., 1., 0.),
                successors=("next",), speed_limit_mps=10.)
    points[:] = 50.
    np.testing.assert_array_equal(lane.centerline, [[0., 0., 0.], [4., 0., 0.]])
    np.testing.assert_array_equal(lane.left_boundary[:, 1], [1., 1.])
    np.testing.assert_array_equal(lane.right_boundary[:, 1], [-1., -1.])
    assert lane.successors == ("next",) and lane.speed_limit_mps == 10.
    with pytest.raises(ValueError, match="read-only"):
        lane.centerline[0, 0] = 3.
    with pytest.raises(ValueError, match="three-dimensional"):
        Lane("bad", np.zeros((2, 2)), np.zeros((2, 3)), np.zeros((2, 3)))


def test_traffic_control():
    pose = np.eye(4)
    pose[:3, 3] = (10., 3., 2.)
    control = TrafficControl("sign", "stop_sign", "206", pose, (.02, .8, .8), "face_center",
                              lane_ids=("lane",), stop_line_id="line")
    pose[:3, 3] = 0.
    np.testing.assert_array_equal(control.pose[:3, 3], [10., 3., 2.])
    np.testing.assert_array_equal(control.pose[:3, 0], [1., 0., 0.])
    assert control.extent == (.02, .8, .8) and control.stop_line_id == "line"
    assert control.pose_reference == "face_center"
    with pytest.raises(ValueError, match="read-only"):
        control.pose[0, 3] = 0.
    with pytest.raises(ValueError, match="thickness, width, height"):
        TrafficControl("sign", "stop_sign", "206", np.eye(4), (.02, -.8, .8), "face_center")
    with pytest.raises(ValueError, match="pose_reference"):
        TrafficControl("sign", "stop_sign", "206", np.eye(4), (.02, .8, .8), "origin")


def test_stop_line():
    points = np.array([[10., -1., 0.], [10., 1., 0.]])
    line = StopLine("line", points, "STOP", "stop_sign", ("lane",), "sign")
    points[:] = 0.
    np.testing.assert_array_equal(line.points, [[10., -1., 0.], [10., 1., 0.]])
    assert line.lane_ids == ("lane",) and line.control_id == "sign" and line.native_category == "STOP"
    with pytest.raises(ValueError, match="read-only"):
        line.points[0, 0] = 0.
    with pytest.raises(ValueError, match="at least 2"):
        StopLine("line", np.zeros((1, 3)), "STOP")


def test_scenario_map():
    polygon = np.array([[0., -1., 0.], [4., -1., 0.], [4., 1., 0.], [0., 1., 0.]])
    boundary = polygon[:2].copy()
    missing = {"traffic_controls.traffic_light": "This source has no traffic-light layer"}
    scenario_map = ScenarioMap(drivable_area=(polygon,), road_boundaries=(boundary,),
                                unavailable_layers=missing)
    polygon[:] = 9.
    boundary[:] = 8.
    missing.clear()
    np.testing.assert_array_equal(scenario_map.drivable_area[0][0], [0., -1., 0.])
    np.testing.assert_array_equal(scenario_map.road_boundaries[0], [[0., -1., 0.], [4., -1., 0.]])
    assert scenario_map.unavailable_layers["traffic_controls.traffic_light"]
    with pytest.raises(ValueError, match="read-only"):
        scenario_map.drivable_area[0][0, 0] = 0.
    with pytest.raises(ValueError, match="at least 3"):
        ScenarioMap(drivable_area=(np.zeros((2, 3)),))
    for key in ("unknown.layer", "unavailable_layers", "lanes.", "lanes.speed_limit_mps.extra",
                "traffic_controls.bogus"):
        with pytest.raises(ValueError, match="Unsupported unavailable-layer key"):
            ScenarioMap(unavailable_layers={key: "No evidence"})
    assert ScenarioMap(unavailable_layers={"crosswalks.partial": "One polygon was unclosed"})
