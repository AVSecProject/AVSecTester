"""Navigation follows map intent while remaining independent of actual ego motion."""
import numpy as np
import pytest

from avsectester.closed_loop.simulators.navigation import RouteGenerator
from avsectester.data_types import Unavailable
from avsectester.geometry.poses import make_pose, rotation_matrix
from avsectester.maps import Lane, ScenarioMap


def lane(identity, points, **options):
    centerline = np.asarray(points, dtype=float)
    return Lane(identity, centerline, centerline + [0, 1, 0], centerline - [0, 1, 0], **options)


def test_map_route():
    road = lane("road", [[0, 0, 0], [120, 0, 0]])
    generator = RouteGenerator("map", [[0, 0, 0], [10, 0, 0]], ScenarioMap(lanes={road.id: road}))
    route = generator.route(np.eye(4))
    np.testing.assert_allclose(route[:, 0], np.linspace(0, 80, 20))
    np.testing.assert_allclose(route[:, 1:], 0.0)
    np.testing.assert_allclose(generator.route(make_pose(np.eye(3), (10, 0, 0)))[:, 0],
                               np.linspace(0, 80, 20))
    assert RouteGenerator("none", (), None).route(np.eye(4)) is None

    with pytest.raises(ValueError, match="more than 10 m"):
        RouteGenerator("map", [[0, 20, 0], [1, 20, 0]], ScenarioMap(lanes={road.id: road}))
    with pytest.raises(ValueError, match="choose route_generator='recorded' or 'none'"):
        RouteGenerator("map", [[0, 0, 0], [1, 0, 0]], Unavailable("no map"))


def test_lane_sequence():
    first = lane("first", [[0, 0, 0], [20, 0, 0]])
    second = lane("second", [[0, 6, 0], [120, 6, 0]])
    scenario_map = ScenarioMap(lanes={first.id: first, second.id: second})
    generator = RouteGenerator("map", [[0, 0, 0], [10, 0, 0], [20, 0, 0], [25, 6, 0], [35, 6, 0]],
                               scenario_map)
    route = generator.route(np.eye(4))
    np.testing.assert_allclose(route[0], [0, 0, 0])
    assert route[-1, 1] == pytest.approx(6)
    short = lane("short", [[0, 0, 0], [10, 0, 0]])
    consistent = lane("consistent", [[0, 3, 0], [120, 3, 0]])
    generator = RouteGenerator("map", [[0, 0, 0], [5, 0, 0], [15, 3, 0]],
                               ScenarioMap(lanes={short.id: short, consistent.id: consistent}))
    np.testing.assert_allclose(generator.route(np.eye(4))[0], [0, 3, 0])


def test_successors():
    first = lane("first", [[0, 0, 0], [20, 0, 0]], successors=("turn", "straight"))
    straight = lane("straight", [[20, 0, 0], [120, 0, 0]])
    turn = lane("turn", [[20, 0, 0], [70, 70, 0]])
    generator = RouteGenerator("map", [[0, 0, 0], [10, 0, 0]],
                               ScenarioMap(lanes={value.id: value for value in (first, straight, turn)}))
    np.testing.assert_allclose(generator.route(np.eye(4)),
                               np.column_stack((np.linspace(0, 80, 20), np.zeros((20, 2)))))
    # A recording ending exactly at the lane endpoint can still reach its successor.
    generator = RouteGenerator("map", [[10, 0, 0], [20, 0, 0]],
                               ScenarioMap(lanes={value.id: value for value in (first, straight, turn)}))
    assert generator.route(make_pose(np.eye(3), (20, 0, 0)))[-1, 0] == pytest.approx(80)


def test_heading():
    opposing = lane("opposing", [[120, 0, 0], [0, 0, 0]])
    forward = lane("forward", [[0, 3, 0], [120, 3, 0]])
    shoulder = lane("shoulder", [[0, 0, 0], [120, 0, 0]], lane_type="shoulder")
    generator = RouteGenerator("map", [[0, 0, 0], [10, 0, 0]],
                               ScenarioMap(lanes={value.id: value for value in (opposing, forward, shoulder)}))
    np.testing.assert_allclose(generator.route(np.eye(4))[0], [0, 3, 0])


def test_recorded_route():
    generator = RouteGenerator("recorded", [[0, 0, 0], [10, 0, 0]], None)
    route = generator.route(np.eye(4))
    np.testing.assert_allclose(route[:3, 0], np.arange(3) * (80 / 19))
    np.testing.assert_allclose(route[3:], np.tile([10, 0, 0], (17, 1)))
    stationary = RouteGenerator("recorded", [[2, 3, 4], [2, 3, 4]], None)
    np.testing.assert_allclose(stationary.route(np.eye(4)), np.tile([2, 3, 0], (20, 1)))

    with pytest.raises(ValueError, match="folds back"):
        RouteGenerator("recorded", [[0, 0, 0], [5, 0, 0], [0, 0, 0]], None)
    with pytest.raises(ValueError, match="at least two finite"):
        RouteGenerator("recorded", [[0, 0, 0]], None)


def test_route_in_ego_coordinates():
    generator = RouteGenerator("recorded", [[0, 0, 0], [100, 0, 0]], None)
    route = generator.route(make_pose(rotation_matrix((0, 0, 90)), (10, 5, 2)))
    np.testing.assert_allclose(route[0], [-5, 0, 0], atol=1e-12)
    np.testing.assert_allclose(route[:, 0], -5, atol=1e-12)
    np.testing.assert_allclose(route[:, 1], -np.linspace(0, 80, 20), atol=1e-12)
