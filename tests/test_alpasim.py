"""Pure mapping helpers for the AlpaSim backend (no gRPC / no alpasim install needed).

Covers the AlpaSim<->interface data mapping: egomotion -> EgoState/speed, and Control.trajectory ->
waypoints. The live relay (gRPC EgodriverService + runtime) is not exercised here.
"""

from types import SimpleNamespace

import pytest
from avsectester.alpasim import control_waypoints, ego_from_egomotion, speed_of
from avsectester.plane import Control


def _vec(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def _ego_msg(poses, velocities):
    return SimpleNamespace(
        trajectory=SimpleNamespace(
            poses=[
                SimpleNamespace(
                    pose=SimpleNamespace(vec=_vec(*p), quat=SimpleNamespace(w=1, x=0, y=0, z=0)),
                    timestamp_us=t,
                )
                for p, t in poses
            ]
        ),
        dynamic_states=[SimpleNamespace(linear_velocity=_vec(*v)) for v in velocities],
    )


def test_ego_from_egomotion_uses_latest_pose_and_velocity():
    ego = ego_from_egomotion(_ego_msg([((0, 0, 0), 10), ((5, 1, 0), 20)], [(3, 0, 0), (4, 3, 0)]))
    assert ego.position == (5, 1, 0)
    assert ego.orientation == (1, 0, 0, 0)
    assert ego.t_us == 20
    assert ego.velocity == (4, 3, 0)
    assert speed_of(ego) == pytest.approx(5.0)  # 3-4-5


def test_ego_from_egomotion_is_none_when_no_poses():
    empty = SimpleNamespace(trajectory=SimpleNamespace(poses=[]), dynamic_states=[])
    assert ego_from_egomotion(empty) is None
    assert speed_of(None) == 0.0


def test_control_waypoints_normalizes_both_forms():
    control = Control(trajectory=[((1, 2, 3), (1, 0, 0, 0), 100), ((4, 5, 6), 200)])
    waypoints = control_waypoints(control)
    assert waypoints[0] == ((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0), 100)
    assert waypoints[1] == ((4.0, 5.0, 6.0), (1.0, 0.0, 0.0, 0.0), 200)  # identity rotation filled


def test_control_waypoints_rejects_actuator_only_control():
    with pytest.raises(ValueError, match="trajectory"):
        control_waypoints(Control(throttle=0.5, brake=0.0))
