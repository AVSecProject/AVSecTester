"""The in-process NuRec backend: dynamics, the closed loop with a stub renderer, and checkpointing.

Fully in-process — no gRPC, no server, no scene — so it runs anywhere and is trivial to debug.
"""

import math

import numpy as np
import pytest
from avsectester.backend import AVStack, run
from avsectester.plane import Control
from avsectester.simulators.nurec import EgoPose, KinematicBicycle, NuRecBackend, StubRenderer


class ThrottleStack(AVStack):
    def __call__(self, obs):
        return Control(throttle=1.0)


def test_bicycle_accelerates_brakes_and_turns():
    dyn = KinematicBicycle(max_accel=3.0, max_brake=8.0, max_steer=0.5)
    # throttle from rest -> speed increases. Straight (steer 0) -> yaw unchanged
    p = dyn.step(EgoPose(), Control(throttle=1.0), dt=0.1)
    assert p.speed == pytest.approx(0.3) and p.yaw == 0.0 and p.x > 0
    # brake decays speed, clamped at 0
    assert dyn.step(EgoPose(speed=0.2), Control(brake=1.0), dt=0.1).speed == 0.0
    # steering while moving changes heading
    assert dyn.step(EgoPose(speed=5.0), Control(steer=1.0), dt=0.1).yaw != 0.0


def test_backend_loop_drives_in_process_with_stub_renderer():
    backend = NuRecBackend({"dt": 0.1}, renderer=StubRenderer(height=120, width=160))
    trace = run(backend, ThrottleStack(), frames=5)
    assert len(trace.records) == 5
    speeds = [r.speed for r in trace.records]
    assert speeds == sorted(speeds) and speeds[-1] > 0  # accelerating under throttle
    # the (stub) camera really flows through the Observation
    obs = backend._observe()
    assert "camera_front" in obs.sensor_data
    assert isinstance(backend.renderer, StubRenderer)
    image = obs.sensor_data["camera_front"]
    assert image.shape == (120, 160, 3) and image.dtype == np.uint8


def test_restoring_checkpoint_replays_the_same_subsequent_motion():
    backend = NuRecBackend({"dt": 0.1, "ego0": {"x": 10.0, "y": -3.0, "yaw": 0.4}})
    run(backend, ThrottleStack(), frames=3)
    checkpoint = backend.checkpoint()
    controls = [Control(throttle=0.7, steer=0.3), Control(brake=0.2, steer=-0.1)]
    expected = [backend.step(control) for control in controls]

    backend.restore(checkpoint)
    assert backend.checkpoint() == checkpoint
    replayed = [backend.step(control) for control in controls]

    for actual, original in zip(replayed, expected):
        assert actual.vehicle_state == original.vehicle_state
        assert actual.frame == original.frame
        assert actual.t == original.t
        assert actual.ego_speed == original.ego_speed


def test_reset_restores_configured_pose_after_a_previous_drive():
    backend = NuRecBackend({"dt": 0.1, "ego0": {"x": 10.0, "y": -3.0, "yaw": 0.4}})
    initial = backend.reset()
    backend.step(Control(throttle=1.0, steer=0.5))

    restarted = backend.reset()

    assert restarted.frame == 0
    assert restarted.vehicle_state == initial.vehicle_state


@pytest.mark.parametrize("dt", [0.05, 0.1])
def test_trajectory_follower_interpolates_from_current_origin_before_first_waypoint(dt):
    from avsectester.simulators.nurec import TrajectoryFollower

    pose = EgoPose(x=10.0, y=5.0, yaw=math.pi / 2, t=2.0)
    plan = Control(trajectory=[
        ((2.0, 0.0, 0.0), (1, 0, 0, 0), 2_100_000),
        ((4.0, 0.0, 0.0), (1, 0, 0, 0), 2_200_000),
    ])
    moved = TrajectoryFollower().step(pose, plan, dt)

    assert moved.x == pytest.approx(10.0)
    assert moved.y == pytest.approx(5.0 + 20.0 * dt)
    assert moved.speed == pytest.approx(20.0)
    assert moved.t == pytest.approx(2.0 + dt)


def test_trajectory_replanning_does_not_multiply_speed_when_physics_runs_faster_than_policy():
    from avsectester.simulators.nurec import TrajectoryFollower

    follower = TrajectoryFollower()
    pose = EgoPose(speed=3.0)
    for _ in range(40):
        plan = Control(trajectory=[
            ((pose.speed * 0.1, 0.0, 0.0), (1, 0, 0, 0), round((pose.t + 0.1) * 1e6)),
        ])
        pose = follower.step(pose, plan, dt=0.05)

    assert pose.speed == pytest.approx(3.0)
    assert pose.x == pytest.approx(6.0)
    # An absent plan coasts instead of repeating an old trajectory.
    assert follower.step(EgoPose(speed=5.0), Control(), dt=0.1).speed == pytest.approx(4.6)


@pytest.mark.parametrize("yaw,expected_xy", [(0.0, (13.0, -3.0)), (math.pi / 2, (9.0, -1.0))])
def test_trajectory_follower_interpolates_in_rig_then_transforms_to_world(yaw, expected_xy):
    from avsectester.simulators.nurec import TrajectoryFollower

    # At 2.15 s, halfway between waypoints, the rig-frame offset is (3, 1).
    plan = Control(
        trajectory=[
            ((2.0, 0.0, 0.0), (1, 0, 0, 0), 2_100_000),
            ((4.0, 2.0, 0.0), (1, 0, 0, 0), 2_200_000),
        ]
    )
    pose = EgoPose(x=10.0, y=-4.0, yaw=yaw, t=2.0)

    moved = TrajectoryFollower().step(pose, plan, dt=0.15)

    assert (moved.x, moved.y) == pytest.approx(expected_xy)
    assert moved.t == pytest.approx(2.15)
    assert moved.speed == pytest.approx(math.sqrt(10) / 0.15)
    assert moved.yaw == pytest.approx(yaw + math.atan2(1, 3))
