"""The in-process NuRec backend: dynamics, the closed loop with a stub renderer, and checkpointing.

Fully in-process — no gRPC, no server, no scene — so it runs anywhere and is trivial to debug.
"""

import math

import pytest
from avsectester.backend import AVStack, run
from avsectester.plane import Control
from avsectester.simulators.nurec import EgoPose, KinematicBicycle, NuRecBackend, StubRenderer


class ThrottleStack(AVStack):
    def __call__(self, obs):
        return Control(throttle=1.0)


def test_bicycle_accelerates_brakes_and_turns():
    dyn = KinematicBicycle(max_accel=3.0, max_brake=8.0, max_steer=0.5)
    # throttle from rest -> speed increases; straight (steer 0) -> yaw unchanged
    p = dyn.step(EgoPose(), Control(throttle=1.0), dt=0.1)
    assert p.speed == pytest.approx(0.3) and p.yaw == 0.0 and p.x > 0
    # brake decays speed, clamped at 0
    assert dyn.step(EgoPose(speed=0.2), Control(brake=1.0), dt=0.1).speed == 0.0
    # steering while moving changes heading
    assert dyn.step(EgoPose(speed=5.0), Control(steer=1.0), dt=0.1).yaw != 0.0


def test_backend_loop_drives_in_process_with_stub_renderer():
    backend = NuRecBackend({"dt": 0.1})
    trace = run(backend, ThrottleStack(), frames=5)
    assert len(trace.records) == 5
    speeds = [r.speed for r in trace.records]
    assert speeds == sorted(speeds) and speeds[-1] > 0  # accelerating under throttle
    # the (stub) camera really flows through the Observation
    obs = backend._observe()
    assert "camera_front" in obs.sensor_data
    assert isinstance(backend.renderer, StubRenderer)


def test_checkpoint_and_restore_round_trip():
    backend = NuRecBackend({"dt": 0.1})
    run(backend, ThrottleStack(), frames=3)
    ckpt = backend.checkpoint()
    moved = backend.step(Control(throttle=1.0))
    assert moved.ego_speed > ckpt["pose"]["speed"]
    backend.restore(ckpt)
    assert backend.pose.speed == ckpt["pose"]["speed"]
    assert backend.frame == ckpt["frame"]


def test_trajectory_follower_tracks_a_straight_plan_and_coasts_when_empty():
    from avsectester.simulators.nurec import TrajectoryFollower

    follower = TrajectoryFollower()
    # straight rig-frame plan: +2 m forward at t=0.1 s, +4 m at 0.2 s (t in microseconds)
    plan = Control(trajectory=[((2.0, 0.0, 0.0), (1, 0, 0, 0), 100_000),
                               ((4.0, 0.0, 0.0), (1, 0, 0, 0), 200_000)])
    p = follower.step(EgoPose(t=0.0), plan, dt=0.1)  # interp at 0.1 s -> 2 m ahead
    assert p.x == pytest.approx(2.0) and p.y == pytest.approx(0.0)
    assert p.speed == pytest.approx(20.0)  # 2 m / 0.1 s
    # no plan -> coast (decelerate), not crash
    assert follower.step(EgoPose(speed=5.0), Control(), dt=0.1).speed == pytest.approx(4.6)


def test_stub_render_returns_a_frame_shaped_image():
    import numpy as np

    r = StubRenderer(cameras=["camera_front_wide_120fov"], height=120, width=160)
    frame = r.render(EgoPose(x=1.0, y=2.0, yaw=math.pi, t=0.5), "camera_front_wide_120fov")
    assert frame.shape == (120, 160, 3) and frame.dtype == np.uint8
