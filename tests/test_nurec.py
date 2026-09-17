"""The in-process NuRec backend: dynamics, the closed loop with a stub renderer, and checkpointing.

Fully in-process — no gRPC, no server, no scene — so it runs anywhere and is trivial to debug.
"""

import math

import pytest
from avsectester.backend import AVStack, run
from avsectester.nurec import EgoPose, KinematicBicycle, NuRecBackend, StubRenderer
from avsectester.plane import Control


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


def test_stub_render_is_deterministic_for_a_pose():
    r = StubRenderer()
    pose = EgoPose(x=1.23456, y=2.0, yaw=math.pi, t=0.5)
    assert r.render(pose, "camera_front") == r.render(pose, "camera_front")
