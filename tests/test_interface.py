"""The core interface (data plane + WorldBackend/AVStack + run loop) works without any backend.

Uses a trivial in-memory backend and stack — no CARLA/avstack — proving the interface is
independent of both the simulator and whether the AV stack is modular or end-to-end.
"""

from avsectester.backend import AVStack, WorldBackend, run
from avsectester.plane import Control, Observation


class ThrottleBackend(WorldBackend):
    """Toy dynamics: speed integrates throttle minus brake (the 'shared physical control')."""

    def __init__(self):
        self.i, self.speed = 0, 0.0

    def reset(self):
        self.i, self.speed = 0, 0.0
        return Observation(t=0.0, frame=0, ego_speed=0.0)

    def step(self, control: Control) -> Observation:
        self.i += 1
        self.speed = max(0.0, self.speed + control.throttle - control.brake)
        return Observation(t=self.i * 0.05, frame=self.i, ego_speed=self.speed)


class GoStack(AVStack):
    def __call__(self, obs: Observation) -> Control:
        return Control(throttle=1.0)


class BrakeStack(AVStack):
    def __call__(self, obs: Observation) -> Control:
        return Control(brake=1.0)


def test_run_loop_drives_and_records_true_state():
    trace = run(ThrottleBackend(), GoStack(), frames=5)
    assert len(trace.records) == 5
    assert trace.final_speed > 0 and trace.braking_frames == 0


def test_perturb_is_the_only_attack_seam():
    calls = {"n": 0}

    def perturb(obs: Observation) -> Observation:
        calls["n"] += 1  # an attack would edit obs.sensor_data here
        return obs

    trace = run(ThrottleBackend(), BrakeStack(), frames=3, perturb=perturb)
    assert calls["n"] == 3  # the box saw the perturbed stream every frame
    assert trace.braking_frames == 3  # Trace still records the backend's true state
