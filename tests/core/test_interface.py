"""The core interface (data plane + WorldBackend/AVStack + run loop) works without any backend.

Uses a trivial in-memory backend and stack — no CARLA/avstack — proving the interface is
independent of both the simulator and whether the AV stack is modular or end-to-end.
"""

from dataclasses import replace

import pytest
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


def test_run_loop_drives_and_records_true_state():
    trace = run(ThrottleBackend(), GoStack(), frames=5)
    assert len(trace.records) == 5
    assert [r.speed for r in trace.records] == [1, 2, 3, 4, 5]
    assert [r.t for r in trace.records] == pytest.approx([0.05, 0.10, 0.15, 0.20, 0.25])
    assert trace.braking_frames == 0


def test_perturb_replacement_controls_the_drive_but_not_recorded_state():
    class SpeedLimitStack(AVStack):
        def reset(self, observation):
            self.initial = observation
            self.seen = []

        def __call__(self, observation):
            self.seen.append(observation)
            return Control(brake=1.0) if observation.ego_speed >= 2 else Control(throttle=1.0)

    originals = []

    def perturb(obs: Observation) -> Observation:
        originals.append(obs)
        return replace(obs, ego_speed=100.0)

    clean = run(ThrottleBackend(), SpeedLimitStack(), frames=3)
    stack = SpeedLimitStack()
    attacked = run(ThrottleBackend(), stack, frames=3, perturb=perturb)

    assert [r.speed for r in clean.records] == [1, 2, 1]
    assert stack.initial is originals[0]  # reset sees the initial backend observation
    assert [o.frame for o in originals] == [0, 1, 2]
    assert [o.frame for o in stack.seen] == [0, 1, 2]
    assert all(seen is not original for seen, original in zip(stack.seen, originals))
    assert [o.ego_speed for o in stack.seen] == [100, 100, 100]
    assert [o.ego_speed for o in originals] == [0, 0, 0]
    assert attacked.braking_frames == 3
    assert [r.speed for r in attacked.records] == [0, 0, 0]



def test_on_step_observes_attacked_input_and_control_before_world_advances():
    backend = ThrottleBackend()
    recorded = []

    def on_step(index, seen, control):
        recorded.append((index, seen.frame, seen.ego_speed, control.throttle, backend.i))

    trace = run(backend, GoStack(), frames=2,
                perturb=lambda obs: replace(obs, ego_speed=100), on_step=on_step)
    assert recorded == [(0, 0, 100, 1.0, 0), (1, 1, 100, 1.0, 1)]
    assert [row.speed for row in trace.records] == [1.0, 2.0]
