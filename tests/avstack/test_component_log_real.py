"""Component logging on REAL execution — a real avstack ``ModularDrivingPipeline`` (Passthrough detector
+ real BasicBoxTracker3D + ForwardCollisionPlanner + VehiclePIDController), instrumented and driven by
``run_logged``. No stubs: the captured per-stage outputs are genuine avstack objects, and the phantom
case runs the real ``PhantomInjection`` attack and checks it propagates perception -> tracking ->
control. Needs avstack (present in the full env); skipped otherwise.
"""

import pytest

pytest.importorskip("avstack")

from avsectester.backend import WorldBackend
from avsectester.evaluation.component_log import run_logged
from avsectester.plane import Observation
from avsectester.stacks.modular import ModularAVStack

_CFG = dict(
    type="ModularDrivingPipeline",
    perception=dict(type="Passthrough3DObjectDetector"),
    tracking=dict(type="BasicBoxTracker3D"),
    planning=dict(type="ForwardCollisionPlanner", target_speed=6.0),
    control=dict(type="VehiclePIDController",
                 args_lateral=dict(K_P=1.0, K_D=0.0, K_I=0.0),
                 args_longitudinal=dict(K_P=0.5, K_D=0.0, K_I=0.0)),
)


def _backend(make_detections, make_ego, leads_per_frame):
    """A backend feeding the real pipeline real detections (``leads_per_frame[i]`` = lead xyzs for frame i)
    + a real ego VehicleState, so perception->tracking->planning->control actually runs."""

    class _B(WorldBackend):
        def reset(self):
            self.f = 0
            return self._obs()

        def _obs(self):
            xyzs = leads_per_frame[self.f] if self.f < len(leads_per_frame) else []
            dets = make_detections(xyzs=xyzs, frame=self.f)
            ego = make_ego(xyz=(0.0, 0.0, 0.0), yaw=0.0, speed=5.0, t=self.f * 0.1)
            return Observation(t=self.f * 0.1, frame=self.f, ego_speed=5.0,
                               sensor_data={"lidar": dets}, vehicle_state=ego)

        def step(self, control):
            self.f += 1
            return self._obs()

    return _B()


def test_component_log_captures_real_pipeline_outputs(make_detections, make_ego):
    stack = ModularAVStack(_CFG)
    stack.instrument()
    leads = [[(20.0 - 1.5 * f, 0.0, 0.0)] for f in range(8)]     # one real lead straight ahead, approaching
    trace, comp = run_logged(_backend(make_detections, make_ego, leads), stack, frames=8)

    assert len(trace.records) == 8
    assert comp.stage_names == ["perception", "tracking", "planning", "control"]
    assert comp.counts("perception") == [1] * 8                 # the detector passes the real lead through
    # the REAL tracker confirms a track only after a couple of frames (not an instant stub)
    assert comp.counts("tracking")[0] == 0 and comp.counts("tracking")[-1] == 1
    # the captured stage objects are the genuine avstack outputs
    plan = comp.steps[-1].stages["planning"]
    assert plan.top()[1].target_speed == 0.0                    # planner brakes for the lead
    assert comp.steps[-1].stages["control"].brake > 0.0         # and the PID actually brakes


def test_real_phantom_attack_propagates_perception_tracking_control(make_detections, make_ego):
    from avstack.config import HOOKS

    empty = [[] for _ in range(10)]                             # no real object anywhere
    clean = ModularAVStack(_CFG)
    clean.instrument()
    _, cc = run_logged(_backend(make_detections, make_ego, empty), clean, frames=10)

    attacked = ModularAVStack(_CFG)
    attacked.pipeline.perception.register_post_hook(HOOKS.build(dict(type="PhantomInjection")))  # real attack
    attacked.instrument()                                       # capture after the attack -> sees the phantom
    _, ca = run_logged(_backend(make_detections, make_ego, empty), attacked, frames=10)

    assert cc.counts("perception") == [0] * 10 and cc.counts("tracking") == [0] * 10  # clean: nothing
    # in-system propagation of the REAL attack: +1 detection/frame -> a confirmed track -> braking
    assert ca.degradation(cc, "perception") == [1] * 10
    assert ca.counts("tracking")[-1] == 1
    assert ca.steps[-1].stages["control"].brake > 0.0 and cc.steps[-1].stages["control"].brake == 0.0
