"""Capture real avstack outputs while a synthetic backend supplies detections and ego poses.

Covers snapshot isolation, hook ordering and matching metrics with CPU tracking, planning and control.
This suite requires avstack and fails if it is missing.
"""

import pytest

from avsectester.backend import WorldBackend, run
from avsectester.evaluation.component_log import run_logged
from avsectester.plane import Observation
from avsectester.stacks.modular import ModularAVStack


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
            ego = make_ego(xyz=(0.0, 0.0, 0.0), yaw=0.0, speed=5.0, t=self.f * 0.05)
            return Observation(
                t=self.f * 0.05,
                frame=self.f,
                ego_speed=5.0,
                sensor_data={"lidar": dets},
                vehicle_state=ego,
            )

        def step(self, control):
            self.f += 1
            return self._obs()

    return _B()


def test_component_log_captures_phantom_after_attack_hooks(
    pipeline_config, make_detections, make_ego
):
    from avstack.config import HOOKS

    empty = [[] for _ in range(10)]  # no real object anywhere
    clean = ModularAVStack(pipeline_config)
    clean.instrument()
    _, cc = run_logged(_backend(make_detections, make_ego, empty), clean, frames=10)

    attacked = ModularAVStack(pipeline_config)
    attacked.pipeline.perception.register_post_hook(
        HOOKS.build(dict(type="PhantomInjection"))
    )  # real attack
    attacked.instrument()  # capture after the attack -> sees the phantom
    _, ca = run_logged(_backend(make_detections, make_ego, empty), attacked, frames=10)

    assert (
        cc.counts("perception") == [0] * 10 and cc.counts("tracking") == [0] * 10
    )  # clean: nothing
    # in-system propagation of the REAL attack: +1 detection/frame -> a confirmed track -> braking
    assert ca.degradation(cc, "perception") == [1] * 10
    assert ca.counts("tracking")[-1] == 1
    assert (
        ca.steps[-1].stages["control"].brake > 0.0 and cc.steps[-1].stages["control"].brake == 0.0
    )


def test_component_snapshots_preserve_pipeline_history_and_driving(
    pipeline_config, make_detections, make_ego
):
    import numpy as np

    stack = ModularAVStack(pipeline_config)
    assert stack.component_log() is None
    stack.instrument()
    leads = [[(20.0 - 1.5 * f, 0.0, 0.0)] for f in range(8)]
    trace, comp = run_logged(_backend(make_detections, make_ego, leads), stack, frames=8)
    plain_trace = run(
        _backend(make_detections, make_ego, leads), ModularAVStack(pipeline_config), frames=8
    )
    assert trace.records == plain_trace.records
    assert comp.stage_names == ["perception", "tracking", "planning", "control"]
    assert comp.counts("perception") == [1] * 8
    assert comp.counts("tracking")[0] == 0
    assert comp.counts("tracking")[-1] == 1
    speeds = [step.stages["planning"].top()[1].target_speed for step in comp.steps]
    assert speeds == [6.0] * 6 + [0.0] * 2
    assert trace.records[0].brake == 0.0 and trace.records[-1].brake > 0.0
    positions = [
        step.stages["tracking"][0].position.x[0]
        for step in comp.steps
        if len(step.stages["tracking"])
    ]
    assert len(positions) >= 2 and np.all(np.diff(positions) < 0)
    # Further use of the same model must not rewrite the completed trace either.
    stack(
        Observation(
            t=0.4,
            frame=8,
            sensor_data={"lidar": make_detections([], frame=8)},
            vehicle_state=make_ego(t=0.4),
        )
    )
    assert [step.stages["planning"].top()[1].target_speed for step in comp.steps] == speeds
    assert [
        step.stages["tracking"][0].position.x[0]
        for step in comp.steps
        if len(step.stages["tracking"])
    ] == positions


def test_snapshot_detaches_mutable_sensor_reference_chain(make_detections):
    import numpy as np
    from avstack.geometry import GlobalOrigin3D, ReferenceFrame
    from avsectester.backend import AVStack
    from avsectester.plane import Control

    ego_frame = ReferenceFrame(np.array([10.0, 0.0, 0.0]), np.quaternion(1), GlobalOrigin3D)
    sensor_frame = ReferenceFrame(np.array([1.0, 0.0, 0.0]), np.quaternion(1), ego_frame)
    detections = make_detections([(5.0, 0.0, 0.0)], reference=sensor_frame)

    class Backend(WorldBackend):
        def reset(self):
            return Observation(t=0, frame=0)

        def step(self, control):
            ego_frame.x = np.array([100.0, 0.0, 0.0])
            return Observation(t=1, frame=1)

    class Stack(AVStack):
        def __call__(self, obs):
            return Control()

        def component_log(self):
            return {"perception": detections}

    _, comp = run_logged(Backend(), Stack(), frames=1)
    saved = comp.steps[0].stages["perception"][0]
    np.testing.assert_allclose(
        saved.position.change_reference(GlobalOrigin3D, inplace=False).x, [16.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        detections[0].position.change_reference(GlobalOrigin3D, inplace=False).x, [106.0, 0.0, 0.0]
    )


@pytest.mark.parametrize("stage", ["perception", "tracking"])
@pytest.mark.parametrize("radius,expected_matches", [(4.0, 2), (2.0, 1)])
def test_component_performance_matches_objects_with_configured_radius(
    make_detections, make_ego, stage, radius, expected_matches
):
    from avsectester.evaluation.component_log import ComponentTrace, StepLog

    # One exact match, one 3 m offset, one false positive and one missed truth.
    detections = make_detections([(0, 0, 0), (13, 0, 0), (100, 0, 0)])
    objects = (
        detections if stage == "perception" else [make_ego(xyz=(x, 0, 0)) for x in (0, 13, 100)]
    )
    truths = [make_ego(xyz=(x, 0, 0)) for x in (0, 10, 50)]
    comp = ComponentTrace([StepLog(0, {stage: objects})])
    (metrics,) = comp.performance(stage, [truths], assign_radius=radius)
    assert metrics.n_tracks == metrics.n_truths == 3
    assert metrics.n_assign == expected_matches
    assert metrics.precision == pytest.approx(expected_matches / 3)
    assert metrics.recall == pytest.approx(expected_matches / 3)
