"""Real CPU pipeline outputs retain hook order, history and canonical detections."""

import numpy as np
import pytest

pytest.importorskip("avstack.config", reason="Native ModularAVStack integration requires avstack")

from avsectester.attack.implemented_attacks.phantom_injection_attack.attack import PhantomInjectionAttack
from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.avstacks.modular import ModularAVStack
from avsectester.closed_loop.datatypes import Control, Observation, WorldSnapshot
from avsectester.closed_loop.runner import ClosedLoopRunner, RunRequest
from avsectester.closed_loop.simulator import Simulator, SimulatorRuntime
from avsectester.datasets import ActorState


def _runtime(make_detections, make_ego, leads_per_frame, *, advance=None):
    """A native simulator double supplying actual avstack values without a CARLA server."""
    class World(Simulator):
        def create_runtime(self, seed=None):
            return WorldRuntime(self)

    class WorldRuntime(SimulatorRuntime):
        def _initialize_world(self, start, initialization, world):
            self.f = 0

        def _observe(self):
            from avstack.geometry import ReferenceFrame

            xyzs = leads_per_frame[self.f] if self.f < len(leads_per_frame) else []
            ego = make_ego(speed=5, t=self.f * .05)
            # Native captures declare the sensor installation relative to the true body,
            # even when its mounting transform is the identity.
            sensor_reference = ReferenceFrame(np.zeros(3), np.quaternion(1), ego.as_reference())
            return Observation(
                t=self.f * .05, frame=self.f, ego_speed=5.,
                sensor_data={"lidar": make_detections(xyzs=xyzs, frame=self.f,
                                                     reference=sensor_reference)},
                vehicle_state=ego,
            )

        def _advance_native(self, control):
            if advance is not None:
                advance()
            self.f += 1

        def _state(self):
            return WorldSnapshot(self.f * .05, self.f, ego_speed=5.,
                                 ego=ActorState(np.eye(4), (5., 0., 0.), "vehicle", (4., 2., 1.5)))

        def close(self):
            self._capture = None
            self._world_initialized = False

    return World().create_runtime()


def _run(runtime, avstack, decisions, *, attacks=()):
    try:
        return ClosedLoopRunner().run(runtime, avstack, RunRequest(decisions, attacks=attacks))
    finally:
        runtime.close()


def test_component_log_captures_phantom_after_attack_effects(pipeline_config, make_detections, make_ego):
    empty = [[] for _ in range(11)]
    clean, attacked = ModularAVStack(pipeline_config), ModularAVStack(pipeline_config)
    clean.instrument()
    attacked.instrument()
    clean_log = _run(_runtime(make_detections, make_ego, empty), clean, 10)
    attack = PhantomInjectionAttack(decision_steps=10).create_runtime(None, None)
    attacked_log = _run(_runtime(make_detections, make_ego, empty), attacked, 10, attacks=(attack,))
    assert [len(row.components["perception"]) for row in clean_log.steps] == [0] * 10
    assert [len(row.components["tracking"]) for row in clean_log.steps] == [0] * 10
    assert [len(row.components["perception"]) for row in attacked_log.steps] == [1] * 10
    assert len(attacked_log.steps[-1].components["tracking"]) == 1
    assert attacked_log.steps[-1].components["control"].brake > 0
    assert clean_log.steps[-1].components["control"].brake == 0
    assert [len(row.detections) for row in attacked_log.steps] == [1] * 10


def test_component_snapshots_preserve_pipeline_history_and_driving(pipeline_config, make_detections, make_ego):
    avstack = ModularAVStack(pipeline_config)
    assert avstack.component_log() is None
    avstack.instrument()
    leads = [[(20. - 1.5 * frame, 0., 0.)] for frame in range(9)]
    captured = _run(_runtime(make_detections, make_ego, leads), avstack, 8)
    plain = _run(_runtime(make_detections, make_ego, leads), ModularAVStack(pipeline_config), 8)
    assert [row.executed_command for row in captured.steps] == [row.executed_command for row in plain.steps]
    assert [len(row.detections) for row in captured.steps] == [1] * 8
    assert [len(row.detections) for row in plain.steps] == [1] * 8
    assert list(captured.steps[0].components) == ["perception", "tracking", "planning", "control"]
    assert len(captured.steps[0].components["tracking"]) == 0
    assert len(captured.steps[-1].components["tracking"]) == 1
    speeds = [row.components["planning"].top()[1].target_speed for row in captured.steps]
    assert speeds == [6.] * 6 + [0.] * 2
    assert captured.steps[0].executed_command.brake == 0
    assert captured.steps[-1].executed_command.brake > 0
    positions = [row.components["tracking"][0].position.x[0]
                 for row in captured.steps if len(row.components["tracking"])]
    assert len(positions) >= 2 and np.all(np.diff(positions) < 0)
    avstack(Observation(t=.4, frame=8, sensor_data={"lidar": make_detections([], frame=8)},
                      vehicle_state=make_ego(t=.4)))
    assert [row.components["planning"].top()[1].target_speed for row in captured.steps] == speeds
    assert [row.components["tracking"][0].position.x[0]
            for row in captured.steps if len(row.components["tracking"])] == positions


def test_snapshot_detaches_mutable_sensor_reference_chain(make_detections, make_ego):
    from avstack.geometry import GlobalOrigin3D, ReferenceFrame

    ego_frame = ReferenceFrame(np.array([10., 0., 0.]), np.quaternion(1), GlobalOrigin3D)
    sensor_frame = ReferenceFrame(np.array([1., 0., 0.]), np.quaternion(1), ego_frame)
    detections = make_detections([(5., 0., 0.)], reference=sensor_frame)

    class Driver(AVStack):
        def __call__(self, observation):
            return Control()

        def component_log(self):
            return {"perception": detections}

    log = _run(_runtime(make_detections, make_ego, [[]],
                        advance=lambda: setattr(ego_frame, "x", np.array([100., 0., 0.]))), Driver(), 1)
    saved = log.steps[0].components["perception"][0]
    np.testing.assert_allclose(saved.position.change_reference(GlobalOrigin3D, inplace=False).x, [16, 0, 0])
    np.testing.assert_allclose(detections[0].position.change_reference(GlobalOrigin3D, inplace=False).x, [106, 0, 0])


@pytest.mark.parametrize("yaw", [0., np.pi / 2])
def test_detections_use_true_capture_ego_frame_and_detection_box_center(pipeline_config, make_detections, make_ego, yaw):
    from avstack.geometry import ReferenceFrame

    ego = make_ego(xyz=(20., 10., 0.), yaw=yaw)
    sensor = ReferenceFrame(np.array([1., 2., 1.]), np.quaternion(1), ego.as_reference())
    inputs = make_detections([(5., 0., 0.)], reference=sensor)
    avstack = ModularAVStack(pipeline_config)
    calls = []
    def count(output):
        calls.append(output)
        return (output,)
    avstack.pipeline.perception.register_post_hook(count)
    observation = Observation(0., 0, sensor_data={"lidar": inputs}, vehicle_state=ego)
    avstack(observation)
    captured = avstack.detections()
    assert len(calls) == 1  # reading canonical boxes never executes perception again
    assert len(captured) == 1
    np.testing.assert_allclose(captured[0].position, [6., 2., 1.75], atol=1e-6)
    assert captured[0].extent == (4., 1.8, 1.5)
    assert captured[0].yaw == pytest.approx(0., abs=1e-6)
    assert captured[0].category == "Car" and captured[0].score == pytest.approx(.8)
    sensor.x = np.array([90., 0., 0.])
    assert avstack.detections() == captured
    assert len(calls) == 1


def test_detection_capture_observes_native_hooks_added_after_construction(pipeline_config, make_detections, make_ego):
    avstack = ModularAVStack(pipeline_config)
    def remove(output):
        output.data.clear()
        return (output,)
    avstack.pipeline.perception.register_post_hook(remove)
    avstack(Observation(0., 0, sensor_data={"lidar": make_detections([(8, 0, 0)])},
                         vehicle_state=make_ego()))
    assert avstack.detections() == ()
    assert avstack.component_log() is None


def test_recorded_detections_use_sensor_mount_and_ignore_model_localization(
    pipeline_config, make_detections, make_ego,
):
    from avstack.geometry import ReferenceFrame

    true_ego = make_ego(xyz=(20., 10., 0.), yaw=np.pi / 3)
    false_ego = make_ego(xyz=(300., -400., 40.), yaw=-np.pi / 2)
    sensor = ReferenceFrame(np.array([1., 2., 1.]), np.quaternion(1), true_ego.as_reference())
    inputs = make_detections([(5., 0., 0.)], reference=sensor)
    avstack = ModularAVStack(pipeline_config)
    # The driving pipeline still receives the false localization, while recorder-only boxes
    # use the true body parent of the installed sensor.
    avstack(Observation(0, 0, {'lidar': inputs}, vehicle_state=false_ego))
    captured = avstack.detections()
    np.testing.assert_allclose(captured[0].position, [6., 2., 1.75], atol=1e-6)
    assert captured[0].yaw == pytest.approx(0., abs=1e-6)
