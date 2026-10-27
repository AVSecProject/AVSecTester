"""Driving decisions, actions and failures are observed through ClosedLoopRunner.run."""

from dataclasses import replace

import numpy as np
import pytest

from avsectester.attack.runtime import AttackRuntime, AVStackAttackAction, SensorAttackAction, WorldAttackAction
from avsectester.attack.insertion import InsertObject
from avsectester.data_types import Control, Detection, Observation
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.closed_loop.simulators.nurec import NuRecSimulator
from avsectester.execution.runner import ClosedLoopRunner, RecordingPolicy, RunRequest
from avsectester.execution.services import ClosedLoopServices, SensorRequest
from avsectester.runlog import Recorder, RunLog
from tests.fakes import (
    ConstantAVStack, DeceleratingAVStack, HistoryAVStack, MemoryDataset, MemorySimulator,
    NuRecImageBoundary,
)


RECORDING = RecordingPolicy(("front",), ("raw", "model_input"))


def test_run():
    data = MemoryDataset()
    start = data.scenario_start("scene", data.frames("scene")[2])
    with MemorySimulator().create_runtime() as simulator_runtime:
        log = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(Control(throttle=.3)),
                                     RunRequest(2, start=start, recording=RECORDING))
        assert log.status == "completed"
        assert len(log.steps) == 2
        for index, record in enumerate(log.steps):
            assert record.raw_observation.frame == index + 2
            assert record.model_observation.frame == record.physical_pre.frame
            np.testing.assert_array_equal(record.raw_observation.sensor_data["front"],
                                          record.model_observation.sensor_data["front"])
            assert record.produced_command == record.executed_command == Control(throttle=.3)
            assert record.physical_post.frame == record.physical_pre.frame + 1
            assert record.physical_post.t == pytest.approx(record.physical_pre.t + .1)
        empty = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(), RunRequest(0))
        assert empty.status == "completed" and not empty.steps

    with MemorySimulator().create_runtime() as simulator_runtime:
        with pytest.raises(ValueError, match="unavailable sensors"):
            ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(), RunRequest(
                1, recording=RecordingPolicy(("missing",))))

    unsupported = MemorySimulator(can_observe_past=False)
    with unsupported.create_runtime() as runtime:
        with pytest.raises(NotImplementedError, match="cannot provide the 2 recorded past observations"):
            ClosedLoopRunner().run(runtime, HistoryAVStack(), RunRequest(1))


def test_run_with_hooks():
    hooks = HookSet([
        Hook("sensors.post", lambda value, context: replace(value, ego_speed=value.ego_speed + 1.)),
        Hook("observation", lambda value, context: replace(value, ego_speed=value.ego_speed + 5.)),
        Hook("command", lambda value, context: replace(value, brake=.7)),
    ])
    with MemorySimulator().create_runtime() as simulator_runtime:
        log = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(Control(throttle=.2)),
                                     RunRequest(2, hooks=hooks, recording=RECORDING))
    for record in log.steps:
        assert record.model_observation.ego_speed == record.raw_observation.ego_speed + 6.
        assert record.produced_command == Control(throttle=.2)
        assert record.executed_command == Control(throttle=.2, brake=.7)


def test_run_with_past(tmp_path):
    class Paint(SensorAttackAction):
        sensor = "front"

        def apply(self, data, view):
            data[:] = 100
            return data

    class PaintAttackRuntime(AttackRuntime):
        def step(self, context):
            return (Paint(),)

    dataset, simulator = MemoryDataset(), MemorySimulator()
    frames = dataset.frames("scene")
    start = dataset.scenario_start("scene", frames[3])
    environment = {"sensors": [{"type": "GaussianNoise", "camera": "front",
                                 "options": {"severity": .2}}]}
    hooks = HookSet([Hook("sensors.post", lambda value, context:
                         replace(value, ego_speed=value.ego_speed + 5.))], seed=7)
    avstack = HistoryAVStack()
    with simulator.create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, avstack, RunRequest(
            1, start=start, environment=environment, hooks=hooks,
            attacks=(PaintAttackRuntime(),), recording=RECORDING, output_path=tmp_path / "run"))
    past = avstack.past_observations
    assert [observation.t for observation in past] == pytest.approx([-.2, -.1])
    assert [observation.ego_speed for observation in past] == [1., 1.]
    assert avstack.initial_observation.ego_speed == 6.
    raw = log.steps[0].raw_observation.sensor_data["front"]
    assert np.all(raw == 23)
    assert np.any(log.steps[0].model_observation.sensor_data["front"] !=
                  log.steps[0].raw_observation.sensor_data["front"])
    with ClosedLoopServices(dataset, simulator) as services, services.session(7) as session:
        for historical, frame in zip(past, frames[1:3]):
            original = session.read_sensors(SensorRequest("scene", frame, ("front",)))
            conditioned = session.read_sensors(SensorRequest(
                "scene", frame, ("front",), environment=environment))
            np.testing.assert_array_equal(historical.sensor_data["front"],
                                          conditioned.observation.sensor_data["front"])
            assert np.any(historical.sensor_data["front"] != original.observation.sensor_data["front"])
    saved = RunLog.read(tmp_path / "run")
    initialization = next(event for event in saved.events if event.kind == "initialization")
    assert initialization.data["past_steps"] == 2
    assert [item["t"] for item in initialization.data["past_observations"]] == pytest.approx([-.2, -.1])
    assert all(set(item) == {"t", "vehicle_state"}
               for item in initialization.data["past_observations"])
    with simulator.create_runtime() as runtime:
        empty = ClosedLoopRunner().run(runtime, avstack, RunRequest(
            0, start=start, environment=environment, hooks=hooks))
        assert empty.status == "completed" and not empty.steps
        assert len(avstack.past_observations) == 2
        assert avstack.initial_observation.ego_speed == 6.


def test_run_with_attacks(tmp_path):
    class WrongStage(AttackRuntime):
        def step(self, context):
            return (AVStackAttackAction("sensors.post", lambda value, context: value),)

    with MemorySimulator().create_runtime() as runtime:
        recorder = Recorder()
        with pytest.raises(ValueError, match="use SensorAttackAction to modify sensor data"):
            ClosedLoopRunner().run(
                runtime, ConstantAVStack(), RunRequest(1, attacks=(WrongStage(),)), recorder=recorder)
        assert recorder.log.status == "failed"

    lifecycle = []
    resource = (tmp_path / "world-resource.txt").open("w")

    class MoveHost(WorldAttackAction):
        def __init__(self, x, cleanup=False):
            self.x, self.cleanup = x, cleanup

        def apply(self, simulator_runtime, resources):
            actor = simulator_runtime.state().actors["lead"]
            pose = actor.pose.copy()
            pose[0, 3] = self.x
            simulator_runtime.set_actor_state("lead", replace(actor, pose=pose))
            if self.cleanup:
                resources.callback(resource.close)

    class Paint(SensorAttackAction):
        sensor = "front"

        def apply(self, data, view):
            data[:] = 42
            return data.astype(float) + .5

    class ShiftPoints(SensorAttackAction):
        sensor = "lidar"

        def apply(self, data, view):
            data[..., 0] += 2.
            return data

    class ScalePoints(SensorAttackAction):
        sensor = "lidar"

        def apply(self, data, view):
            data[..., 0] *= 3.
            return data

    class OnlineAttack(AttackRuntime):
        def setup(self, context):
            lifecycle.append("setup")
            return (MoveHost(12., cleanup=True),)

        def step(self, context):
            context.record_output("decision", context.step)
            return (Paint(), ShiftPoints(), ScalePoints(), MoveHost(14.),
                    AVStackAttackAction("perception.post", self.inject))

        def inject(self, value, context):
            return [*value, Detection((6., 0., 0.), (4., 2., 1.5), 0., "car", .9)]

        def stop(self, context):
            lifecycle.append("stop")

    points = np.tile(np.array([5.25, -2., .5, .8], np.float32), (2, 3, 1))
    with MemorySimulator(extra_sensors={"lidar": points}).create_runtime() as simulator_runtime:
        log = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(perception_output=[]),
                                     RunRequest(1, attacks=(OnlineAttack(),), recording=RecordingPolicy(
                                         ("front", "lidar"), ("raw", "model_input"))))
    assert lifecycle == ["setup", "stop"]
    assert resource.closed
    record = log.steps[0]
    assert record.physical_pre.actors["lead"].pose[0, 3] == 12.
    assert record.physical_post.actors["lead"].pose[0, 3] == 14.
    assert np.all(record.model_observation.sensor_data["front"] == 42)
    assert record.model_observation.sensor_data["front"].dtype == np.uint8
    assert np.any(record.raw_observation.sensor_data["front"] != 42)
    np.testing.assert_array_equal(record.raw_observation.sensor_data["lidar"], points)
    shifted = points.copy()
    shifted[..., 0] = 21.75
    np.testing.assert_array_equal(record.model_observation.sensor_data["lidar"], shifted)
    assert record.model_observation.sensor_data["lidar"].dtype == points.dtype
    assert len(record.detections) == 1
    assert record.outputs[0].name == "decision" and record.outputs[0].value == 0
    receipts = [action for event in log.events for action in event.data.get("actions", [])
                if action["kind"] == "receipt"]
    receipts += [action for action in record.actions if action["kind"] == "receipt"]
    assert {action["category"] for action in receipts} == {"world", "sensor", "avstack"}
    assert all(action["outcome"] == "applied" for action in receipts)
    assert all(isinstance(action["details"], dict) for action in receipts)


def test_run_failure(tmp_path):
    stopped = []
    resource = (tmp_path / "run-resource.txt").open("w")

    class ActiveAttack(AttackRuntime):
        def stop(self, context):
            stopped.append(context.stage)

    with MemorySimulator().create_runtime() as simulator_runtime:
        with pytest.raises(RuntimeError) as failure:
            ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(fail_on_position=1),
                                   RunRequest(3, attacks=(ActiveAttack(),)),
                                   release_owned_resources=resource.close)
        log = failure.value.run_log
        assert log.status == "failed"
        assert len(log.steps) == 1
        assert any(event.kind == "failure" for event in log.events)
        assert stopped == ["stop"] and resource.closed
        # A failed run relinquishes the simulator reservation and handler binding.
        again = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(), RunRequest(1))
        assert again.status == "completed"
        with pytest.raises(KeyboardInterrupt) as cancelled:
            ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(interrupt_on_position=0), RunRequest(1))
        assert cancelled.value.run_log.status == "cancelled"


def test_run_with_inserted_object(make_object_in_front):
    with MemorySimulator().create_runtime() as runtime:
        runtime.reset()
        clean = runtime.sensor_view("front")
        asset, pose = make_object_in_front(clean)

        class InsertAttackRuntime(AttackRuntime):
            def setup(self, context):
                return (InsertObject("sign", asset, pose),)

        log = ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(
            1, attacks=(InsertAttackRuntime(),), recording=RECORDING))
    step = log.steps[0]
    assert np.any(step.raw_observation.sensor_data["front"] != clean.data)
    initialization = next(event for event in log.events if event.kind == "initialization")
    receipts = [item for item in initialization.data["actions"] if item["kind"] == "receipt"]
    assert any(item["outcome"] == "applied" and item["details"]["object_id"] == "sign"
               for item in receipts)
    assert step.object_visibility["sign"]["front"].visible_pixels > 0


def test_run_with_decelerating_feedback(nurec_longitudinal_dataset):
    from avsectester.closed_loop.simulators.ego_dynamics import TrajectoryFollower

    dataset = nurec_longitudinal_dataset
    start = dataset.scenario_start("scene", dataset.frames("scene")[15])
    camera, _ = dataset.camera_geometry("scene")
    simulator = NuRecSimulator(
        NuRecImageBoundary(camera), TrajectoryFollower(), time_step=.1,
        depth="none", route_generator="none", physics_update_mode="none")
    with simulator.create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, DeceleratingAVStack(), RunRequest(10, start=start))
    assert len(log.steps) == 10
    assert np.all(np.diff([step.physical_pre.ego_speed for step in log.steps]) < 0)
    np.testing.assert_allclose([step.model_observation.ego_speed for step in log.steps],
                               [step.physical_pre.ego_speed for step in log.steps], atol=1e-12)


def test_recording_policy():
    pixels = np.zeros((2, 3, 3), np.uint8)
    observation = Observation(0., 0, sensor_data={"front": pixels, "other": pixels.copy()},
                              calibration={"front": "calibration"})
    policy = RecordingPolicy(("front",), ("raw",), every_n_steps=2)
    captured = policy.observation(observation, 0, "raw")
    assert list(captured.sensor_data) == ["front"]
    assert captured.calibration == observation.calibration
    assert policy.observation(observation, 1, "raw").sensor_data == {}
    assert policy.observation(observation, 2, "model_input").sensor_data == {}
    assert list(observation.sensor_data) == ["front", "other"]
    with pytest.raises(ValueError, match="every_n_steps"):
        RecordingPolicy(every_n_steps=0)
    with pytest.raises(ValueError, match="representations"):
        RecordingPolicy(representations=("unknown",))


def test_run_request(tmp_path):
    request = RunRequest(2, retain_steps=False, output_path=tmp_path / "streamed")
    with MemorySimulator().create_runtime() as runtime:
        streamed = ClosedLoopRunner().run(runtime, ConstantAVStack(), request)
    assert streamed.status == "completed" and not streamed.steps
    assert len(RunLog.read(request.output_path).steps) == 2
    with pytest.raises(ValueError, match="requires an output_path"):
        RunRequest(1, retain_steps=False)
    with pytest.raises(ValueError, match="nonnegative"):
        RunRequest(-1)
