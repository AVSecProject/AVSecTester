"""Driving decisions, actions and failures are observed through ClosedLoopRunner.run."""

from dataclasses import replace

import numpy as np
import pytest

from avsectester.attack.runtime import AttackRuntime, AVStackAttackAction, SensorAttackAction, WorldAttackAction
from avsectester.attack.insertion import InsertObject
from avsectester.data_types import Control, Detection
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.closed_loop.simulators.nurec import NuRecSimulator
from avsectester.execution.runner import ClosedLoopRunner, RecordingPolicy, RunRequest
from avsectester.execution.services import ClosedLoopServices, SensorRequest
from avsectester.runlog import RunLog
from avsectester.rendering import ObjectVisibility
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
        assert record.physical_pre.ego_speed == record.raw_observation.ego_speed


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
    assert np.all(log.steps[0].raw_observation.sensor_data["front"] == 23)
    assert np.any(log.steps[0].model_observation.sensor_data["front"] != 23)
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


def test_insert_object(tmp_path, make_object_in_front):
    with MemorySimulator().create_runtime() as simulator_runtime:
        simulator_runtime.reset()
        clean = simulator_runtime.sensor_view("front")
        asset, world_from_object = make_object_in_front(clean)

        class InsertAttackRuntime(AttackRuntime):
            def setup(self, context):
                return (InsertObject("sign", asset, world_from_object),)

        log = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(), RunRequest(
            2, attacks=(InsertAttackRuntime(),), recording=RECORDING, output_path=tmp_path / "run"))
        assert log.status == "completed"
        record = log.steps[0]
        assert np.any(record.raw_observation.sensor_data["front"] != clean.data)
        assert record.physical_pre.objects.keys() == record.physical_post.objects.keys() == {"sign"}
        assert simulator_runtime.state().objects == {}
        receipts = [item for event in log.events for item in event.data.get("actions", ())
                    if item["kind"] == "receipt"]
        insertion = next(item for item in receipts if item.get("details", {}).get("object_id") == "sign")
        assert insertion["outcome"] == "applied"
        assert set(insertion["details"]) == {"object_id", "asset_category", "world_from_object"}
        assert insertion["details"]["asset_category"] == "stop_sign"
        np.testing.assert_allclose(insertion["details"]["world_from_object"], world_from_object)
        initialization = next(event for event in log.events if event.kind == "initialization")
        assert initialization.data["object_visibility"] == record.object_visibility
        for step in log.steps:
            assert set(step.object_visibility) == {"sign"}
            assert set(step.object_visibility["sign"]) == {"front"}
            assert isinstance(step.object_visibility["sign"]["front"], ObjectVisibility)
            assert step.object_visibility["sign"]["front"].visible_pixels > 0
        saved = RunLog.read(tmp_path / "run")
        assert [step.object_visibility for step in saved.steps] == [
            step.object_visibility for step in log.steps]

        class RemoveObject(WorldAttackAction):
            def apply(self, simulator_runtime, resources):
                simulator_runtime.remove_object("sign")

        class InsertThenRemoveAttackRuntime(InsertAttackRuntime):
            def step(self, context):
                return (RemoveObject(),)

        removed = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(), RunRequest(
            1, attacks=(InsertThenRemoveAttackRuntime(),)))
        assert removed.status == "completed"
        assert removed.steps[0].physical_pre.objects.keys() == {"sign"}
        assert removed.steps[0].physical_post.objects == simulator_runtime.state().objects == {}

    restored_log = RunLog.read(tmp_path / "run")
    restored_insertion = next(item for event in restored_log.events
                              for item in event.data.get("actions", ())
                              if item["kind"] == "receipt"
                              and item.get("details", {}).get("object_id") == "sign")
    assert set(restored_insertion["details"]) == set(insertion["details"])
    assert restored_insertion["details"]["asset_category"] == "stop_sign"
    for actual, expected in zip(restored_log.steps, log.steps):
        assert actual.object_visibility == expected.object_visibility
        assert isinstance(actual.object_visibility["sign"]["front"], ObjectVisibility)
    restored_initialization = next(event for event in restored_log.events if event.kind == "initialization")
    assert restored_initialization.data["object_visibility"] == initialization.data["object_visibility"]
    restored = restored_log.steps[0]
    for state in (restored.physical_pre.objects["sign"], restored.physical_post.objects["sign"]):
        np.testing.assert_allclose(state.pose, world_from_object)
        assert state.extent == asset.extent and state.category == asset.category
        assert state.box_center == (0., 0., 1.)


def test_run_rejects_invalid_request():
    with pytest.raises(ValueError, match="nonnegative"):
        RunRequest(-1)
    with MemorySimulator().create_runtime() as simulator_runtime:
        with pytest.raises(ValueError, match="unavailable sensors"):
            ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(), RunRequest(
                1, recording=RecordingPolicy(("missing",))))

        class InvalidSensorAttack(AttackRuntime):
            def step(self, context):
                return (AVStackAttackAction("sensors.post", lambda value, context: value),)

        with pytest.raises(ValueError, match="use SensorAttackAction to modify sensor data") as failure:
            ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(),
                                   RunRequest(1, attacks=(InvalidSensorAttack(),)))
        assert failure.value.run_log.status == "failed"
    unsupported = MemorySimulator()
    unsupported.can_observe_past = False
    with unsupported.create_runtime() as runtime:
        with pytest.raises(NotImplementedError, match="cannot provide the 2 recorded past observations"):
            ClosedLoopRunner().run(runtime, HistoryAVStack(), RunRequest(1))


def test_run_with_decelerating_feedback(nurec_longitudinal_recording, deceleration_case):
    dynamics, expected_drop, tolerance = deceleration_case
    dataset = nurec_longitudinal_recording
    start = dataset.scenario_start("scene", dataset.frames("scene")[15])
    camera, _ = dataset.camera_geometry("scene")
    simulator = NuRecSimulator(
        NuRecImageBoundary(camera), dynamics,
        depth="none", route_generator="none", physics_update_mode="none",
    time_step=.1)
    with simulator.create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, DeceleratingAVStack(), RunRequest(10, start=start))
    assert log.status == "completed" and len(log.steps) == 10
    speeds = np.array([log.steps[0].physical_pre.ego_speed,
                       *(step.physical_post.ego_speed for step in log.steps)])
    assert speeds[0] == pytest.approx(10.)
    assert speeds[0] - speeds[-1] == pytest.approx(expected_drop, abs=tolerance)
    np.testing.assert_allclose(speeds, 10 - expected_drop * np.arange(11) / 10,
                               atol=tolerance, rtol=0)
    np.testing.assert_allclose([step.model_observation.ego_speed for step in log.steps],
                               speeds[:-1], atol=1e-12)
