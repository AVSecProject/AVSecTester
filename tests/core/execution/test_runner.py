"""Driving decisions, actions and failures are observed through ClosedLoopRunner.run."""

import pickle
from dataclasses import dataclass, replace

import numpy as np
import pytest

from avsectester.attack.runtime import (
    AttackRuntime, AVStackAttackAction, ResultSubscription, SensorAttackAction, WorldAttackAction,
)
from avsectester.attack.insertion import InsertObject
from avsectester.data_types import Control, Detection, Observation, StateEstimate
from avsectester.closed_loop.avstacks.base import ObservationPipeline
from avsectester.closed_loop.avstacks.implemented.external_control import ExternalAVStack
from avsectester.closed_loop.hooks import Hook, HookSet, Plugin
from avsectester.closed_loop.timing import HistoryRequirement, SensorTiming, TimingConfiguration
from avsectester.closed_loop.simulators.base import SensorGeometry
from avsectester.closed_loop.simulators.ego_dynamics import PlanarEgoState, TrajectoryFollower
from avsectester.closed_loop.simulators.nurec import NuRecSimulator
from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.datasets.carla import CarlaDataset
from avsectester.execution.runner import ClosedLoopRunner, RecordingPolicy, RunRequest
from avsectester.execution.services import ClosedLoopServices, SensorRequest
from avsectester.runlog import Recorder, RunLog
from tests.fakes import (
    ConstantAVStack, DeceleratingAVStack, ExternallyControlledMemorySimulator, HistoryAVStack,
    MemoryDataset, MemorySimulator,
    CAMERA, CAM_FROM_WORLD, CarlaNativeBoundary, NuRecImageBoundary,
)


RECORDING = RecordingPolicy(("front",), ("raw", "model_input"))


def test_externally_controlled_ego():
    with ExternallyControlledMemorySimulator().create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, ExternalAVStack(), RunRequest(
            2, timing=TimingConfiguration(planner_delay_us=100000)))
    assert log.status == "completed" and len(log.steps) == 2
    for step in log.steps:
        assert step.produced_command is None and step.executed_command is None
        assert step.executed_command_issue_time_us is None
        assert step.telemetry["ego_control"] == {"throttle": .2, "brake": 0., "steer": 0.}
        assert step.physical_post.ego.pose[0, 3] > step.physical_pre.ego.pose[0, 3]


def test_external_control_actions():
    applied = []
    captured = []

    class SetSpeed(WorldAttackAction):
        def apply(self, simulator_runtime, resources):
            simulator_runtime.control_actor("lead", Control(throttle=3.))

    class WorldOnly(AttackRuntime):
        def step(self, context):
            return (SetSpeed(),)

    class Paint(SensorAttackAction):
        sensor = "front"

        def apply(self, data, view):
            applied.append("sensor")
            data[:] = 37
            return data

    def change_command(value, context):
        applied.append("command")
        return Control(throttle=1.)

    class NonWorld(AttackRuntime):
        subscriptions = (ResultSubscription("observation"),)

        def __init__(self, action):
            self.action = action

        def step(self, context):
            captured.append(context.observation.sensor_data["front"].copy())
            return (self.action,)

    with ExternallyControlledMemorySimulator().create_runtime() as runtime:
        log = ClosedLoopRunner().run(
            runtime, ExternalAVStack(), RunRequest(1, attacks=(WorldOnly(),)))
        assert log.steps[0].physical_post.actors["lead"].velocity == (3., 0., 0.)
        for action in (Paint(), AVStackAttackAction("command", change_command)):
            with pytest.raises(ValueError, match="do not reach the external") as failure:
                ClosedLoopRunner().run(
                    runtime, ExternalAVStack(), RunRequest(1, attacks=(NonWorld(action),)))
            assert failure.value.run_log.status == "failed" and not failure.value.run_log.steps
            assert applied == []
            np.testing.assert_array_equal(runtime.observe().sensor_data["front"], captured[-1])


def test_external_input_checks_before_reset(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch, None)
    recipe = carla_dataset.native("scene")
    scenario = {**recipe.scenario, "ego": {**recipe.scenario["ego"],
        "blueprint_attributes": {"role_name": "ego"}}}
    dataset = CarlaDataset({"scene": replace(recipe, scenario=scenario)})
    start = dataset.scenario_start("scene", dataset.frames("scene")[0])
    called = []

    def modify_input(value, context):
        called.append(True)
        value.sensor_data.clear()
        return value

    requests = (
        RunRequest(1, start=start, hooks=HookSet([Hook("observation", modify_input)])),
        RunRequest(1, start=start, environment={"sensors": [{"type": "Brightness"}]}),
    )
    simulator = CarlaSimulator({"rate": 10}, camera_warmup_s=0.,
                               ego_control="ros2", realtime_factor=1000.)
    for request in requests:
        with simulator.create_runtime() as runtime:
            with pytest.raises(ValueError, match="do not reach the external"):
                ClosedLoopRunner().run(runtime, ExternalAVStack(), request)
            assert runtime.native["client"] is None
    assert called == []


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
            payload = record.model_observation.sensor_data["front"]
            np.testing.assert_array_equal(
                payload, log.captures[index].model_observation.sensor_data["front"])
            np.testing.assert_array_equal(
                payload, log.captures[index].raw_observation.sensor_data["front"])
            assert record.physical_pre.ego.pose.flags.writeable
            assert record.physical_post.frame == record.physical_pre.frame + 1
            assert record.physical_post.t == pytest.approx(record.physical_pre.t + .1)
        empty = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(), RunRequest(0))
        assert empty.status == "completed" and not empty.steps
        raw_only = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(), RunRequest(
            1, recording=RecordingPolicy(("front",), ("raw",))))
        assert "front" in raw_only.steps[0].sensor_geometry
        no_payloads = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(), RunRequest(
            1, recording=RecordingPolicy(("missing",), ())))
        assert no_payloads.status == "completed"
        assert no_payloads.steps[0].model_observation.sensor_data == {}

    cameras = {name: SensorGeometry(name, CAMERA, CAM_FROM_WORLD) for name in ("rear", "front")}
    with MemorySimulator(camera_geometries=cameras).create_runtime() as runtime:
        ordered = ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(
            1, recording=RecordingPolicy(tuple(cameras), ("raw", "model_input"))))
    record = ordered.steps[0]
    assert tuple(record.raw_observation.sensor_data) == tuple(cameras)
    assert tuple(record.model_observation.deliveries) == tuple(cameras)
    assert tuple(record.sensor_geometry) == tuple(cameras)

    unsupported = MemorySimulator(can_observe_past=False)
    with unsupported.create_runtime() as runtime:
        with pytest.raises(NotImplementedError,
                           match="cannot provide recorded acquisition history"):
            ClosedLoopRunner().run(runtime, HistoryAVStack(), RunRequest(1))


def test_zero_decision_run():
    lifecycle = []

    class SetupOnly(AttackRuntime):
        def setup(self, context):
            lifecycle.append("setup")
            return ()

        def step(self, context):
            raise AssertionError("No attack step belongs to a zero-decision run")

        def stop(self, context):
            lifecycle.append("stop")

    driver = HistoryAVStack()
    driver.history = HistoryRequirement()
    hooks = HookSet([Hook("observation", lambda value, context: replace(value, ego_speed=12.))])
    with MemorySimulator().create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, driver, RunRequest(
            0, attacks=(SetupOnly(),), hooks=hooks, recording=RECORDING))
    assert lifecycle == ["setup", "stop"] and log.steps == [] and len(log.captures) == 1
    assert driver.initial_observation.ego_speed == 12.
    assert log.captures[0].model_observation.ego_speed == 12.


def test_executed_control_lifecycle():
    controls = {"decision": [], "intermediate": [], "pre": [], "post": [], "feedback": []}

    class Feedback(Plugin):
        def transform(self, value, context):
            return value

        def after_step(self, context):
            controls["feedback"].append(context.control)

    def sensors(value, context):
        key = "decision" if round(context.physical_state.t * 1e6) % 100000 == 0 else "intermediate"
        controls[key].append(context.control)
        return value

    def remember(key):
        def callback(value, context):
            controls[key].append(context.control)
            return value
        return callback

    dataset = MemoryDataset()
    start = dataset.scenario_start("scene", dataset.frames("scene")[2])
    hooks = HookSet([Hook("observation", Feedback()), Hook("sensors.post", sensors),
        Hook("command", lambda value, context: Control(throttle=.2 + .1 * context.step)),
        Hook("simulator.step.pre", remember("pre")), Hook("simulator.step.post", remember("post"))])
    with MemorySimulator().create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, ConstantAVStack(Control(throttle=.2)), RunRequest(
            3, start=start, hooks=hooks,
            timing=TimingConfiguration(sensors={"front": SensorTiming(50000)},
                                       planner_delay_us=100000)))
    executed = [step.executed_command for step in log.steps]
    assert controls["decision"] == [None, None, None]
    assert (controls["pre"] == controls["post"] == controls["feedback"]
            == controls["intermediate"] == executed)


def test_run_termination(tmp_path):
    stopped = []
    received = []
    resource = (tmp_path / "owned-resource.txt").open("w")
    recorder = Recorder(path=tmp_path / "terminated")

    class ActiveAttack(AttackRuntime):
        def stop(self, context):
            stopped.append(context.stage)

    def terminate(record):
        received.append(record)
        return record.step == 2

    request = RunRequest(5, attacks=(ActiveAttack(),), terminate_when=terminate)
    with MemorySimulator().create_runtime() as runtime:
        log = ClosedLoopRunner().run(
            runtime, ConstantAVStack(), request, recorder=recorder,
            release_owned_resources=resource.close,
        )
        assert resource.closed
        assert stopped == ["stop"]
        assert [record.step for record in received] == [0, 1, 2]
        assert [record.actions for record in received] == [record.actions for record in log.steps]
        assert [record.outputs for record in received] == [record.outputs for record in log.steps]
        assert [record.executed_command for record in received] == [
            record.executed_command for record in log.steps]
        assert log.status == "completed"
        assert [step.step for step in log.steps] == [0, 1, 2]
        saved = RunLog.read(tmp_path / "terminated")
        assert len(saved.steps) == 3
        termination = next(event for event in saved.events if event.kind == "termination")
        assert termination.data["status"] == "completed"
        assert termination.data["terminated_by_condition"] is True
        full = ClosedLoopRunner().run(
            runtime, ConstantAVStack(), replace(request, terminate_when=None, attacks=()),
        )
        assert full.status == "completed"
        assert len(full.steps) == 5
        assert all("terminated_by_condition" not in event.data for event in full.events)


@pytest.mark.parametrize("failing_callback", ["after_step", "terminate_when", "plan_step"])
def test_run_records_completed_decision_on_handoff_failure(failing_callback):
    failure = RuntimeError("decision handoff failed")
    planned = []

    def fail():
        raise failure

    callbacks = dict.fromkeys(("after_step", "terminate_when", "plan_step"), lambda: None)
    callbacks[failing_callback] = fail

    class Feedback(Plugin):
        def transform(self, observation, context):
            return observation

        def after_step(self, context):
            callbacks["after_step"]()

    class Plan(AttackRuntime):
        def step(self, context):
            if context.step > 0:
                planned.append(context.step)
                context.record_output("planned_step", context.step)
                callbacks["plan_step"]()
            return ()

    def terminate(record):
        callbacks["terminate_when"]()
        return False

    request = RunRequest(2, hooks=HookSet((Hook("observation", Feedback()),)),
                         attacks=(Plan(),), terminate_when=terminate)
    with MemorySimulator().create_runtime() as runtime:
        with pytest.raises(RuntimeError, match="decision handoff failed") as caught:
            ClosedLoopRunner().run(runtime, ConstantAVStack(), request)
    assert caught.value is failure
    log = caught.value.run_log
    assert log.status == "failed"
    complete, = log.steps
    assert complete.step == 0
    assert not any(output.name == "planned_step" for output in complete.outputs)
    cleanup, = (event for event in log.events if event.kind == "attack_cleanup")
    assert [output.value for output in cleanup.data["outputs"]
            if output.name == "planned_step"] == planned


def test_run_with_hooks():
    hooks = HookSet([
        Hook("sensors.post", lambda value, context: replace(value, ego_speed=value.ego_speed + 1.)),
        Hook("observation", lambda value, context: replace(value, ego_speed=value.ego_speed + 5.,
            vehicle_state=PlanarEgoState(x=42., t=value.t))),
        Hook("command", lambda value, context: replace(value, brake=.7)),
    ])
    with MemorySimulator().create_runtime() as simulator_runtime:
        log = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(Control(throttle=.2)),
                                     RunRequest(2, hooks=hooks, recording=RECORDING))
    for record in log.steps:
        assert record.model_observation.ego_speed == record.raw_observation.ego_speed + 6.
        assert record.model_observation.delivered("front")[-1].state.vehicle_state.x == 42.
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
    past = avstack.initial_observation.delivered("front")[:-1]
    assert [observation.t for observation in past] == pytest.approx([-.2, -.1])
    assert [observation.state.ego_speed for observation in past] == [1., 1.]
    assert avstack.initial_observation.ego_speed == 6.
    raw = log.steps[0].raw_observation.sensor_data["front"]
    assert np.all(raw == 23)
    assert np.any(log.steps[0].model_observation.sensor_data["front"] != raw)
    with ClosedLoopServices(dataset, simulator) as services, services.session(7) as session:
        for historical, frame in zip(past, frames[1:3]):
            original = session.read_sensors(SensorRequest("scene", frame, ("front",)))
            conditioned = session.read_sensors(SensorRequest(
                "scene", frame, ("front",), environment=environment))
            np.testing.assert_array_equal(historical.data,
                                          conditioned.observation.sensor_data["front"])
            assert np.any(historical.data != original.observation.sensor_data["front"])
    saved = RunLog.read(tmp_path / "run")
    initialization = next(event for event in saved.events if event.kind == "initialization")
    assert initialization.data["past_captures"] == 2
    np.testing.assert_allclose(
        [item["t"] for item in initialization.data["past_observations"]], [-.2, -.1])
    assert all(set(item) == {"t", "vehicle_state"}
               for item in initialization.data["past_observations"])
    with simulator.create_runtime() as runtime:
        empty = ClosedLoopRunner().run(runtime, avstack, RunRequest(
            0, start=start, environment=environment, hooks=hooks))
        assert empty.status == "completed" and not empty.steps
        assert len(avstack.initial_observation.delivered("front")) - 1 == 2
        assert avstack.initial_observation.ego_speed == 6.


def test_initial_delivery_context():
    dataset = MemoryDataset()
    start = dataset.scenario_start("scene", dataset.frames("scene")[3])
    driver = HistoryAVStack()
    driver.history = HistoryRequirement(duration_us=140000, samples=3)

    driver.observation_pipeline = ObservationPipeline(lambda value, context: StateEstimate(
        PlanarEgoState(x=float(context.physical_state.ego.pose[0, 3]), t=value.t), value.ego_speed))
    with MemorySimulator().create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, driver, RunRequest(1, start=start,
            timing=TimingConfiguration(sensors={"front": SensorTiming(33000, 17000)})))
    origin = next(capture.time_us for capture in log.captures if capture.capture_id == 0)
    samples = driver.initial_observation.delivered("front")
    assert [sample.time_us - origin for sample in samples] == [-66000, -33000, 0]
    assert [sample.frame_start_us - origin for sample in samples] == [-83000, -50000, -17000]
    assert [sample.state.time_us for sample in samples] == [sample.time_us for sample in samples]
    assert [estimate.time_us - origin
            for estimate in driver.initial_observation.estimated_states()] == [
                -200000, -100000, 0]


def test_run_with_attacks(tmp_path):
    class WrongStage(AttackRuntime):
        def step(self, context):
            return (AVStackAttackAction("sensors.post", lambda value, context: value),)

    with MemorySimulator().create_runtime() as runtime:
        recorder = Recorder()
        with pytest.raises(ValueError, match="use SensorAttackAction to modify sensor data"):
            ClosedLoopRunner().run(
                runtime, ConstantAVStack(), RunRequest(1, attacks=(WrongStage(),)),
                recorder=recorder)
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
        log = ClosedLoopRunner().run(
            simulator_runtime, ConstantAVStack(perception_output=[]), RunRequest(
                1, attacks=(OnlineAttack(),),
                recording=RecordingPolicy(("front", "lidar"), ("raw", "model_input"))))
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
        assert len(log.captures) == 2
        assert all(capture.model_observation is not None for capture in log.captures)
        assert any(event.kind == "failure" for event in log.events)
        assert stopped == ["stop"] and resource.closed
        # A failed run relinquishes the simulator reservation and handler binding.
        again = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(), RunRequest(1))
        assert again.status == "completed"
        with pytest.raises(KeyboardInterrupt) as cancelled:
            ClosedLoopRunner().run(
                simulator_runtime, ConstantAVStack(interrupt_on_position=0), RunRequest(1))
        assert cancelled.value.run_log.status == "cancelled"

    def remove_camera(value, context):
        context.record_output("capture", True)
        return replace(value, sensor_data={})

    def note(value, context):
        context.record_output(context.stage, True)
        return value

    hooks = HookSet([Hook("sensors.post", remove_camera),
                     Hook("localization.post", note), Hook("observation", note)])
    with MemorySimulator().create_runtime() as runtime:
        with pytest.raises(ValueError, match="unavailable sensors") as missing:
            ClosedLoopRunner().run(runtime, ConstantAVStack(),
                RunRequest(1, hooks=hooks, recording=RECORDING))
    failed = next(event for event in missing.value.run_log.events if event.kind == "failure")
    assert [output.name for output in failed.data["outputs"]] == [
        "capture", "localization.post", "observation"]

    def fail_input(value, context):
        if context.step == 1:
            raise RuntimeError("input failed")
        return value

    with MemorySimulator().create_runtime() as runtime:
        with pytest.raises(RuntimeError, match="input failed") as failed_input:
            ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(3,
                recording=RECORDING,
                timing=TimingConfiguration(sensors={"front": SensorTiming(33000)}),
                hooks=HookSet([Hook("observation", fail_input)])))
    captures = failed_input.value.run_log.captures
    assert [capture.time_us for capture in captures] == [0, 33000, 66000, 99000]
    assert captures[0].model_observation is not None
    assert all(capture.model_observation is None for capture in captures[1:])
    assert all("front" in capture.raw_observation.sensor_data for capture in captures)

    def fail_capture(value, context):
        raise RuntimeError("sensor processing failed")

    with MemorySimulator().create_runtime() as runtime:
        with pytest.raises(RuntimeError, match="sensor processing failed") as failed_capture:
            ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(1,
                recording=RECORDING, hooks=HookSet([Hook("sensors.post", fail_capture)])))
    captures = failed_capture.value.run_log.captures
    assert len(captures) == 1 and captures[0].model_observation is None
    assert "front" in captures[0].raw_observation.sensor_data


def test_run_with_inserted_object(make_object_in_front):
    cameras = {name: SensorGeometry(name, CAMERA, CAM_FROM_WORLD) for name in ("rear", "front")}
    with MemorySimulator(camera_geometries=cameras).create_runtime() as runtime:
        runtime.reset()
        clean = runtime.sensor_view("front")
        asset, pose = make_object_in_front(clean)

        class InsertAttackRuntime(AttackRuntime):
            def setup(self, context):
                return (InsertObject("sign", asset, pose),)

        log = ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(
            1, attacks=(InsertAttackRuntime(),),
            recording=RecordingPolicy(tuple(cameras), ("raw", "model_input"))))
    step = log.steps[0]
    assert tuple(step.sensor_geometry) == tuple(cameras)
    assert tuple(step.object_visibility["sign"]) == ("front", "rear")
    assert np.any(step.raw_observation.sensor_data["front"] != clean.data)
    initialization = next(event for event in log.events if event.kind == "initialization")
    receipts = [item for item in initialization.data["actions"] if item["kind"] == "receipt"]
    assert any(item["outcome"] == "applied" and item["details"]["object_id"] == "sign"
               for item in receipts)
    assert step.object_visibility["sign"]["front"].visible_pixels > 0


@pytest.mark.parametrize("stage", ["sensors.post", "observation"])
@pytest.mark.parametrize("recording", [
    RecordingPolicy(("rear",), ("model_input",)),
    RecordingPolicy(("front",), ("raw",)),
])
def test_physical_evidence_after_input_drop(stage, recording, make_object_in_front):
    cameras = {name: SensorGeometry(name, CAMERA, CAM_FROM_WORLD) for name in ("rear", "front")}
    with MemorySimulator(camera_geometries=cameras).create_runtime() as runtime:
        runtime.reset()
        asset, pose = make_object_in_front(runtime.sensor_view("front"))

        class Insert(AttackRuntime):
            def setup(self, context):
                return (InsertObject("sign", asset, pose),)

        def drop(value, context):
            return replace(value, sensor_data={key: data for key, data in value.sensor_data.items()
                                               if key != "front"})

        log = ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(2,
            attacks=(Insert(),), recording=recording, hooks=HookSet([Hook(stage, drop)])))
    for step in log.steps:
        assert tuple(step.sensor_geometry) == recording.sensor_ids
        assert set(step.object_visibility["sign"]) == set(cameras)
        assert set(step.physical_captures) == set(cameras)
        assert "front" not in step.model_observation.sensor_data


def test_run_with_decelerating_feedback(nurec_longitudinal_dataset):
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
    configured = RecordingPolicy(("front",), ("raw",), every_n_captures=2)
    assert pickle.loads(pickle.dumps(configured)) == configured
    assert hash(configured) == hash(RecordingPolicy(("front",), ("raw",), every_n_captures=2))
    policy = RecordingPolicy(("front",), ("raw",), every_n_captures=2)
    captured = policy.observation(observation, 0, "raw")
    assert list(captured.sensor_data) == ["front"]
    assert captured.calibration == observation.calibration
    assert policy.observation(observation, 1, "raw").sensor_data == {}
    assert policy.observation(observation, 2, "model_input").sensor_data == {}
    assert list(observation.sensor_data) == ["front", "other"]
    sampled = RecordingPolicy(("front",), ("raw", "model_input"), every_n_captures=2)
    assert sampled.observation(observation, 0, "raw").sensor_data.keys() == {"front"}
    assert sampled.observation(observation, 1, "model_input").sensor_data == {}
    assert sampled.observation(observation, 2, "model_input").sensor_data.keys() == {"front"}
    missing = replace(observation, sensor_data={})
    assert sampled.observation(missing, 1, "raw").sensor_data == {}
    with pytest.raises(ValueError, match="unavailable sensors"):
        sampled.observation(missing, 2, "raw")
    with pytest.raises(ValueError, match="every_n_captures"):
        RecordingPolicy(every_n_captures=0)
    with pytest.raises(ValueError, match="representations"):
        RecordingPolicy(representations=("unknown",))


def test_recorded_payload_values():
    def edit(observation, context):
        observation.sensor_data["modified"] += observation.frame
        observation.sensor_data["replaced"] = np.full(3, 91, np.uint8)
        observation.sensor_data["masked"].mask = [observation.frame == 0, observation.frame != 0]
        return observation

    sensors = {"modified": np.arange(3, dtype=np.uint8), "replaced": np.zeros(3, np.uint8),
               "masked": np.ma.array([1., 2.], mask=[False, False])}
    with MemorySimulator(camera_geometries={}, extra_sensors=sensors).create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(
            2, recording=RecordingPolicy(tuple(sensors), ("raw", "model_input")),
            hooks=HookSet([Hook("sensors.post", edit)])))
    for index, step in enumerate(log.steps):
        raw, model = step.raw_observation.sensor_data, step.model_observation.sensor_data
        np.testing.assert_array_equal(raw["modified"], [0, 1, 2])
        np.testing.assert_array_equal(raw["replaced"], [0, 0, 0])
        np.testing.assert_array_equal(raw["masked"].mask, [False, False])
        np.testing.assert_array_equal(model["modified"], np.arange(3) + index)
        np.testing.assert_array_equal(model["replaced"], [91, 91, 91])
        np.testing.assert_array_equal(model["masked"].mask, [index == 0, index != 0])
        capture = log.captures[index]
        for sensor in sensors:
            np.testing.assert_array_equal(
                model[sensor], capture.model_observation.sensor_data[sensor])
            assert not raw[sensor].flags.writeable and not model[sensor].flags.writeable


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


def test_multirate_acquisitions_and_delivery(tmp_path):
    localized_at, observations, acquisitions = [], [], {}

    def localize(value, context):
        state = context.physical_state
        localized_at.append((round(state.t * 1e6), tuple(value.sensor_data)))
        return StateEstimate(
            PlanarEgoState(x=float(state.ego.pose[0, 3]), t=value.t), state.ego_speed)

    def observe(value, context):
        value.sensor_data["front"] = np.full_like(value.sensor_data["front"], 81)
        observations.append(value)
        return value

    def remember_capture(value, context):
        acquisitions[round(value.t * 1e6)] = value.sensor_data.get("front")
        return value

    driver = ConstantAVStack(Control(throttle=.2))
    driver.observation_pipeline = ObservationPipeline(localize)
    timing = TimingConfiguration(sensors={"front": SensorTiming(33000), "imu": SensorTiming(50000)},
        pose_reporting_interval_us=40000)
    with MemorySimulator(extra_sensors={"imu": np.zeros(3)}).create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, driver, RunRequest(
            3, timing=timing, hooks=HookSet([Hook("sensors.post", remember_capture),
                                           Hook("observation", observe)]),
            environment={"sensors": [{"type": "Brightness", "camera": "front"}]},
            recording=RECORDING, output_path=tmp_path / "run"))
    assert [time for time, _ in localized_at] == [
        0, 40000, 80000, 100000, 140000, 180000, 200000, 240000, 280000]
    assert [set(sensors) for _, sensors in localized_at] == [
        {"front", "imu"}, set(), set(), {"imu"}, set(), set(), {"imu"}, set(), set()]
    assert [capture.time_us for capture in log.captures] == [
        0, 33000, 50000, 66000, 99000, 100000, 132000, 150000, 165000, 198000,
        200000, 231000, 250000, 264000, 297000, 300000]
    assert [[sample.time_us for sample in value.delivered("front")] for value in observations] == [
        [0], [33000, 66000, 99000], [132000, 165000, 198000]]
    for observation in observations:
        for sample in observation.delivered("front"):
            assert sample.state.time_us == sample.time_us
            assert sample.state.t == sample.t == sample.time_us / 1e6
            assert sample.state.vehicle_state.t == sample.t
    assert [[state.time_us for state in value.estimated_states()] for value in observations] == [
        [0], [40000, 80000, 100000], [140000, 180000, 200000]]
    delivered = [sample for value in observations for sample in value.delivered("front")]
    assert len({sample.capture_id for sample in delivered}) == len(delivered)
    for value in observations:
        for sample in value.delivered("front")[:-1]:
            np.testing.assert_array_equal(sample.data, acquisitions[sample.time_us])
    first, last = observations[1].estimated_states()[:2]
    middle = observations[1].delivered("front")[1].state
    weight = (middle.time_us - first.time_us) / (last.time_us - first.time_us)
    assert middle.vehicle_state.x == pytest.approx(
        first.vehicle_state.x + weight * (last.vehicle_state.x - first.vehicle_state.x))
    assert log.steps[1].raw_observation.sensor_data == {}
    assert all(tuple(step.sensor_geometry) == ("front",) for step in log.steps)
    saved = RunLog.read(tmp_path / "run")
    assert saved.format_version == 2
    assert saved.steps[-1].delivered_captures == log.steps[-1].delivered_captures
    captures = {capture.time_us: capture for capture in saved.captures}
    for time_us in (0, 99000, 198000):
        assert np.all(captures[time_us].model_observation.sensor_data["front"] == 81)
    for time_us in (33000, 66000, 132000, 165000):
        np.testing.assert_array_equal(captures[time_us].model_observation.sensor_data["front"],
                                      acquisitions[time_us])
        assert np.any(captures[time_us].model_observation.sensor_data["front"] !=
                      captures[time_us].raw_observation.sensor_data["front"])
    assert all(capture.model_observation is None for time_us, capture in captures.items()
               if time_us > 200000)


def test_persistent_sensor_actions():
    @dataclass
    class Paint(SensorAttackAction):
        active_for_us: int = 0
        sensor = "front"

        def apply(self, data, view):
            return np.full_like(data, 101)

    class PaintRuntime(AttackRuntime):
        def step(self, context):
            return (Paint(active_for_us=150000),) if context.step == 0 else ()

    with MemorySimulator().create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(
            2, timing=TimingConfiguration(sensors={"front": SensorTiming(25000)}),
            attacks=(PaintRuntime(),), recording=RECORDING))
    delivered_captures = [capture for capture in log.captures
                          if capture.model_observation is not None]
    assert [int(capture.model_observation.sensor_data["front"][0, 0, 0])
            for capture in delivered_captures] == [101, 101, 101, 101, 101]
    assert [capture.time_us for capture in log.captures if capture.model_observation is None] == [
        125000, 150000, 175000, 200000]

    class NoFrameRuntime(AttackRuntime):
        def step(self, context):
            return (Paint(active_for_us=50000),) if context.step == 1 else ()

    with MemorySimulator().create_runtime() as runtime:
        missing = ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(
            3, timing=TimingConfiguration(sensors={"front": SensorTiming(200000)}),
            attacks=(NoFrameRuntime(),)))
    receipts = [action for step in missing.steps for action in step.actions
                if action["kind"] == "receipt"]
    assert len(receipts) == 1 and receipts[0]["outcome"] == "not_applied"
    assert receipts[0]["reason"] == "no_frame_during_active_interval"


def test_action_at_skipped_perception():
    class FrameDriver(ConstantAVStack):
        supported_stages = frozenset({"perception.post"})

        def __call__(self, observation):
            if "front" in observation.sensor_data:
                self._emit("perception.post", [])
            return Control(throttle=.2)

    class PerceptionAttack(AttackRuntime):
        def step(self, context):
            return ((AVStackAttackAction("perception.post", lambda value, context: value),)
                    if context.step == 1 else ())

    with MemorySimulator().create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, FrameDriver(), RunRequest(3,
            attacks=(PerceptionAttack(),),
            timing=TimingConfiguration(sensors={"front": SensorTiming(150000)})))
    receipts = [action for step in log.steps for action in step.actions
                if action["kind"] == "receipt"]
    assert log.status == "completed" and len(receipts) == 1
    assert receipts[0]["outcome"] == "not_applied"
    assert receipts[0]["reason"] == "stage_not_executed"

    def drop(value, context):
        return replace(value, sensor_data={}) if context.step == 1 else value

    with MemorySimulator().create_runtime() as runtime:
        with pytest.raises(RuntimeError, match="Attack actions were not applied"):
            ClosedLoopRunner().run(runtime, FrameDriver(), RunRequest(3,
                attacks=(PerceptionAttack(),), hooks=HookSet([Hook("sensors.post", drop)])))


def test_dropped_deliveries():
    inputs = []

    def drop(value, context):
        return replace(value, sensor_data={}, calibration={}) if context.step == 1 else value

    def remember(value, context):
        inputs.append(value)
        return value

    hooks = HookSet([Hook("sensors.post", drop), Hook("observation", remember)])
    with MemorySimulator().create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(3, hooks=hooks))
    assert log.steps[1].detection_capture_id is None
    assert "front" not in inputs[1].sensor_data and inputs[1].deliveries["front"] == ()
    assert inputs[0].delivered("front")[-1].capture_id == 0
    assert inputs[2].delivered("front")[-1].capture_id == 2
    with MemorySimulator().create_runtime() as runtime:
        raw_only = ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(
            3, hooks=hooks, recording=RecordingPolicy(("front",), ("raw",))))
    assert "front" in raw_only.steps[1].raw_observation.sensor_data
    assert "front" not in raw_only.steps[1].model_observation.sensor_data

    def drop_last_frame(value, context):
        if round(value.t * 1e6) == 99000:
            return replace(value, sensor_data={}, calibration={})
        return value

    inputs.clear()
    with MemorySimulator().create_runtime() as runtime:
        dropped = ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(2,
            timing=TimingConfiguration(sensors={"front": SensorTiming(33000)}), recording=RECORDING,
            hooks=HookSet([Hook("sensors.post", drop_last_frame), Hook("observation", remember)])))
    assert [sample.time_us for sample in inputs[1].delivered("front")] == [33000, 66000]
    np.testing.assert_array_equal(inputs[1].sensor_data["front"],
                                  inputs[1].delivered("front")[-1].data)
    assert dropped.steps[1].physical_captures["front"] == 3
    assert dropped.steps[1].delivered_captures["front"] == (1, 2)
    assert dropped.captures[3].model_observation is None
    assert "front" in dropped.captures[3].raw_observation.sensor_data


def test_force_gt(nurec_longitudinal_dataset):
    class CommandRuntime(AttackRuntime):
        def step(self, context):
            assert context.decision_period_us == 100000
            assert context.force_gt == (context.step < 2)
            return (AVStackAttackAction("command", lambda value, ctx: value),)

    dataset = nurec_longitudinal_dataset
    frames = dataset.frames("scene")
    camera, _ = dataset.camera_geometry("scene")
    simulator = NuRecSimulator(NuRecImageBoundary(camera), TrajectoryFollower(),
        time_step=.1, depth="none", route_generator="none", physics_update_mode="none")
    with simulator.create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, DeceleratingAVStack(), RunRequest(3,
            start=dataset.scenario_start("scene", frames[15]),
            attacks=(CommandRuntime(),),
            timing=TimingConfiguration(force_gt_duration_us=200000,
                                       skip_driver_during_force_gt=True)))
    receipts = [action for step in log.steps for action in step.actions
                if action["kind"] == "receipt"]
    assert [action["outcome"] == "not_applied" for action in receipts] == [True, True, False]
    assert [step.force_gt for step in log.steps] == [True, True, False]
    assert [step.produced_command is None for step in log.steps] == [True, True, False]
    assert [step.components for step in log.steps[:2]] == [{}, {}]
    for step, frame in zip(log.steps[:2], frames[16:18]):
        np.testing.assert_array_equal(
            step.physical_post.ego.pose, dataset.frame_data("scene", frame).ego.pose)
        assert step.executed_command is not None
        assert step.executed_command_issue_time_us == round(step.physical_pre.t * 1e6)

    recorded = MemoryDataset()
    start = recorded.scenario_start("scene", recorded.frames("scene")[3])
    with MemorySimulator().create_runtime() as runtime:
        actuator = ClosedLoopRunner().run(
            runtime, ConstantAVStack(Control(throttle=.2)), RunRequest(
                3, start=start, timing=TimingConfiguration(
                    force_gt_duration_us=200000, planner_delay_us=100000)))
    assert [step.force_gt for step in actuator.steps] == [True, True, False]
    assert [step.executed_command for step in actuator.steps[:2]] == [None, None]
    assert [step.executed_command_issue_time_us for step in actuator.steps] == [None, None, 400000]
    assert [step.produced_command for step in actuator.steps] == [Control(throttle=.2)] * 3
    assert actuator.steps[2].executed_command == Control(throttle=.2)
    for step, frame in zip(actuator.steps[:2], recorded.frames("scene")[4:6]):
        np.testing.assert_array_equal(
            step.physical_post.ego.pose, recorded.frame_data("scene", frame).ego.pose)


@pytest.mark.parametrize("lidar_interval_us", [100000, 70000])
def test_one_shot_sensor_action_requires_issuing_frame(lidar_interval_us):
    applied = []

    class Paint(SensorAttackAction):
        sensor = "front"

        def apply(self, data, view):
            applied.append(view.time)
            return np.full_like(data, 100)

    class Runtime(AttackRuntime):
        def step(self, context):
            return (Paint(),) if context.step == 1 else ()

    simulator = MemorySimulator(extra_sensors={"lidar": np.zeros((2, 3), np.float32)})
    with simulator.create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(3,
            attacks=(Runtime(),),
            timing=TimingConfiguration(sensors={"front": SensorTiming(33000),
                                               "lidar": SensorTiming(lidar_interval_us)}),
            recording=RecordingPolicy(("front", "lidar"), ("raw", "model_input"))))
    assert log.status == "completed" and applied == []
    receipts = [action for step in log.steps for action in step.actions
                if action["kind"] == "receipt"]
    assert len(receipts) == 1
    assert receipts[0]["step"] == 1 and receipts[0]["sim_time"] == .1
    assert receipts[0]["outcome"] == "not_applied"
    assert receipts[0]["reason"] == "sensor_not_acquired_at_issue_time"
    for capture in (item for item in log.captures if item.model_observation is not None):
        assert set(capture.model_observation.sensor_data) <= set(capture.sensor_ids)
        np.testing.assert_array_equal(capture.raw_observation.sensor_data.get("front"),
                                      capture.model_observation.sensor_data.get("front"))


@pytest.mark.parametrize("stage", ["sensors.post", "localization.pre", "observation"])
def test_input_hook_payloads(tmp_path, stage):
    def edit(value, context):
        value.sensor_data["front"].fill(81)
        value.calibration["front"] = "edited lens"
        return value

    driver = ConstantAVStack()
    driver.observation_pipeline = ObservationPipeline(
        lambda value, context: StateEstimate(value.vehicle_state, value.ego_speed))
    cameras = {name: SensorGeometry(name, CAMERA, CAM_FROM_WORLD) for name in ("front", "right")}
    recording = RecordingPolicy(("right", "front"), ("raw", "model_input"), every_n_captures=2)
    with MemorySimulator(camera_geometries=cameras).create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, driver, RunRequest(
            3, hooks=HookSet([Hook(stage, edit)]), recording=recording))
    assert list(log.steps[0].sensor_geometry) == ["right", "front"]
    assert list(log.steps[2].sensor_geometry) == ["right", "front"]
    assert log.steps[1].sensor_geometry == {}
    for step in (log.steps[0], log.steps[2]):
        assert np.all(step.model_observation.sensor_data["front"] == 81)
        assert step.model_observation.calibration == {"front": "edited lens"}
        assert "right" not in step.model_observation.calibration
        assert np.all(step.raw_observation.sensor_data["front"] != 81)
    assert log.steps[1].raw_observation.sensor_data == {}
    assert log.steps[1].model_observation.sensor_data == {}
    for capture in log.captures:
        retained = set(capture.raw_observation.sensor_data)
        if capture.model_observation is not None:
            retained.update(capture.model_observation.sensor_data)
        assert set(capture.sensor_geometry) == retained & set(cameras)

    model_driver = ConstantAVStack()
    model_driver.observation_pipeline = ObservationPipeline(
        lambda value, context: StateEstimate(value.vehicle_state, value.ego_speed))
    output = tmp_path / stage
    with MemorySimulator(camera_geometries=cameras).create_runtime() as runtime:
        ClosedLoopRunner().run(runtime, model_driver, RunRequest(
            3, hooks=HookSet([Hook(stage, edit)]), output_path=output, retain_sensor_data=False,
            recording=RecordingPolicy(("front",), ("model_input",))))
    persisted = RunLog.read(output)
    assert len(persisted.captures) == 4
    for capture in persisted.captures[:-1]:
        assert capture.raw_observation.sensor_data == {}
        assert np.all(capture.model_observation.sensor_data["front"] == 81)
        assert capture.model_observation.calibration == {"front": "edited lens"}
    assert persisted.captures[-1].model_observation is None


def test_command_hook_snapshot():
    def brake(value, context):
        value.throttle, value.brake = 0., .7
        return value

    with MemorySimulator().create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, ConstantAVStack(Control(throttle=.3)), RunRequest(
            1, hooks=HookSet([Hook("command", brake)])))
    assert log.steps[0].produced_command.throttle == .3
    assert log.steps[0].produced_command.brake == 0.
    assert log.steps[0].executed_command.throttle == 0.
    assert log.steps[0].executed_command.brake == .7


@pytest.mark.parametrize("delay_us, issue_times_us", [
    (0, [300000, 400000, 500000, 600000]),
    (73000, [300000, 300000, 400000, 500000]),
    (200000, [300000, 300000, 300000, 400000]),
])
def test_planner_delay(delay_us, issue_times_us):
    dataset = MemoryDataset()
    start = dataset.scenario_start("scene", dataset.frames("scene")[3])
    with MemorySimulator().create_runtime() as runtime:
        log = ClosedLoopRunner().run(runtime, ConstantAVStack(), RunRequest(
            4, start=start, timing=TimingConfiguration(planner_delay_us=delay_us)))
    assert [step.executed_command_issue_time_us for step in log.steps] == issue_times_us
