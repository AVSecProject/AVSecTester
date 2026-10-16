"""Separated capture/clock operations, legacy adapters and initialization ordering."""

import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.backend import WorldBackend
from avsectester.plane import Control, Observation, WorldSnapshot
from avsectester.simulator_runtime import SimulatorRuntime
from avsectester.simulators.carla import CarlaBackend
from avsectester.simulators.nurec import NuRecBackend, StubRenderer


class LegacyBackend(WorldBackend):
    def __init__(self):
        self.frame = 0
        self.resets = self.steps = self.closes = 0

    def reset(self):
        self.resets += 1
        self.frame = 0
        return Observation(8.0, self.frame, sensor_data={"sample": [0]}, ego_speed=2.0)

    def step(self, control):
        self.steps += 1
        self.frame += 1
        return Observation(8.0 + self.frame / 10, self.frame, ego_speed=2.0)

    def close(self):
        self.closes += 1


class CountingRenderer(StubRenderer):
    def __init__(self):
        super().__init__(height=2, width=3)
        self.positions = []
        self.closes = 0

    def render(self, pose, camera):
        self.positions.append((pose.t, pose.x))
        return np.full((2, 3, 3), len(self.positions), dtype=np.uint8)

    def close(self):
        self.closes += 1


def test_legacy_bridge_caches_captures_without_extra_backend_calls():
    backend = LegacyBackend()
    simulator = SimulatorRuntime(backend)
    assert simulator.reset() is None
    first = simulator.observe()
    first.sensor_data["sample"].append(99)
    first.t = -1

    assert simulator.observe().sensor_data == {"sample": [0]}
    assert simulator.clock.t == 8
    assert backend.resets == 1 and backend.steps == 0
    assert simulator.step(Control()) is None
    assert simulator.clock.frame == 1 and simulator.clock.t == 8.1
    assert simulator.observe().frame == 1
    assert backend.steps == 1


def test_clock_and_state_are_independent_snapshots():
    backend = NuRecBackend({"ego0": {"speed": 4.0}}, renderer=CountingRenderer())
    simulator = SimulatorRuntime(backend)
    simulator.reset()
    clock = simulator.clock
    with pytest.raises(FrozenInstanceError):
        clock.t = -1
    state = simulator.state()
    state.vehicle_state.speed = -100
    assert simulator.state().ego_speed == 4
    simulator.step(Control())
    assert clock.t == 0 and clock.frame == 0
    assert simulator.clock.frame == 1


def test_nurec_observe_reuses_capture_and_isolates_caller_mutation():
    renderer = CountingRenderer()
    backend = NuRecBackend({"dt": 0.1}, renderer=renderer)
    simulator = SimulatorRuntime(backend)
    simulator.reset()
    observation = simulator.observe()
    observation.sensor_data["camera_front"][:] = 100
    observation.vehicle_state.x = 100
    for _ in range(3):
        assert (simulator.observe().sensor_data["camera_front"] == 1).all()
        assert simulator.observe().vehicle_state.x == 0
    assert len(renderer.positions) == 1

    simulator.step(Control(throttle=1.0))
    assert len(renderer.positions) == 2
    assert simulator.clock.t == pytest.approx(0.1)
    assert (simulator.observe().sensor_data["camera_front"] == 2).all()
    assert len(renderer.positions) == 2


def test_nurec_setup_precedes_capture_and_start_sees_selected_state():
    from avsectester.attack_runtime import AttackRuntime, ResultSubscription
    from avsectester.closed_loop import ClosedLoopRunner, RunRequest
    from avsectester.backend import AVStack
    from avsectester.runtime import Hook, HookSet

    renderer = CountingRenderer()
    backend = NuRecBackend({}, renderer=renderer)
    events = []
    def setup(value, context):
        assert not renderer.positions
        backend.pose.x = 7
        events.append("setup")
    class Attack(AttackRuntime):
        reads = (ResultSubscription("physical_state"),)
        def start(self, context):
            assert renderer.positions == [(0, 7)]
            assert context.physical_state.vehicle_state.x == 7
            events.append("start")
            return ()
    class Driver(AVStack):
        def __call__(self, observation):
            return Control()
    ClosedLoopRunner().run(backend, Driver(), RunRequest(
        0, hooks=HookSet((Hook("world.setup", setup),)), attacks=(Attack(),)))
    assert events == ["setup", "start"]


def test_restore_invalidates_capture_without_advancing_restored_world():
    renderer = CountingRenderer()
    backend = NuRecBackend({"dt": 0.1}, renderer=renderer)
    simulator = SimulatorRuntime(backend)
    simulator.reset()
    checkpoint = backend.checkpoint()
    simulator.step(Control(throttle=1))
    backend.restore(checkpoint)

    assert simulator.clock.frame == 0 and simulator.clock.t == 0
    restored = simulator.observe()
    assert restored.frame == 0 and restored.vehicle_state.x == 0
    assert (restored.sensor_data["camera_front"] == 3).all()
    simulator.observe()
    assert len(renderer.positions) == 3


@pytest.mark.parametrize("method", ["reset", "step"])
def test_capture_backend_rejects_legacy_overrides_at_class_definition(method):
    with pytest.raises(TypeError, match="legacy reset/step"):
        type("InvalidBackend", (NuRecBackend,), {method: lambda self: None})


def test_native_initialization_extension_preserves_config():
    class NewInitialization(NuRecBackend):
        def _initialize_world(self, source, initialization, environment):
            super()._initialize_world(source, initialization, environment)
            self.pose.x = 6
    backend = NewInitialization({}, renderer=CountingRenderer())
    simulator = SimulatorRuntime(backend)
    simulator.reset()
    assert simulator.observe().vehicle_state.x == 6
    assert backend.config == {}


def test_unsupported_initialization_rejects_before_reset_side_effects():
    backend = LegacyBackend()
    simulator = SimulatorRuntime(backend)
    with pytest.raises(TypeError, match="SourceData"):
        simulator.reset(source={"scene": "unsupported"})
    with pytest.raises(ValueError, match="initialization"):
        simulator.reset(initialization={"ticks": 2})
    assert backend.resets == 0


def test_uninitialized_and_failed_advances_do_not_expose_stale_capture():
    class FailingBackend(LegacyBackend):
        def step(self, control):
            super().step(control)
            raise RuntimeError("capture failed")

    simulator = SimulatorRuntime(FailingBackend())
    for operation in (simulator.observe, simulator.state, lambda: simulator.step(Control())):
        with pytest.raises(RuntimeError, match="Reset"):
            operation()
    simulator.reset()
    with pytest.raises(RuntimeError, match="capture failed"):
        simulator.step(Control())
    with pytest.raises(RuntimeError, match="Reset"):
        simulator.observe()


def test_native_capture_failure_requires_reset_and_preserves_physical_state():
    class FailingRenderer(CountingRenderer):
        def render(self, pose, camera):
            if pose.t:
                raise RuntimeError("renderer disconnected")
            return super().render(pose, camera)

    backend = NuRecBackend({"dt": 0.1}, renderer=FailingRenderer())
    simulator = SimulatorRuntime(backend)
    simulator.reset()
    with pytest.raises(RuntimeError, match="renderer disconnected"):
        simulator.step(Control(throttle=1))
    assert backend.ground_truth().frame == 1
    assert backend.ground_truth().t == pytest.approx(0.1)
    for observe in (simulator.observe, backend.observe):
        with pytest.raises(RuntimeError, match="Reset"):
            observe()
    simulator.reset()
    assert simulator.observe().frame == 0


def test_failed_start_cannot_leave_previous_capture_readable():
    renderer = CountingRenderer()
    backend = NuRecBackend({}, renderer=renderer)
    simulator = SimulatorRuntime(backend)
    simulator.reset()

    def fail(*args, **kwargs):
        raise RuntimeError("deployment failed")

    with pytest.raises(RuntimeError, match="deployment failed"):
        backend._emit = fail
        simulator.reset()
    assert len(renderer.positions) == 1
    for observe in (simulator.observe, backend.observe):
        with pytest.raises(RuntimeError, match="Reset"):
            observe()
    del backend._emit
    simulator.reset()
    assert len(renderer.positions) == 2


def test_bridge_close_does_not_close_borrowed_backend():
    backend = LegacyBackend()
    simulator = SimulatorRuntime(backend)
    simulator.reset()
    simulator.close()
    assert backend.closes == 0
    backend.close()
    assert backend.closes == 1


@pytest.mark.parametrize("setup_ticks", [0, 2])
def test_carla_plain_reset_ticks_after_world_setup_and_observe_never_ticks(monkeypatch, setup_ticks):
    events = []
    snapshot = SimpleNamespace(frame=0, timestamp=SimpleNamespace(elapsed_seconds=0.0))

    def tick():
        events.append("tick")
        snapshot.frame += 1
        snapshot.timestamp.elapsed_seconds += 0.05

    client = SimpleNamespace(
        world=SimpleNamespace(get_snapshot=lambda: snapshot, get_actors=lambda: []), tick=tick, close=lambda: None,
    )
    transform = SimpleNamespace(
        location=SimpleNamespace(x=0, y=0, z=0),
        rotation=SimpleNamespace(pitch=0, yaw=0, roll=0),
    )
    ego = SimpleNamespace(
        actor=SimpleNamespace(type_id="vehicle.test"), spawn_transform=transform,
        initialize=lambda *args: None, destroy=lambda: None,
        apply_control=lambda control: events.append("control"),
    )
    registry = SimpleNamespace(build=lambda config, **kwargs: ego if kwargs else client)
    monkeypatch.setitem(sys.modules, "avcarla.config", SimpleNamespace(CARLA=registry))
    backend = CarlaBackend({"client": {}, "ego": {"spawn": 0}})

    def capture():
        events.append("capture")
        return Observation(snapshot.timestamp.elapsed_seconds, snapshot.frame)

    def emit(stage, value, **kwargs):
        events.append(stage)
        return value

    monkeypatch.setattr(backend, "_observe", capture)
    monkeypatch.setattr(backend, "_emit", emit)
    monkeypatch.setattr(
        backend, "ground_truth",
        lambda: WorldSnapshot(snapshot.timestamp.elapsed_seconds, snapshot.frame),
    )
    backend.setup_ticks = setup_ticks
    simulator = SimulatorRuntime(backend)

    def initialize():
        events.append("initialize")
        # Plain resets keep the original order: no native tick before world.setup.
        assert simulator.state().frame == 0
        # An allocated deployment tick consumes the shared budget, not an extra decision.
        if setup_ticks:
            backend.setup_tick()

    original_emit = backend._emit
    def setup_emit(stage, value, **kwargs):
        result = original_emit(stage, value, **kwargs)
        if stage == "world.setup":
            initialize()
        return result
    backend._emit = setup_emit
    simulator.reset()
    # world.setup, its k deployment ticks, then the capture tick that makes them visible.
    assert events == ["world.setup", "initialize", *(["tick"] * (setup_ticks + 1)), "capture"]
    assert simulator.observe().frame == simulator.observe().frame == setup_ticks + 1
    assert events.count("capture") == 1
    assert events.count("tick") == setup_ticks + 1
    simulator.step(Control())
    assert events[-5:] == ["world.step.pre", "control", "tick", "world.step.post", "capture"]
    assert simulator.observe().frame == setup_ticks + 2
    assert events.count("capture") == 2
    assert events.count("tick") == setup_ticks + 2
    backend.close()
