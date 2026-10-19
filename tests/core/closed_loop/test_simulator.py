"""Simulator capture/clock operations, the legacy adapter and initialization ordering."""

import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.closed_loop.simulator import WorldBackend
from avsectester.closed_loop.datatypes import Control, Observation, WorldSnapshot
from avsectester.closed_loop.simulator import LegacySimulatorAdapter
from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.closed_loop.simulators.nurec import NuRecSimulatorRuntime, NuRecSimulator, StubRenderer


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


def test_legacy_adapter_caches_captures_without_extra_backend_calls():
    backend = LegacyBackend()
    simulator = LegacySimulatorAdapter(backend)
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
    backend = NuRecSimulator({"ego0": {"speed": 4.0}}, renderer=CountingRenderer()).create_runtime()
    simulator = backend
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
    backend = NuRecSimulator({"dt": 0.1}, renderer=renderer).create_runtime()
    simulator = backend
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


def test_attack_setup_world_effect_is_visible_in_the_first_capture():
    from avsectester.attack.runtime import AttackRuntime, ResultSubscription, WorldAttackEffect
    from avsectester.closed_loop.runner import ClosedLoopRunner, RunRequest
    from avsectester.closed_loop.avstack import AVStack
    from avsectester.closed_loop.hooks import Hook, HookSet

    renderer = CountingRenderer()
    backend = NuRecSimulator({}, renderer=renderer).create_runtime()
    events = []
    def setup(value, context):
        assert not renderer.positions
        backend.pose.x = 7
        events.append("hook")
    class Move(WorldAttackEffect):
        def apply(self, simulator, resources):
            simulator.pose.x += 2
    class Attack(AttackRuntime):
        subscriptions = (ResultSubscription("physical_state"),)
        def setup(self, context):
            # After simulator.setup hooks and before the first capture.
            assert not renderer.positions
            assert context.physical_state.vehicle_state.x == 7
            events.append("setup")
            return (Move(),)
    class Driver(AVStack):
        def __call__(self, observation):
            return Control()
    log = ClosedLoopRunner().run(backend, Driver(), RunRequest(
        0, hooks=HookSet((Hook("simulator.setup", setup),)), attacks=(Attack(),)))
    assert events == ["hook", "setup"]
    assert renderer.positions == [(0, 9)]
    initialization = next(event for event in log.events if event.kind == "initialization")
    receipt = next(e for e in initialization.data["effects"] if e["kind"] == "receipt")
    assert receipt["category"] == "world" and receipt["stage"] == "simulator.setup"


def test_restore_invalidates_capture_without_advancing_restored_world():
    renderer = CountingRenderer()
    backend = NuRecSimulator({"dt": 0.1}, renderer=renderer).create_runtime()
    simulator = backend
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
def test_simulator_rejects_legacy_overrides_at_class_definition(method):
    with pytest.raises(TypeError, match="not reset/step"):
        type("InvalidRuntime", (NuRecSimulatorRuntime,), {method: lambda self: None})


def test_native_initialization_extension_preserves_config():
    class NewInitialization(NuRecSimulatorRuntime):
        def __init__(self, *args, **kwargs):
            super().__init__(NuRecSimulator(*args, **kwargs))

        def _initialize_world(self, source, initialization, environment):
            super()._initialize_world(source, initialization, environment)
            self.pose.x = 6
    backend = NewInitialization({}, renderer=CountingRenderer())
    simulator = backend
    simulator.reset()
    assert simulator.observe().vehicle_state.x == 6
    assert backend.config == {}


def test_unsupported_initialization_rejects_before_reset_side_effects():
    backend = LegacyBackend()
    simulator = LegacySimulatorAdapter(backend)
    with pytest.raises(TypeError, match="ScenarioStart"):
        simulator.reset(start={"scene": "unsupported"})
    with pytest.raises(ValueError, match="initialization"):
        simulator.reset(initialization={"ticks": 2})
    assert backend.resets == 0


def test_uninitialized_and_failed_advances_do_not_expose_stale_capture():
    class FailingBackend(LegacyBackend):
        def step(self, control):
            super().step(control)
            raise RuntimeError("capture failed")

    simulator = LegacySimulatorAdapter(FailingBackend())
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

    backend = NuRecSimulator({"dt": 0.1}, renderer=FailingRenderer()).create_runtime()
    simulator = backend
    simulator.reset()
    with pytest.raises(RuntimeError, match="renderer disconnected"):
        simulator.step(Control(throttle=1))
    # A failed capture leaves no observable world; the advanced physical state remains.
    assert backend._state().frame == 1
    assert backend._state().t == pytest.approx(0.1)
    for observe in (simulator.observe, backend.observe):
        with pytest.raises(RuntimeError, match="Reset"):
            observe()
    simulator.reset()
    assert simulator.observe().frame == 0


def test_failed_start_cannot_leave_previous_capture_readable():
    renderer = CountingRenderer()
    backend = NuRecSimulator({}, renderer=renderer).create_runtime()
    simulator = backend
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


def test_adapter_close_does_not_close_borrowed_backend():
    backend = LegacyBackend()
    simulator = LegacySimulatorAdapter(backend)
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
    backend = CarlaSimulator({"client": {}, "ego": {"spawn": 0}}).create_runtime()

    def capture():
        events.append("capture")
        return Observation(snapshot.timestamp.elapsed_seconds, snapshot.frame)

    def emit(stage, value, **kwargs):
        events.append(stage)
        return value

    monkeypatch.setattr(backend, "_observe", capture)
    monkeypatch.setattr(backend, "_emit", emit)
    monkeypatch.setattr(
        backend, "_state",
        lambda: WorldSnapshot(snapshot.timestamp.elapsed_seconds, snapshot.frame),
    )
    backend.setup_ticks = setup_ticks
    simulator = backend

    def initialize():
        events.append("initialize")
        # Plain resets keep the original order: no native tick before simulator.setup.
        assert simulator.state().frame == 0
        # An allocated deployment tick consumes the shared budget, not an extra decision.
        if setup_ticks:
            backend.setup_tick()

    original_emit = backend._emit
    def setup_emit(stage, value, **kwargs):
        result = original_emit(stage, value, **kwargs)
        if stage == "simulator.setup":
            initialize()
        return result
    backend._emit = setup_emit
    simulator.reset()
    # simulator.setup, its k deployment ticks, then the capture tick that makes them visible.
    assert events == ["simulator.setup", "initialize", *(["tick"] * (setup_ticks + 1)), "capture"]
    assert simulator.observe().frame == simulator.observe().frame == setup_ticks + 1
    assert events.count("capture") == 1
    assert events.count("tick") == setup_ticks + 1
    simulator.step(Control())
    assert events[-5:] == ["simulator.step.pre", "control", "tick", "simulator.step.post", "capture"]
    assert simulator.observe().frame == setup_ticks + 2
    assert events.count("capture") == 2
    assert events.count("tick") == setup_ticks + 2
    backend.close()


def test_carla_state_reads_other_actors_from_the_latest_tick_and_actors_can_be_taken_over(monkeypatch):
    from unittest.mock import Mock

    from avsectester.datasets.base import ActorState

    monkeypatch.setitem(sys.modules, "carla", SimpleNamespace(
        VehicleControl=lambda **kw: kw, Rotation=lambda **kw: kw, Location=lambda *a: a,
        Transform=lambda *a: a, Vector3D=lambda *a: a,
    ))
    simulated = SimpleNamespace(
        get_transform=lambda: SimpleNamespace(get_matrix=lambda: np.eye(4)),
        get_velocity=lambda: SimpleNamespace(x=1.0, y=2.0, z=0.0),
    )
    npc, lead = Mock(id=7, type_id="vehicle.audi.tt"), Mock(id=8, type_id="walker.pedestrian.0001")
    npc.bounding_box = SimpleNamespace(extent=SimpleNamespace(x=2.0, y=1.0, z=0.8))
    snapshot = SimpleNamespace(frame=3, timestamp=SimpleNamespace(elapsed_seconds=0.3),
                               find=lambda actor_id: simulated if actor_id in (1, 7) else None)
    backend = CarlaSimulator({"client": {}, "ego": {}}).create_runtime()
    backend.client = SimpleNamespace(world=SimpleNamespace(get_snapshot=lambda: snapshot),
                                     traffic_manager_port=8000)
    state = SimpleNamespace(velocity=SimpleNamespace(norm=lambda: 0.0))
    backend.ego = SimpleNamespace(actor=Mock(id=1), get_object_state=lambda: state)
    backend.npcs = [SimpleNamespace(actor=npc)]
    backend.lead = lead
    backend._recipe_replay = Mock()

    actors = backend._state().actors
    # The lead was spawned after the latest tick, so it has no simulated state yet.
    assert set(actors) == {"npc:0"}
    assert actors["npc:0"].velocity == (1.0, -2.0, 0.0) and actors["npc:0"].category == "vehicle"

    backend.control_actor("npc:0", Control(throttle=0.4, steer=-0.1))
    npc.apply_control.assert_called_once_with({"throttle": 0.4, "brake": 0.0, "steer": -0.1})
    npc.set_autopilot.assert_called_once_with(False, 8000)
    backend._recipe_replay.release.assert_called_once_with("npc:0")
    backend.set_actor_state("lead", ActorState(np.eye(4), (0.0, 0.0, 0.0)))
    lead.set_transform.assert_called_once()
    with pytest.raises(ValueError, match="driving system"):
        backend.control_actor("ego", Control())
    with pytest.raises(KeyError, match="Unknown CARLA actor"):
        backend.control_actor("npc:5", Control())


def test_reset_accepts_only_native_world_settings_before_creating_the_world():
    simulator = NuRecSimulator(config={}, renderer=StubRenderer())
    with simulator.create_runtime() as runtime:
        for envelope in ({"sensors": []}, {"world": {}}, {"sensors": [], "weather": "rain"}):
            with pytest.raises(ValueError, match="world settings"):
                runtime.reset(world=envelope)
            with pytest.raises(RuntimeError, match="Reset"):
                runtime.observe()
        runtime.reset(world={})
        assert runtime.observe() is not None
