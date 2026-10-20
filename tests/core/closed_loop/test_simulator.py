"""Simulator capture/clock operations and initialization ordering."""

import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.closed_loop.datatypes import Control
from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.closed_loop.simulators.nurec import NuRecSimulatorRuntime, NuRecSimulator, StubRenderer


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


def test_clock_and_state_are_independent_snapshots():
    simulator_runtime = NuRecSimulator({"ego0": {"speed": 4.0}}, renderer=CountingRenderer()).create_runtime()
    simulator_runtime.reset()
    clock = simulator_runtime.clock
    with pytest.raises(FrozenInstanceError):
        clock.t = -1
    state = simulator_runtime.state()
    state.vehicle_state.speed = -100
    assert simulator_runtime.state().ego_speed == 4
    simulator_runtime.step(Control())
    assert clock.t == 0 and clock.frame == 0
    assert simulator_runtime.clock.frame == 1


def test_nurec_observe_reuses_capture_and_isolates_caller_mutation():
    renderer = CountingRenderer()
    simulator_runtime = NuRecSimulator({"dt": 0.1}, renderer=renderer).create_runtime()
    simulator_runtime.reset()
    observation = simulator_runtime.observe()
    observation.sensor_data["camera_front"][:] = 100
    observation.vehicle_state.x = 100
    for _ in range(3):
        assert (simulator_runtime.observe().sensor_data["camera_front"] == 1).all()
        assert simulator_runtime.observe().vehicle_state.x == 0
    assert len(renderer.positions) == 1

    simulator_runtime.step(Control(throttle=1.0))
    assert len(renderer.positions) == 2
    assert simulator_runtime.clock.t == pytest.approx(0.1)
    assert (simulator_runtime.observe().sensor_data["camera_front"] == 2).all()
    assert len(renderer.positions) == 2


def test_attack_setup_world_effect_is_visible_in_the_first_capture():
    from avsectester.attack.runtime import AttackRuntime, ResultSubscription, WorldAttackEffect
    from avsectester.closed_loop.runner import ClosedLoopRunner, RunRequest
    from avsectester.closed_loop.avstack import AVStack
    from avsectester.closed_loop.hooks import Hook, HookSet

    renderer = CountingRenderer()
    simulator_runtime = NuRecSimulator({}, renderer=renderer).create_runtime()
    events = []
    def setup(value, context):
        assert not renderer.positions
        simulator_runtime.pose.x = 7
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
    log = ClosedLoopRunner().run(simulator_runtime, Driver(), RunRequest(
        0, hooks=HookSet((Hook("simulator.setup", setup),)), attacks=(Attack(),)))
    assert events == ["hook", "setup"]
    assert renderer.positions == [(0, 9)]
    initialization = next(event for event in log.events if event.kind == "initialization")
    receipt = next(e for e in initialization.data["effects"] if e["kind"] == "receipt")
    assert receipt["category"] == "world" and receipt["stage"] == "simulator.setup"


def test_restore_invalidates_capture_without_advancing_restored_world():
    renderer = CountingRenderer()
    simulator_runtime = NuRecSimulator({"dt": 0.1}, renderer=renderer).create_runtime()
    simulator_runtime.reset()
    checkpoint = simulator_runtime.checkpoint()
    simulator_runtime.step(Control(throttle=1))
    simulator_runtime.restore(checkpoint)

    assert simulator_runtime.clock.frame == 0 and simulator_runtime.clock.t == 0
    restored = simulator_runtime.observe()
    assert restored.frame == 0 and restored.vehicle_state.x == 0
    assert (restored.sensor_data["camera_front"] == 3).all()
    simulator_runtime.observe()
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
    simulator_runtime = NewInitialization({}, renderer=CountingRenderer())
    simulator_runtime.reset()
    assert simulator_runtime.observe().vehicle_state.x == 6
    assert simulator_runtime.config == {}


def test_native_capture_failure_requires_reset_and_preserves_physical_state():
    class FailingRenderer(CountingRenderer):
        def render(self, pose, camera):
            if pose.t:
                raise RuntimeError("renderer disconnected")
            return super().render(pose, camera)

    simulator_runtime = NuRecSimulator({"dt": 0.1}, renderer=FailingRenderer()).create_runtime()
    simulator_runtime.reset()
    with pytest.raises(RuntimeError, match="renderer disconnected"):
        simulator_runtime.step(Control(throttle=1))
    # A failed capture leaves no observable world; the advanced physical state remains.
    assert simulator_runtime._state().frame == 1
    assert simulator_runtime._state().t == pytest.approx(0.1)
    with pytest.raises(RuntimeError, match="Reset"):
        simulator_runtime.observe()
    simulator_runtime.reset()
    assert simulator_runtime.observe().frame == 0


def test_failed_start_cannot_leave_previous_capture_readable():
    renderer = CountingRenderer()
    simulator_runtime = NuRecSimulator({}, renderer=renderer).create_runtime()
    simulator_runtime.reset()

    def fail(*args, **kwargs):
        raise RuntimeError("deployment failed")

    with pytest.raises(RuntimeError, match="deployment failed"):
        simulator_runtime._emit = fail
        simulator_runtime.reset()
    assert len(renderer.positions) == 1
    with pytest.raises(RuntimeError, match="Reset"):
        simulator_runtime.observe()
    del simulator_runtime._emit
    simulator_runtime.reset()
    assert len(renderer.positions) == 2


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
    simulator_runtime = CarlaSimulator({}).create_runtime()
    simulator_runtime.client = SimpleNamespace(world=SimpleNamespace(get_snapshot=lambda: snapshot),
                                     traffic_manager_port=8000)
    state = SimpleNamespace(velocity=SimpleNamespace(norm=lambda: 0.0))
    ego_actor = SimpleNamespace(id=1, type_id="vehicle.ego",
        bounding_box=SimpleNamespace(extent=SimpleNamespace(x=2.0, y=1.0, z=0.7)),
        get_transform=simulated.get_transform, get_velocity=simulated.get_velocity)
    simulator_runtime.ego = SimpleNamespace(actor=ego_actor, get_object_state=lambda: state)
    simulator_runtime.npcs = [SimpleNamespace(actor=npc)]
    simulator_runtime.lead = lead
    simulator_runtime._recipe_replay = Mock()

    physical = simulator_runtime._state()
    assert physical.ego.extent == (4.0, 2.0, 1.4)
    assert physical.ego.velocity == (1.0, -2.0, 0.0)
    actors = physical.actors
    # The lead was spawned after the latest tick, so it has no simulated state yet.
    assert set(actors) == {"npc:0"}
    assert actors["npc:0"].velocity == (1.0, -2.0, 0.0) and actors["npc:0"].category == "vehicle"

    simulator_runtime.control_actor("npc:0", Control(throttle=0.4, steer=-0.1))
    npc.apply_control.assert_called_once_with({"throttle": 0.4, "brake": 0.0, "steer": -0.1})
    npc.set_autopilot.assert_called_once_with(False, 8000)
    simulator_runtime._recipe_replay.release.assert_called_once_with("npc:0")
    simulator_runtime.set_actor_state("lead", ActorState(np.eye(4), (0.0, 0.0, 0.0)))
    lead.set_transform.assert_called_once()
    with pytest.raises(ValueError, match="driving system"):
        simulator_runtime.control_actor("ego", Control())
    with pytest.raises(KeyError, match="Unknown CARLA actor"):
        simulator_runtime.control_actor("npc:5", Control())


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


@pytest.mark.parametrize("position,expected", [(0, True), (1, False), (2, False), (3, True), (4, True)])
def test_shared_deployment_boundary_predicate(position, expected):
    from avsectester.closed_loop.simulator import DeploymentPlan, can_deploy_at

    assert can_deploy_at(position, DeploymentPlan(2), 1) is expected


def test_deployment_boundary_predicate_rejects_invalid_positions():
    from avsectester.closed_loop.simulator import DeploymentPlan, can_deploy_at

    with pytest.raises(ValueError, match="position"):
        can_deploy_at(-1, DeploymentPlan(), 0)
