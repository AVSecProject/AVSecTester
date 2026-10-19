"""Selected-boundary deployment, replay isolation and reset state ownership."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.attack.runtime import AttackRuntime, ResultSubscription
from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.simulator import DeploymentPlan
from avsectester.closed_loop.runner import ClosedLoopRunner, RunRequest
from avsectester.datasets.base import ActorState, FrameData
from avsectester.datasets.carla import CarlaDataset, CarlaRecipe, PrescribedMotion
from avsectester.closed_loop.datatypes import Control, Observation, WorldSnapshot
from avsectester.closed_loop.frame_data import Initialization
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.closed_loop.simulators.carla import CarlaSimulator, CarlaSimulatorRuntime
from avsectester.closed_loop.simulators.nurec import NuRecSimulator, StubRenderer


class FakeCarla(CarlaSimulatorRuntime):
    """Fake only native operations; use the real capture lifecycle."""
    def __init__(self):
        super().__init__(CarlaSimulator({"client": {"rate": 10}, "ego": {"x": 42}, "lead": {"gap": 9}}))
        self.events = []

    def _initialize_world(self, source, initialization, environment):
        config = self._resolve_scenario(source, initialization, environment)
        self.replay_scenario = config
        self.x = config["ego"]["x"]
        self.position = 0
        self.base_origin = self.x
        self.rendered = self.x  # what the sensors last captured; changes only on a tick
        self.inserted = []
        self._base_actor_ids = {1}
        self.client = SimpleNamespace(world=SimpleNamespace(get_actors=lambda: [SimpleNamespace(id=1), *self.inserted]))
        self._recipe = source.payload["recipe"] if source else None
        self._recipe_replay = SimpleNamespace(release_ego=lambda: None)
        self.events.append("base")

    @property
    def native(self):
        return {"actors": {"ego": SimpleNamespace(id=1)}}

    def _replay_tick(self):
        self.position += 1
        self.x += 1
        self.rendered = self.x
        self.events.append(("replay", self.position))

    def _hold_tick(self):
        self.rendered = self.x
        self.events.append(("hold", self.position))

    def _before_selected_capture(self, position):
        pass

    def _advance_native(self, control):
        self._replay_tick()

    def _observe(self):
        self.events.append(("capture", self.position))
        return Observation(self.position*.1, self.position, ego_speed=0, vehicle_state=self.rendered)

    def _state(self):
        return WorldSnapshot(self.position*.1, self.position, self.x, 0)

    def _capture_frame_data(self, frame):
        pose = np.eye(4)
        pose[0, 3] = self.x
        return FrameData(frame, None, {"ego": ActorState(pose, (0, 0, 0))},
                              provenance={"actual_scenario_time": self.position*.1})

    def close(self):
        pass


class Driver(AVStack):
    def __call__(self, obs):
        return Control()


def inputs(position, k):
    scenario = {"client": {"rate": 10, "synchronous": True}, "ego": {"x": 0}}
    recipe = CarlaRecipe(scenario, 8, .1, {"ego": PrescribedMotion((Control(),)*8)})
    data = CarlaDataset({"s": recipe})
    frame = data.frames("s")[position]
    pose = np.eye(4)
    pose[0, 3] = position
    expected = FrameData(frame, None, {"ego": ActorState(pose, (0, 0, 0))},
                              provenance={"actual_scenario_time": position*.1})
    init = Initialization(expected, resolved_scenario=scenario, provenance={"deployment_ticks": k})
    return data.scenario_start("s", frame), init


@pytest.mark.parametrize("position,k", [(5, 2), (5, 0), (0, 2)])
def test_pre_roll_is_hook_free_and_deployment_preserves_boundary(position, k):
    source, initialization = inputs(position, k)
    clean, attacked = FakeCarla(), FakeCarla()
    counts = []
    def setup(value, context):
        backend = context.simulator_runtime
        # The window is k participant ticks plus CARLA's capture tick, carved out of pre-roll.
        assert backend.position == max(0, position-k-1)
        backend.events.append("setup")
        backend.inserted.append(SimpleNamespace(id=99))
        if k:
            backend.setup_tick()  # participant tick is included in the shared budget
    def step(value, context):
        counts.append(context.stage)
    starts = []
    class Attack(AttackRuntime):
        subscriptions = (ResultSubscription("physical_state"),)
        def setup(self, context):
            # Setup runs inside the deployment window, after simulator.setup hooks, before capture.
            assert not any(isinstance(e, tuple) and e[0] == "capture" for e in attacked.events)
            assert context.frame == attacked.position
            starts.append(context.frame)
            return ()
    for backend, hooks, attacks in (
        (clean, HookSet(), ()),
        (attacked, HookSet((Hook("simulator.setup", setup), Hook("simulator.step.pre", step),
                           Hook("simulator.step.post", step))), (Attack(),)),
    ):
        ClosedLoopRunner().run(backend, Driver(), RunRequest(
            0, start=source, initialization=initialization, deployment=DeploymentPlan(k),
            hooks=hooks, attacks=attacks))
    assert counts == [] and len(starts) == 1 and starts[0] <= position
    assert clean.position == attacked.position == position
    assert clean.x == attacked.x == position
    assert [e for e in clean.events if isinstance(e, tuple)] == [e for e in attacked.events if isinstance(e, tuple)]
    assert attacked.initialization_report["inserted_actor_ids"] == [99]


def test_rejects_short_boundary_before_world_creation_and_tick_budget_overrun():
    backend = FakeCarla()
    source, initialization = inputs(1, 2)
    with pytest.raises(ValueError, match="deployment window"):
        backend.reset(source, initialization, deployment=DeploymentPlan(2))
    assert backend.events == []
    source, initialization = inputs(3, 1)
    def overrun(value, context):
        context.simulator_runtime.setup_tick()
        context.simulator_runtime.setup_tick()
    with pytest.raises(RuntimeError, match="budget"):
        ClosedLoopRunner().run(backend, Driver(), RunRequest(0, start=source,
            initialization=initialization, deployment=DeploymentPlan(1),
            hooks=HookSet((Hook("simulator.setup", overrun),))))


def test_fake_carla_source_then_plain_reset_uses_configured_origin():
    backend = FakeCarla()
    config = deepcopy(backend.scenario)
    source, initialization = inputs(4, 0)
    backend.reset(source, initialization)
    assert backend.observe().vehicle_state == 4
    backend.reset()
    assert backend.observe().vehicle_state == 42
    assert backend.scenario == config and backend.lead_cfg == config["lead"]
    assert backend.initialization_report == {}


def test_nurec_source_then_plain_reset_restores_origin_bounds_and_renderer_defaults():
    from test_scenario_start_initialization import nurec_source

    renderer = StubRenderer(height=2, width=2)
    renderer.start_transform = np.eye(4)
    renderer.start_timestamp_us = 7000000
    renderer.recorded_time_bounds = (7, 9)
    backend = NuRecSimulator({"scene": "default", "ego0": {"x": 2}, "dt": .05}, renderer).create_runtime()
    original = deepcopy(backend.config)
    backend.reset(nurec_source())
    assert backend.observe().vehicle_state.x == 12
    assert backend.renderer.origin_timestamp_us == 100500000
    backend.reset()
    assert backend.observe().vehicle_state.x == 2
    assert backend._recorded_end is None and backend.initialization_report == {}
    assert backend.renderer.origin_timestamp_us == backend.renderer.start_timestamp_us == 7000000
    assert backend.renderer.origin_bounds == backend.renderer.recorded_time_bounds == (7, 9)
    np.testing.assert_array_equal(backend.renderer.origin_transform, backend.renderer.start_transform)
    assert backend.config == original


def test_restore_captures_before_observe_without_render_hook_reentry():
    backend = NuRecSimulator({}, StubRenderer(height=2, width=2)).create_runtime()
    calls = []
    original = backend._emit
    def emit(stage, value, sensor=None):
        calls.append(stage)
        return original(stage, value, sensor)
    backend._emit = emit
    backend.reset()
    checkpoint = backend.checkpoint()
    backend.restore(checkpoint)
    assert calls.count("render.pre") == 2
    count = len(calls)
    backend.observe()
    backend.observe()
    assert len(calls) == count


@pytest.mark.parametrize("position", [None, 0, 5])
def test_zero_tick_setup_change_is_in_first_capture_without_moving_boundary(position):
    backend = FakeCarla()
    def move(value, context):
        context.simulator_runtime.events.append("setup")
        context.simulator_runtime.x += 100  # a simulator.setup change that declares no deployment ticks
    source, initialization = inputs(position, 0) if position is not None else (None, None)
    ClosedLoopRunner().run(backend, Driver(), RunRequest(
        0, start=source, initialization=initialization, deployment=DeploymentPlan(0),
        hooks=HookSet((Hook("simulator.setup", move),))))
    expected_position = position or 0
    assert backend.position == expected_position
    assert backend.observe().frame == expected_position
    assert backend.observe().vehicle_state == backend.x  # the capture shows the setup change
    capture = backend.events.index(("capture", expected_position))
    assert backend.events.index("setup") < capture - 1  # one capture tick in between


def test_plain_reset_ticks_world_setup_deployment_then_capture():
    backend = FakeCarla()
    def deploy(value, context):
        context.simulator_runtime.events.append("setup")
        context.simulator_runtime.setup_tick()
        context.simulator_runtime.setup_tick()
    ClosedLoopRunner().run(backend, Driver(), RunRequest(
        0, deployment=DeploymentPlan(2), hooks=HookSet((Hook("simulator.setup", deploy),))))
    assert backend.events == ["base", "setup", ("hold", 0), ("hold", 0), ("hold", 0), ("capture", 0)]


def test_capture_tick_counts_toward_the_deployment_window():
    backend = FakeCarla()
    source, initialization = inputs(1, 1)
    with pytest.raises(ValueError, match="deployment window"):
        backend.reset(source, initialization, deployment=DeploymentPlan(1))
    assert backend.events == []


def test_runs_release_the_ego_but_frame_data_generation_keeps_it_prescribed(monkeypatch):
    class Recording(FakeCarla):
        def _initialize_world(self, *args, **kwargs):
            super()._initialize_world(*args, **kwargs)
            self._recipe_replay = SimpleNamespace(release_ego=lambda: self.events.append("release"))

    # Without evidence, a p=0 reset resolves spawns itself; no CARLA server exists here.
    monkeypatch.setattr("avsectester.closed_loop.simulators.carla.prepare_scenario", lambda scenario: scenario)
    source, _ = inputs(0, 0)
    run = Recording()
    run.reset(source)  # p=0 needs no materialized evidence, but the run still drives the ego
    assert "release" in run.events
    materialized = Recording()
    materialized.generate_frame_data(source, (source.frame,))
    assert "release" not in materialized.events
    assert not materialized._generating_frame_data
