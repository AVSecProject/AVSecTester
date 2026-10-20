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
        super().__init__(CarlaSimulator({"rate": 10}))
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
        return FrameData(frame, ego=ActorState(pose, (0, 0, 0)),
                              provenance={"actual_scenario_time": self.position*.1})

    def close(self):
        pass


class Driver(AVStack):
    def __call__(self, obs):
        return Control()


def inputs(position, k):
    scenario = {"client": {"map_name": "Town03"}, "ego": {"x": 0}}
    recipe = CarlaRecipe(scenario, 8, .1, {"ego": PrescribedMotion((Control(),)*8)})
    data = CarlaDataset({"s": recipe})
    frame = data.frames("s")[position]
    pose = np.eye(4)
    pose[0, 3] = position
    expected = FrameData(frame, ego=ActorState(pose, (0, 0, 0)),
                              provenance={"actual_scenario_time": position*.1})
    init = Initialization(expected, resolved_scenario=scenario, provenance={"deployment_ticks": k})
    return data.scenario_start("s", frame), init


@pytest.mark.parametrize("position,k", [(5, 2), (5, 0), (0, 2)])
def test_pre_roll_is_hook_free_and_deployment_preserves_boundary(position, k):
    source, initialization = inputs(position, k)
    clean, attacked = FakeCarla(), FakeCarla()
    counts = []
    def setup(value, context):
        simulator_runtime = context.simulator_runtime
        # The window is k participant ticks plus CARLA's capture tick, carved out of pre-roll.
        assert simulator_runtime.position == max(0, position-k-1)
        simulator_runtime.events.append("setup")
        simulator_runtime.inserted.append(SimpleNamespace(id=99))
        if k:
            simulator_runtime.setup_tick()  # participant tick is included in the shared budget
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
    for simulator_runtime, hooks, attacks in (
        (clean, HookSet(), ()),
        (attacked, HookSet((Hook("simulator.setup", setup), Hook("simulator.step.pre", step),
                           Hook("simulator.step.post", step))), (Attack(),)),
    ):
        ClosedLoopRunner().run(simulator_runtime, Driver(), RunRequest(
            0, start=source, initialization=initialization, deployment=DeploymentPlan(k),
            hooks=hooks, attacks=attacks))
    assert counts == [] and len(starts) == 1 and starts[0] <= position
    assert clean.position == attacked.position == position
    assert clean.x == attacked.x == position
    assert [e for e in clean.events if isinstance(e, tuple)] == [e for e in attacked.events if isinstance(e, tuple)]
    assert attacked.initialization_report["inserted_actor_ids"] == [99]


def test_rejects_short_boundary_before_world_creation_and_tick_budget_overrun():
    simulator_runtime = FakeCarla()
    source, initialization = inputs(1, 2)
    with pytest.raises(ValueError, match="deployment window"):
        simulator_runtime.reset(source, initialization, deployment=DeploymentPlan(2))
    assert simulator_runtime.events == []
    source, initialization = inputs(3, 1)
    def overrun(value, context):
        context.simulator_runtime.setup_tick()
        context.simulator_runtime.setup_tick()
    with pytest.raises(RuntimeError, match="budget"):
        ClosedLoopRunner().run(simulator_runtime, Driver(), RunRequest(0, start=source,
            initialization=initialization, deployment=DeploymentPlan(1),
            hooks=HookSet((Hook("simulator.setup", overrun),))))


def test_carla_rejects_plain_reset_without_reusing_previous_source():
    simulator_runtime = FakeCarla()
    source, initialization = inputs(4, 0)
    simulator_runtime.reset(source, initialization)
    assert simulator_runtime.observe().vehicle_state == 4
    with pytest.raises(ValueError, match="ScenarioStart"):
        simulator_runtime.reset()


def test_nurec_source_then_plain_reset_restores_origin_bounds_and_renderer_defaults():
    from test_scenario_start_initialization import nurec_source

    renderer = StubRenderer(height=2, width=2)
    simulator_runtime = NuRecSimulator({"scene": "default", "ego0": {"x": 2}, "dt": .05}, renderer).create_runtime()
    original = deepcopy(simulator_runtime.config)
    simulator_runtime.reset(nurec_source())
    assert simulator_runtime.observe().vehicle_state.x == 12
    assert simulator_runtime.renderer.origin_timestamp_us == 100500000
    simulator_runtime.reset()
    assert simulator_runtime.observe().vehicle_state.x == 2
    assert simulator_runtime._recorded_end is None and simulator_runtime.initialization_report == {}
    assert simulator_runtime.renderer.origin_timestamp_us is None
    assert simulator_runtime.renderer.origin_bounds is None
    assert simulator_runtime.renderer.origin_transform is None
    assert simulator_runtime.config == original


def test_restore_captures_before_observe_without_render_hook_reentry():
    simulator_runtime = NuRecSimulator({}, StubRenderer(height=2, width=2)).create_runtime()
    calls = []
    original = simulator_runtime._emit
    def emit(stage, value, sensor=None):
        calls.append(stage)
        return original(stage, value, sensor)
    simulator_runtime._emit = emit
    simulator_runtime.reset()
    checkpoint = simulator_runtime.checkpoint()
    simulator_runtime.restore(checkpoint)
    assert calls.count("render.pre") == 2
    count = len(calls)
    simulator_runtime.observe()
    simulator_runtime.observe()
    assert len(calls) == count


@pytest.mark.parametrize("position", [0, 5])
def test_zero_tick_setup_change_is_in_first_capture_without_moving_boundary(position):
    simulator_runtime = FakeCarla()
    def move(value, context):
        context.simulator_runtime.events.append("setup")
        context.simulator_runtime.x += 100  # a simulator.setup change that declares no deployment ticks
    source, initialization = inputs(position, 0)
    ClosedLoopRunner().run(simulator_runtime, Driver(), RunRequest(
        0, start=source, initialization=initialization, deployment=DeploymentPlan(0),
        hooks=HookSet((Hook("simulator.setup", move),))))
    expected_position = position or 0
    assert simulator_runtime.position == expected_position
    assert simulator_runtime.observe().frame == expected_position
    assert simulator_runtime.observe().vehicle_state == simulator_runtime.x  # the capture shows the setup change
    capture = simulator_runtime.events.index(("capture", expected_position))
    assert simulator_runtime.events.index("setup") < capture - 1  # one capture tick in between


def test_origin_reset_ticks_world_setup_deployment_then_capture():
    simulator_runtime = FakeCarla()
    def deploy(value, context):
        context.simulator_runtime.events.append("setup")
        context.simulator_runtime.setup_tick()
        context.simulator_runtime.setup_tick()
    source, initialization = inputs(0, 2)
    ClosedLoopRunner().run(simulator_runtime, Driver(), RunRequest(
        0, start=source, initialization=initialization, deployment=DeploymentPlan(2),
        hooks=HookSet((Hook("simulator.setup", deploy),))))
    assert simulator_runtime.events == ["base", "setup", ("hold", 0), ("hold", 0), ("hold", 0), ("capture", 0)]


def test_capture_tick_counts_toward_the_deployment_window():
    simulator_runtime = FakeCarla()
    source, initialization = inputs(1, 1)
    with pytest.raises(ValueError, match="deployment window"):
        simulator_runtime.reset(source, initialization, deployment=DeploymentPlan(1))
    assert simulator_runtime.events == []


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
