"""Selected-boundary deployment, replay isolation and reset state ownership."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.attack_runtime import AttackRuntime, ResultSubscription
from avsectester.backend import AVStack
from avsectester.capture import DeploymentPlan
from avsectester.closed_loop import ClosedLoopRunner, RunRequest
from avsectester.datasets.base import ActorState, ReferenceFrame
from avsectester.datasets.carla import CarlaDataset, CarlaRecipe, PrescribedMotion
from avsectester.plane import Control, Observation, WorldSnapshot
from avsectester.reference import Initialization
from avsectester.runtime import Hook, HookSet
from avsectester.simulators.carla import CarlaBackend
from avsectester.simulators.nurec import NuRecBackend, StubRenderer


class FakeCarla(CarlaBackend):
    """Fake only native operations; use the real capture lifecycle."""
    def __init__(self):
        super().__init__({"client": {"rate": 10}, "ego": {"x": 42}, "lead": {"gap": 9}})
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

    def ground_truth(self):
        return WorldSnapshot(self.position*.1, self.position, self.x, 0)

    def capture_reference(self, source_frame):
        pose = np.eye(4)
        pose[0, 3] = self.x
        return ReferenceFrame(source_frame, None, {"ego": ActorState(pose, (0, 0, 0))},
                              provenance={"actual_source_time": self.position*.1})

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
    expected = ReferenceFrame(frame, None, {"ego": ActorState(pose, (0, 0, 0))},
                              provenance={"actual_source_time": position*.1})
    init = Initialization(expected, resolved_scenario=scenario, provenance={"deployment_ticks": k})
    return data.source_data("s", frame), init


@pytest.mark.parametrize("position,k", [(5, 2), (5, 0), (0, 2)])
def test_pre_roll_is_hook_free_and_deployment_preserves_boundary(position, k):
    source, initialization = inputs(position, k)
    clean, attacked = FakeCarla(), FakeCarla()
    counts = []
    def setup(value, context):
        backend = context.backend
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
        reads = (ResultSubscription("physical_state"),)
        def start(self, context):
            assert attacked.events[-1] == ("capture", position)
            assert context.physical_state.vehicle_state == position
            starts.append(context.frame)
            return ()
    for backend, hooks, attacks in (
        (clean, HookSet(), ()),
        (attacked, HookSet((Hook("world.setup", setup), Hook("world.step.pre", step),
                           Hook("world.step.post", step))), (Attack(),)),
    ):
        ClosedLoopRunner().run(backend, Driver(), RunRequest(
            0, source=source, initialization=initialization, deployment=DeploymentPlan(k),
            hooks=hooks, attacks=attacks))
    assert counts == [] and starts == [position]
    assert clean.position == attacked.position == position
    assert clean.x == attacked.x == position
    assert [e for e in clean.events if isinstance(e, tuple)] == [e for e in attacked.events if isinstance(e, tuple)]
    assert attacked.initialization_report["inserted_actor_ids"] == [99]


def test_rejects_short_boundary_before_world_creation_and_tick_budget_overrun():
    backend = FakeCarla()
    source, initialization = inputs(1, 2)
    with pytest.raises(ValueError, match="deployment window"):
        backend.reset_world(source, initialization, deployment=DeploymentPlan(2))
    assert backend.events == []
    source, initialization = inputs(3, 1)
    def overrun(value, context):
        context.backend.setup_tick()
        context.backend.setup_tick()
    with pytest.raises(RuntimeError, match="budget"):
        ClosedLoopRunner().run(backend, Driver(), RunRequest(0, source=source,
            initialization=initialization, deployment=DeploymentPlan(1),
            hooks=HookSet((Hook("world.setup", overrun),))))


def test_fake_carla_source_then_plain_reset_uses_configured_origin():
    backend = FakeCarla()
    config = deepcopy(backend.scenario)
    source, initialization = inputs(4, 0)
    backend.reset_world(source, initialization)
    assert backend.observe().vehicle_state == 4
    backend.reset()
    assert backend.observe().vehicle_state == 42
    assert backend.scenario == config and backend.lead_cfg == config["lead"]
    assert backend.initialization_report == {}


def test_nurec_source_then_plain_reset_restores_origin_bounds_and_renderer_defaults():
    from test_source_initialization import nurec_source

    renderer = StubRenderer(height=2, width=2)
    renderer.start_transform = np.eye(4)
    renderer.start_timestamp_us = 7000000
    renderer.source_time_bounds = (7, 9)
    backend = NuRecBackend({"scene": "default", "ego0": {"x": 2}, "dt": .05}, renderer)
    original = deepcopy(backend.config)
    backend.reset_world(nurec_source())
    assert backend.observe().vehicle_state.x == 12
    assert renderer.origin_timestamp_us == 100500000
    backend.reset()
    assert backend.observe().vehicle_state.x == 2
    assert backend._source_end is None and backend.initialization_report == {}
    assert renderer.origin_timestamp_us == renderer.start_timestamp_us == 7000000
    assert renderer.origin_bounds == renderer.source_time_bounds == (7, 9)
    np.testing.assert_array_equal(renderer.origin_transform, renderer.start_transform)
    assert backend.config == original


def test_restore_captures_before_observe_without_render_hook_reentry():
    backend = NuRecBackend({}, StubRenderer(height=2, width=2))
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
        context.backend.events.append("setup")
        context.backend.x += 100  # a world.setup change that declares no deployment ticks
    source, initialization = inputs(position, 0) if position is not None else (None, None)
    ClosedLoopRunner().run(backend, Driver(), RunRequest(
        0, source=source, initialization=initialization, deployment=DeploymentPlan(0),
        hooks=HookSet((Hook("world.setup", move),))))
    expected_position = position or 0
    assert backend.position == expected_position
    assert backend.observe().frame == expected_position
    assert backend.observe().vehicle_state == backend.x  # the capture shows the setup change
    capture = backend.events.index(("capture", expected_position))
    assert backend.events.index("setup") < capture - 1  # one capture tick in between


def test_plain_reset_ticks_world_setup_deployment_then_capture():
    backend = FakeCarla()
    def deploy(value, context):
        context.backend.events.append("setup")
        context.backend.setup_tick()
        context.backend.setup_tick()
    ClosedLoopRunner().run(backend, Driver(), RunRequest(
        0, deployment=DeploymentPlan(2), hooks=HookSet((Hook("world.setup", deploy),))))
    assert backend.events == ["base", "setup", ("hold", 0), ("hold", 0), ("hold", 0), ("capture", 0)]


def test_capture_tick_counts_toward_the_deployment_window():
    backend = FakeCarla()
    source, initialization = inputs(1, 1)
    with pytest.raises(ValueError, match="deployment window"):
        backend.reset_world(source, initialization, deployment=DeploymentPlan(1))
    assert backend.events == []


def test_runs_release_the_ego_but_reference_materialization_keeps_it_prescribed(monkeypatch):
    class Recording(FakeCarla):
        def _initialize_world(self, *args, **kwargs):
            super()._initialize_world(*args, **kwargs)
            self._recipe_replay = SimpleNamespace(release_ego=lambda: self.events.append("release"))

    # Without evidence, a p=0 reset resolves spawns itself; no CARLA server exists here.
    monkeypatch.setattr("avsectester.simulators.carla.prepare_scenario", lambda scenario: scenario)
    source, _ = inputs(0, 0)
    run = Recording()
    run.reset_world(source)  # p=0 needs no materialized evidence, but the run still drives the ego
    assert "release" in run.events
    materialized = Recording()
    materialized.materialize_reference(source, (source.start,))
    assert "release" not in materialized.events
    assert not materialized._materializing
