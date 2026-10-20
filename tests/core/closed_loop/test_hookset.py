"""Hook configuration migration preserves processing order and lifecycle ownership."""

from dataclasses import replace

import pytest

from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.simulator import Simulator, SimulatorRuntime
from avsectester.closed_loop.runner import ClosedLoopRunner, RunRequest
from avsectester.closed_loop.datatypes import Control, Observation, StateEstimate, WorldSnapshot
from avsectester.closed_loop.hooks import Hook, HookSet, Plugin
from avsectester.closed_loop.vehicle_io import VehicleIO


class Config(Simulator):
    def create_runtime(self, seed=None):
        return World()


class World(SimulatorRuntime):
    def __init__(self):
        super().__init__(Config())
        self.resets = 0

    def _initialize_world(self, start, initialization, environment):
        self.resets += 1
        self.frame = 0

    def _state(self):
        return WorldSnapshot(self.frame * .1, self.frame, ego_speed=2.0)

    def _observe(self):
        return Observation(self.frame * .1, self.frame, ego_speed=2.0)

    def _advance_native(self, control):
        self.frame += 1


class Driver(AVStack):
    def __init__(self, vehicle_io=None):
        self._io = vehicle_io or VehicleIO()
        self.seen = []

    @property
    def vehicle_io(self):
        return self._io

    def reset(self, observation):
        self.initial = observation

    def __call__(self, observation):
        self.seen.append(observation)
        return Control()


def test_vehicle_io_localizer_lifecycle_and_first_input_are_not_duplicated():
    calls = []

    class Localizer(Plugin):
        def reset(self, context):
            calls.append("reset")

        def transform(self, observation, context):
            calls.append("localize")
            return StateEstimate(ego_speed=observation.ego_speed + 3)

        def close(self):
            calls.append("close")

    def sensors(observation, context):
        calls.append("sensors")
        return replace(observation, ego_speed=observation.ego_speed * 2)

    def observe(observation, context):
        calls.append("observation")
        return replace(observation, ego_speed=observation.ego_speed * 10)

    driver = Driver(VehicleIO(Localizer()))
    hooks = HookSet([Hook("sensors.post", sensors), Hook("observation", observe)])
    ClosedLoopRunner().run(World(), driver, RunRequest(2, hooks=hooks))
    assert calls == ["reset", "sensors", "localize", "observation",
                     "sensors", "localize", "observation", "close"]
    assert driver.initial is driver.seen[0]
    assert [value.ego_speed for value in driver.seen] == [70, 70]


def test_localizer_shared_with_hook_has_one_lifecycle():
    class Shared(Plugin):
        def __init__(self):
            self.resets = self.closes = 0

        def reset(self, context):
            self.resets += 1

        def transform(self, value, context):
            return StateEstimate(ego_speed=4) if context.stage == "localization" else value

        def close(self):
            self.closes += 1

    shared = Shared()
    ClosedLoopRunner().run(World(), Driver(VehicleIO(shared)), RunRequest(
        2, hooks=HookSet([Hook("observation", shared)])))
    assert shared.resets == shared.closes == 1


def test_vehicle_io_rejects_invalid_localizer():
    with pytest.raises(TypeError, match="localizer must be callable"):
        VehicleIO(object())


def test_explicit_output_drain_releases_history_and_preserves_public_list_identity():
    def record(observation, context):
        context.record_output("input.speed", observation.ego_speed)
        return observation

    world, driver = World(), Driver()
    session = HookSet([Hook("observation", record)]).session(world, driver)
    public_outputs = session.outputs
    with session.activate():
        world.reset()
        raw = world.observe()
        session.observe(raw)
        session.observe(raw)
        assert len(public_outputs) == 2
        first = session.drain_outputs()
        assert session.outputs is public_outputs and public_outputs == []
        session.step = 1
        session.observe(raw)
        second = session.drain_outputs()
    assert [record.step for record in first] == [0, 0]
    assert [record.step for record in second] == [1]
    assert session.drain_outputs() == []


def test_hook_clone_preserves_seed_without_mutating_original():
    hooks = HookSet(seed=17)
    copied = hooks.with_hooks([Hook("observation", lambda value, context: value)])
    assert copied is not hooks and isinstance(copied, HookSet)
    assert copied.seed == 17
    assert hooks.hooks == () and len(copied.hooks) == 1


@pytest.mark.parametrize('stage', ['sensors.post', 'observation', 'command'])
def test_invalid_stage_output_stops_the_remaining_handler_chain(stage):
    following = []
    hooks = HookSet([
        Hook(stage, lambda value, context: object()),
        Hook(stage, lambda value, context: following.append(value)),
    ])
    with pytest.raises(TypeError, match='must return'):
        ClosedLoopRunner().run(World(), Driver(), RunRequest(1, hooks=hooks))
    assert following == []


@pytest.mark.parametrize('stage', ['simulator.setup', 'simulator.step.pre', 'simulator.step.post'])
def test_simulator_handlers_must_return_none_and_stop_the_chain(stage):
    following = []
    hooks = HookSet([
        Hook(stage, lambda value, context: Control()),
        Hook(stage, lambda value, context: following.append(value)),
    ])
    with pytest.raises(TypeError, match='must return None'):
        ClosedLoopRunner().run(World(), Driver(), RunRequest(1, hooks=hooks))
    assert following == []


def test_observation_context_tracks_previous_handler_replacements():
    seen = []
    def first(observation, context):
        return replace(observation, ego_speed=9.)
    def second(observation, context):
        seen.append((observation.ego_speed, context.observation.ego_speed))
        return observation
    ClosedLoopRunner().run(World(), Driver(), RunRequest(
        1, hooks=HookSet([Hook('observation', first), Hook('observation', second)])))
    assert seen == [(9., 9.)]


def test_after_step_has_next_raw_observation_clock_and_independent_participant_copies():
    received = []
    class Feedback(Plugin):
        def transform(self, observation, context):
            return replace(observation, ego_speed=99.)
        def after_step(self, context):
            received.append((context.observation.frame, context.observation.t,
                             context.observation.ego_speed, context.sim_time))
            context.observation.ego_speed = -1
            context.control.brake = 1
    driver = Driver()
    log = ClosedLoopRunner().run(World(), driver, RunRequest(2, hooks=HookSet([
        Hook('observation', Feedback()), Hook('observation', Feedback()),
    ])))
    assert received == [(1, .1, 2., .1), (1, .1, 2., .1),
                        (2, .2, 2., .2), (2, .2, 2., .2)]
    assert all(row.executed_command.brake == 0 for row in log.steps)


def test_simulator_post_step_cannot_change_the_already_executed_command():
    def mutate(value, context):
        context.control.brake = 1
    log = ClosedLoopRunner().run(World(), Driver(), RunRequest(
        1, hooks=HookSet([Hook('simulator.step.post', mutate)])))
    assert log.steps[0].executed_command.brake == 0
