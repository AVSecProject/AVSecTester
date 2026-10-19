"""Hook configuration migration preserves processing order and lifecycle ownership."""

from dataclasses import replace

import pytest

from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.simulator import WorldBackend
from avsectester.backend import run
from avsectester.closed_loop.datatypes import Control, Observation, StateEstimate
from avsectester.closed_loop.hooks import Hook, HookSet, Plugin, Runtime
from avsectester.closed_loop.vehicle_io import VehicleIO


class World(WorldBackend):
    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1
        self.frame = 0
        return Observation(0.0, 0, ego_speed=2.0)

    def step(self, control):
        self.frame += 1
        return Observation(self.frame * 0.1, self.frame, ego_speed=2.0)


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
    run(World(), driver, frames=2, runtime=hooks)
    assert calls == ["reset", "sensors", "localize", "observation",
                     "sensors", "localize", "observation", "close"]
    assert driver.initial is driver.seen[0]
    assert [value.ego_speed for value in driver.seen] == [70, 70]


def test_legacy_runtime_clone_keeps_localizer_without_mutating_driving_io():
    localizer = lambda observation, context: StateEstimate(ego_speed=9)
    runtime = Runtime(localizer=localizer, seed=17, case={"name": "legacy"})
    copied = runtime.with_hooks([Hook("observation", lambda value, context: value)])
    driver = Driver()
    run(World(), driver, frames=1, runtime=copied)
    assert copied is not runtime and isinstance(copied, Runtime)
    assert copied.localizer is localizer
    assert copied.seed == 17 and copied.case == {"name": "legacy"}
    assert runtime.hooks == () and len(copied.hooks) == 1
    assert driver.vehicle_io.localizer is None
    assert driver.initial.ego_speed == 9


def test_conflicting_localizers_fail_before_world_reset():
    world = World()
    driver = Driver(VehicleIO(lambda value, context: StateEstimate()))
    with pytest.raises(ValueError, match="not both"):
        run(world, driver, frames=1, runtime=Runtime(localizer=lambda value, context: StateEstimate()))
    assert world.resets == 0


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
    run(World(), Driver(VehicleIO(shared)), frames=2,
        runtime=HookSet([Hook("observation", shared)]))
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
        raw = world.reset()
        world._remember_ground_truth(raw)
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

