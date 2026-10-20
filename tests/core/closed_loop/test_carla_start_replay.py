"""CARLA start replay consumes matching captures without moving the world's selected boundary."""

from collections import deque
from copy import deepcopy
from types import SimpleNamespace

import pytest

from avsectester.datasets.carla import CarlaDataset, CarlaRecipe, PrescribedMotion
from avsectester.closed_loop.datatypes import Control
from avsectester.closed_loop.simulators.carla import CarlaSimulator, SensorAlignmentError


def simulator_runtime_with_queue(values):
    queue = deque(values)
    simulator_runtime = CarlaSimulator({}).create_runtime()
    simulator_runtime.ego = SimpleNamespace(
        sensors={"front": SimpleNamespace(source_identifier="front", frame0=100)},
        sensor_data_manager=SimpleNamespace(
            has_data=lambda name: bool(queue), pop=lambda name: queue.popleft(),
        ),
    )
    return simulator_runtime, queue


def test_drain_stale_entries_and_keep_current_payload():
    old, current = SimpleNamespace(frame=2), SimpleNamespace(frame=3)
    simulator_runtime, queue = simulator_runtime_with_queue([old, current])
    assert simulator_runtime._aligned_sensor_data(103) == {"front": current}
    assert not queue


def test_future_capture_is_a_clock_error():
    simulator_runtime, _ = simulator_runtime_with_queue([SimpleNamespace(frame=4)])
    with pytest.raises(SensorAlignmentError, match="ahead") as error:
        simulator_runtime._aligned_sensor_data(103)
    assert error.value.report["observed_sensor_frames"] == {"front": 104}


def test_initial_capture_is_reused_without_a_second_queue_pop():
    current = SimpleNamespace(frame=3)
    simulator_runtime, queue = simulator_runtime_with_queue([])
    assert simulator_runtime._aligned_sensor_data(103, initial={"front": current}) == {"front": current}
    assert not queue


def test_missing_aligned_capture_fails_with_last_seen_frame(monkeypatch):
    simulator_runtime, _ = simulator_runtime_with_queue([SimpleNamespace(frame=2)])
    ticks = iter([0, 2])
    monkeypatch.setattr("avsectester.closed_loop.simulators.carla.time.monotonic", lambda: next(ticks))
    with pytest.raises(SensorAlignmentError, match="missing aligned") as error:
        simulator_runtime._aligned_sensor_data(103)
    assert error.value.report["expected_native_frame"] == 103
    assert error.value.report["observed_sensor_frames"] == {"front": 102}


def recipe_source(client, steps=2, time_step=None):
    forbidden = {"connect_ip", "connect_port"} & client.keys()
    if forbidden:
        raise ValueError(f"Connection options belong to CarlaSimulator: {sorted(forbidden)}")
    time_step = time_step if time_step is not None else 1 / client.get("rate", 10)
    scene_settings = {key: value for key, value in client.items() if key == "map_name"}
    scene_settings.setdefault("map_name", "Town03")
    recipe = CarlaRecipe({"client": scene_settings, "ego": {}}, steps, time_step,
                         {"ego": PrescribedMotion((Control(),) * steps)})
    data = CarlaDataset({"s": recipe})
    return data.scenario_start("s", data.frames("s")[0])


def test_source_spawn_resolution_connects_to_the_factory_server(monkeypatch):
    seen = []

    def prepare(scenario):
        seen.append(deepcopy(scenario["client"]))
        return scenario

    monkeypatch.setattr("avsectester.closed_loop.simulators.carla.prepare_scenario", prepare)
    source = recipe_source({"rate": 10})
    simulator_runtime = CarlaSimulator({"rate": 10, "connect_ip": "10.0.0.5", "connect_port": 3000}).create_runtime()
    resolved = simulator_runtime._resolve_scenario(source, None, {})
    assert [(c["connect_ip"], c["connect_port"]) for c in seen] == [("10.0.0.5", 3000)]
    assert (resolved["client"]["connect_ip"], resolved["client"]["connect_port"]) == ("10.0.0.5", 3000)
    # Preparation evidence can contain the previous server; runtime settings still win.
    from avsectester.closed_loop.frame_data import Initialization
    from avsectester.datasets.base import FrameData
    evidence = Initialization(FrameData(source.frame),
        {**resolved, "client": {**resolved["client"], "connect_ip": "127.0.0.1", "connect_port": 2000}})
    replayed = simulator_runtime._resolve_scenario(source, evidence, {})
    assert replayed["client"]["connect_ip"] == "10.0.0.5"
    assert replayed["client"]["connect_port"] == 3000


def test_recipe_time_step_is_compared_like_recipe_validation():
    source = recipe_source({"rate": 30}, steps=1, time_step=0.0333333333)
    CarlaSimulator({"rate": 30}).create_runtime().validate_start(source)


def test_runtime_and_recipe_settings_are_separate_and_world_override_wins(monkeypatch):
    monkeypatch.setattr("avsectester.closed_loop.simulators.carla.prepare_scenario", lambda scenario: scenario)
    simulator_runtime = CarlaSimulator({"rate": 10, "seed": 9}).create_runtime(seed=10)
    source = recipe_source({}, time_step=.1)
    resolved = simulator_runtime._resolve_scenario(source, None, {"weather": "HardRainNoon"})
    assert resolved["client"]["map_name"] == "Town03"
    assert resolved["client"]["rate"] == 10
    assert resolved["client"]["seed"] == 10
    assert resolved["weather"] == "HardRainNoon"
    with pytest.raises(ValueError, match="ScenarioStart"):
        simulator_runtime.reset()


@pytest.mark.parametrize("client", [{"scenario": {}}, {"map_name": "Town03"}, {"rate": 0},
                                    {"rate": True}, {"synchronous": False}])
def test_carla_rejects_scene_settings_and_invalid_runtime_cadence(client):
    with pytest.raises(ValueError):
        CarlaSimulator(client)


def test_recipe_helper_rejects_connection_options_instead_of_discarding_them():
    with pytest.raises(ValueError, match="Connection options belong to CarlaSimulator"):
        recipe_source({"connect_ip": "127.0.0.1", "connect_port": 2000})
