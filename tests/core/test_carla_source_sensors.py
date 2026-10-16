"""Source replay consumes matching captures without moving the world's selected boundary."""

from collections import deque
from copy import deepcopy
from types import SimpleNamespace

import pytest

from avsectester.datasets.carla import CarlaDataset, CarlaRecipe, PrescribedMotion
from avsectester.plane import Control
from avsectester.simulators.carla import CarlaBackend, SensorAlignmentError


def backend_with_queue(values):
    queue = deque(values)
    backend = CarlaBackend({"client": {}, "ego": {}})
    backend.ego = SimpleNamespace(
        sensors={"front": SimpleNamespace(source_identifier="front", frame0=100)},
        sensor_data_manager=SimpleNamespace(
            has_data=lambda name: bool(queue), pop=lambda name: queue.popleft(),
        ),
    )
    return backend, queue


def test_drain_stale_entries_and_keep_current_payload():
    old, current = SimpleNamespace(frame=2), SimpleNamespace(frame=3)
    backend, queue = backend_with_queue([old, current])
    assert backend._source_sensor_data(103) == {"front": current}
    assert not queue


def test_future_capture_is_a_clock_error():
    backend, _ = backend_with_queue([SimpleNamespace(frame=4)])
    with pytest.raises(SensorAlignmentError, match="ahead") as error:
        backend._source_sensor_data(103)
    assert error.value.report["observed_sensor_frames"] == {"front": 104}


def test_initial_capture_is_reused_without_a_second_queue_pop():
    current = SimpleNamespace(frame=3)
    backend, queue = backend_with_queue([])
    assert backend._source_sensor_data(103, initial={"front": current}) == {"front": current}
    assert not queue


def test_missing_aligned_capture_fails_with_last_seen_frame(monkeypatch):
    backend, _ = backend_with_queue([SimpleNamespace(frame=2)])
    ticks = iter([0, 2])
    monkeypatch.setattr("avsectester.simulators.carla.time.monotonic", lambda: next(ticks))
    with pytest.raises(SensorAlignmentError, match="missing aligned") as error:
        backend._source_sensor_data(103)
    assert error.value.report["expected_native_frame"] == 103
    assert error.value.report["observed_sensor_frames"] == {"front": 102}


def recipe_source(client, steps=2, time_step=0.1):
    recipe = CarlaRecipe({"client": {"synchronous": True, **client}, "ego": {}}, steps, time_step,
                         {"ego": PrescribedMotion((Control(),) * steps)})
    data = CarlaDataset({"s": recipe})
    return data.source_data("s", data.frames("s")[0])


def test_source_spawn_resolution_connects_to_the_factory_server(monkeypatch):
    seen = []

    def prepare(scenario):
        seen.append(deepcopy(scenario["client"]))
        return scenario

    monkeypatch.setattr("avsectester.simulators.carla.prepare_scenario", prepare)
    source = recipe_source({"rate": 10, "connect_ip": "127.0.0.1", "connect_port": 2000})
    backend = CarlaBackend({"client": {"rate": 10, "connect_ip": "10.0.0.5", "connect_port": 3000},
                            "ego": {}})
    resolved = backend._resolve_scenario(source, None, {})
    assert [(c["connect_ip"], c["connect_port"]) for c in seen] == [("10.0.0.5", 3000)]
    assert (resolved["client"]["connect_ip"], resolved["client"]["connect_port"]) == ("10.0.0.5", 3000)


def test_recipe_time_step_is_compared_like_recipe_validation():
    source = recipe_source({"rate": 30}, steps=1, time_step=0.0333333333)
    CarlaBackend({"client": {"rate": 30}, "ego": {}}).validate_source(source)


def test_weather_change_reaches_a_resolved_pair_replay():
    backend = CarlaBackend({"client": {}, "ego": {}, "weather": "ClearNoon"})
    backend._pair_scenario = {"client": {}, "ego": {}, "weather": "ClearNoon"}
    backend._pair_replay = {"client": {}, "ego": {}, "weather": "ClearNoon"}
    backend.configure_conditions({"weather": "HardRainNoon"})
    assert backend._resolve_scenario(None, None, {})["weather"] == "HardRainNoon"
