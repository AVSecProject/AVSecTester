"""prepare_scenario: seeded scene selection and settings resolution (our code, not avcarla internals)."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from avsectester.scenario import prepare_scenario


@pytest.fixture
def scene_builder(monkeypatch):
    from avsectester import scenario

    library = Mock()
    library.filter.return_value = [
        SimpleNamespace(id="vehicle.c"),
        SimpleNamespace(id="vehicle.a"),
        SimpleNamespace(id="vehicle.b"),
    ]
    client = SimpleNamespace(
        world=SimpleNamespace(get_blueprint_library=lambda: library),
        spawn_points=list(range(10)),
        scene_settings=lambda: {"map_name": "Town10HD_Opt", "weather": {"wetness": 12.0}},
        close=Mock(),
    )
    build = Mock(return_value=client)
    monkeypatch.setattr(scenario.CARLA, "build", build)
    return client, build, library


def random_scene(seed=0):
    return {
        "client": {"type": "CarlaClient", "seed": seed},
        "ego": {
            "vehicle": "random",
            "spawn": "random",
            "destination": "random",
            "sensors": [{"type": "CarlaLidar", "noise_seed": 123}],
        },
        "npcs": [
            {"type": "CarlaNpc", "spawn": "random", "npc_type": "vehicle"},
            {"type": "CarlaNpc", "spawn": 0, "npc_type": "vehicle.a"},
        ],
    }


def test_scene_selection_uses_seed_and_stable_candidates(scene_builder):
    _, _, library = scene_builder
    config = random_scene()
    before = deepcopy(config)
    first = prepare_scenario(config)
    # Neither blueprint enumeration nor unrelated random draws change the scene.
    library.filter.return_value.reverse()
    np.random.random(50)
    second = prepare_scenario(config)
    assert first == second
    assert config == before
    assert first["ego"]["sensors"][0]["noise_seed"] == 123
    assert first["npcs"][1] == config["npcs"][1]
    assert len({first["ego"]["spawn"], *(npc["spawn"] for npc in first["npcs"])}) == 3
    assert isinstance(first["ego"]["destination"], int)
    assert first["client"]["traffic_manager_seed"] == 0


def test_missing_seed_is_resolved_once_and_explicit_values_survive(scene_builder, monkeypatch):
    from avsectester import scenario

    generate = Mock(return_value=987)
    monkeypatch.setattr(scenario.secrets, "randbelow", generate)
    config = random_scene(None)
    config["client"]["traffic_manager_seed"] = 42
    config["ego"].update(vehicle="vehicle.b", spawn=7, destination=8)
    result = prepare_scenario(config)
    generate.assert_called_once()
    assert result["client"]["seed"] == 987
    assert result["client"]["traffic_manager_seed"] == 42
    assert result["ego"]["vehicle"] == "vehicle.b"
    assert result["ego"]["spawn"] == 7
    assert result["ego"]["destination"] == 8


def test_scene_preparation_releases_client_on_failure(scene_builder):
    client, _, _ = scene_builder
    client.spawn_points = []
    with pytest.raises(ValueError, match="spawn points"):
        prepare_scenario(random_scene())
    client.close.assert_called_once()


@pytest.mark.parametrize("strict", [None, False, True])
def test_scene_preparation_allows_clean_spawn_retries(scene_builder, strict):
    config = random_scene()
    if strict is not None:
        config["client"]["strict_spawn"] = strict
    resolved = prepare_scenario(config)
    assert resolved["client"]["strict_spawn"] is (strict is True)
