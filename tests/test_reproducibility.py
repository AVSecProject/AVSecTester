"""Concrete scene selection and world reset."""

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


def test_retry_spawn_pose_replays_without_offset_or_coordinate_conversion():
    import carla
    from avcarla.actor import parse_spawn, try_spawn_actor
    from avsectester.scenario import _spawn_config

    world = Mock()
    transform = carla.Transform(
        carla.Location(10, -20, 1), carla.Rotation(pitch=5, yaw=90, roll=-2)
    )
    vehicle = Mock(type_id="vehicle.test")
    vehicle.get_transform.side_effect = lambda: transform
    world.try_spawn_actor.side_effect = [None, vehicle]
    spawned = try_spawn_actor(world, SimpleNamespace(id="vehicle.test"), transform)
    assert world.try_spawn_actor.call_count == 2
    assert transform.location.y > -18  # retry moved along the vehicle heading
    spec = {"spawn": 0, "reference_to_spawn": {"location": [4, 5, 6]}}
    recorded = _spawn_config(
        spec, SimpleNamespace(actor=spawned, spawn_transform=transform), "vehicle"
    )
    replay = parse_spawn(
        spawn=0,
        spawn_points=[],
        spawns_chosen=[],
        reference_to_spawn=spec["reference_to_spawn"],
        spawn_transform=recorded["spawn_transform"],
    )
    for field, axes in [("location", ("x", "y", "z")), ("rotation", ("pitch", "yaw", "roll"))]:
        for axis in axes:
            assert getattr(getattr(replay, field), axis) == getattr(getattr(transform, field), axis)


@pytest.mark.parametrize("kind", ["CarlaNpc", "CarlaMobileActor"])
def test_recorded_spawn_failure_never_relocates_even_with_retries_enabled(monkeypatch, kind):
    import avcarla.actor as module

    client = SimpleNamespace(
        world=Mock(),
        spawn_points=[],
        spawns_chosen=[],
        rng=np.random.RandomState(0),
        strict_spawn=False,
    )
    client.world.try_spawn_actor.return_value = None
    monkeypatch.setattr(
        module, "parse_vehicle_blueprint", Mock(return_value=SimpleNamespace(id="vehicle.test"))
    )
    recorded = {
        "location": {"x": 13.0, "y": -20.0, "z": 1.0},
        "rotation": {"pitch": 5.0, "yaw": 90.0, "roll": -2.0},
    }
    before = deepcopy(recorded)
    kwargs = dict(spawn=0, client=client, spawn_transform=recorded)
    if kind == "CarlaMobileActor":
        kwargs.update(
            vehicle="vehicle.test", sensors=[], pipeline={}, autopilot=False, destination=None
        )
    with pytest.raises(RuntimeError, match="configured transform"):
        getattr(module, kind)(**kwargs)
    client.world.try_spawn_actor.assert_called_once()
    assert recorded == before
    assert client.world.try_spawn_actor.call_args.args[1].location.x == 13.0


def test_client_reloads_before_reseeding_traffic_manager(monkeypatch):
    import avcarla.client as module

    events = []
    settings = SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None)
    old_world = Mock()
    old_world.get_settings.side_effect = lambda: deepcopy(settings)
    old_world.get_map.return_value.name = "Town10HD_Opt"

    def configure(value):
        assert value.synchronous_mode
        assert value.fixed_delta_seconds == 0.05
        events.append("settings")

    old_world.apply_settings.side_effect = configure
    new_world = Mock()
    new_world.get_actors.return_value.filter.return_value = []
    manager = Mock()
    manager.set_random_device_seed.side_effect = lambda seed: events.append(("seed", seed))
    raw = Mock()
    raw.get_world.return_value = old_world

    def reload(reset_settings):
        assert reset_settings is False
        events.append("reload")
        return new_world

    raw.reload_world.side_effect = reload
    raw.get_trafficmanager.return_value = manager
    monkeypatch.setattr(module.carla, "Client", Mock(return_value=raw))
    client = module.CarlaClient(
        "localhost",
        2200,
        8200,
        23,
        True,
        20,
        reset_world=True,
        randomize_lights=False,
    )
    assert events == ["settings", "reload", ("seed", 23)]
    assert client.world is new_world
    assert client.map is new_world.get_map.return_value
    raw.get_trafficmanager.assert_called_once_with(8200)
    client.close()
    new_world.apply_settings.assert_called_once_with(settings)


def test_strict_spawn_does_not_move_or_retry():
    import carla
    from avcarla.actor import try_spawn_actor

    world = Mock()
    world.try_spawn_actor.return_value = None
    transform = carla.Transform(carla.Location(10, 20, 1))
    with pytest.raises(RuntimeError, match="configured transform"):
        try_spawn_actor(world, SimpleNamespace(id="vehicle.test"), transform, strict=True)
    world.try_spawn_actor.assert_called_once()
    assert (transform.location.x, transform.location.y, transform.location.z) == (10, 20, 1)


def test_npc_uses_configured_traffic_manager(monkeypatch):
    import avcarla.actor as module

    actor = Mock()
    client = SimpleNamespace(
        world=Mock(),
        spawn_points=[],
        spawns_chosen=[],
        rng=np.random.RandomState(0),
        strict_spawn=True,
        traffic_manager_port=8200,
    )
    monkeypatch.setattr(module, "parse_spawn", Mock())
    monkeypatch.setattr(module, "parse_vehicle_blueprint", Mock())
    spawn = Mock(return_value=actor)
    monkeypatch.setattr(module, "try_spawn_actor", spawn)
    module.CarlaNpc(spawn=0, client=client, npc_type="vehicle")
    actor.set_autopilot.assert_called_once_with(True, 8200)
    assert spawn.call_args.kwargs == {"strict": True}


def test_cleanup_stops_sensor_before_destroying_ego():
    from avcarla.actor import CarlaObject
    from avcarla.sensors import CarlaLidar

    events = []
    sensor = CarlaLidar.__new__(CarlaLidar)
    sensor.object = Mock()
    sensor.object.is_listening = True
    sensor.object.stop.side_effect = lambda: events.append("stop sensor")
    sensor.object.destroy.side_effect = lambda: events.append("destroy sensor")
    ego = CarlaObject.__new__(CarlaObject)
    ego.sensors = {"lidar": sensor}
    ego.actor = Mock()
    ego.actor.destroy.side_effect = lambda: events.append("destroy ego")
    ego.destroy()
    assert events == ["stop sensor", "destroy sensor", "destroy ego"]


def test_npc_cleanup_unregisters_from_traffic_manager():
    from avcarla.actor import CarlaNpc

    npc = CarlaNpc.__new__(CarlaNpc)
    npc.actor = Mock()
    npc.actor.is_alive = True
    npc.traffic_manager_port = 8200
    npc.destroy()
    npc.actor.set_autopilot.assert_called_once_with(False, 8200)
    npc.actor.destroy.assert_called_once()


def test_client_replays_explicit_weather_and_lights(monkeypatch):
    import avcarla.client as module

    light = Mock()
    light.get_opendrive_id.return_value = "road-light-1"
    raw, world = Mock(), Mock()
    raw.get_world.return_value = world
    raw.reload_world.return_value = world
    world.get_weather.return_value = SimpleNamespace(wetness=0.0, cloudiness=0.0)
    world.get_actors.return_value.filter.return_value = [light]
    monkeypatch.setattr(module.carla, "Client", Mock(return_value=raw))
    module.CarlaClient(
        "localhost",
        2200,
        8200,
        5,
        True,
        20,
        reset_world=True,
        weather={"wetness": 35.0},
        traffic_lights={
            "road-light-1": {
                "state": "Red",
                "green_time": 6,
                "yellow_time": 2,
                "red_time": 8,
                "frozen": False,
            }
        },
    )
    assert world.set_weather.call_args.args[0].wetness == 35.0
    light.set_state.assert_called_once_with(module.carla.TrafficLightState.Red)
    light.set_green_time.assert_called_once_with(6)
    light.set_yellow_time.assert_called_once_with(2)
    light.set_red_time.assert_called_once_with(8)
    light.freeze.assert_called_once_with(False)
