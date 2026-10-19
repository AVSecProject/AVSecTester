"""Environment settings preserve run ordering and comparison-frame randomness."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.attack.runtime import AttackRuntime, SensorAttackEffect
from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.datatypes import Control, Observation, WorldSnapshot
from avsectester.closed_loop.environment import (
    SensorEnvironment, merge_environments, normalize_environment, validate_environment,
)
from avsectester.closed_loop.sensor_transforms import GaussianNoise
from avsectester.closed_loop.hooks import Hook, HookSet
from avsectester.closed_loop.runner import ClosedLoopRunner, RecordingPolicy, RunRequest
from avsectester.closed_loop.simulator import Simulator, SimulatorRuntime
from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.closed_loop.simulators.nurec import NuRecSimulator, StubRenderer
from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator


def test_environment_overrides_preserve_unspecified_settings_and_replace_sensor_list():
    base = {"world": {"weather": {"cloudiness": 40, "precipitation": 20}},
            "sensors": [{"type": "GaussianNoise", "options": {"severity": 0.1}, "camera": "front"}]}
    previous = deepcopy(base)
    result = merge_environments(base, {"world": {"weather": {"cloudiness": 80}}}, {})
    assert result["world"]["weather"] == {"cloudiness": 80, "precipitation": 20}
    assert result["sensors"] == base["sensors"]
    assert merge_environments(result, {"sensors": []})["sensors"] == []
    assert base == previous
    for simulator in (NuRecSimulator({}, StubRenderer()), RecordedReplaySimulator()):
        with pytest.raises(ValueError, match="world settings.*weather"):
            validate_environment(base, simulator)
    assert validate_environment(base, CarlaSimulator({"client": {}, "ego": {}})) == base
    with pytest.raises(ValueError, match="Unknown environment"):
        normalize_environment({"weather": "HardRainNoon"})
    with pytest.raises(ValueError, match="Unknown sensor corruption"):
        normalize_environment({"sensors": [{"type": "gaussian_noise"}]})


def test_sensor_randomness_uses_absolute_frame_position_and_does_not_change_raw_payload():
    rgb = np.full((20, 20, 3), 128, dtype=np.uint8)
    a, b = Observation(0, 12, {"front": rgb}), Observation(0, 90, {"front": rgb})
    environment = SensorEnvironment({"sensors": [{"type": "GaussianNoise", "options": {"severity": .3}}]}, 7)
    first, second = environment.apply(a, position=0), environment.apply(b, position=0)
    np.testing.assert_array_equal(first.sensor_data["front"], second.sensor_data["front"])
    assert not np.array_equal(first.sensor_data["front"], environment.apply(a, position=1).sensor_data["front"])
    assert np.all(rgb == 128)
    assert GaussianNoise is __import__("avsectester.perturbations.image", fromlist=["GaussianNoise"]).GaussianNoise
    with pytest.raises(ValueError, match="unavailable camera"):
        SensorEnvironment({"sensors": [{"type": "Fog", "camera": "missing"}]}).apply(a, 0)


def test_sensor_environment_runs_after_attack_and_before_vehicle_io():
    order = []

    class World(SimulatorRuntime):
        def _initialize_world(self, start, initialization, environment):
            self.i = 0
        def _state(self):
            return WorldSnapshot(float(self.i), self.i)
        def _observe(self):
            return Observation(float(self.i), self.i,
                               {"front": np.full((2, 2, 3), 10, dtype=np.uint8)})
        def _advance_native(self, control):
            self.i += 1

    class Config(Simulator):
        def create_runtime(self, seed=None):
            return World(self)

    class Fill(SensorAttackEffect):
        sensor = "front"
        def apply(self, data, view):
            raise AssertionError("Test overrides sensor view access explicitly")

    class Runtime(AttackRuntime):
        def step(self, context):
            order.append("attack")
            class Replace(Fill):
                def apply(self, data, view):
                    return np.full_like(data, 100)
            return (Replace(),)

    class Driver(AVStack):
        def __call__(self, observation):
            order.append("avstack")
            assert observation.sensor_data["front"].min() == 180
            return Control()

    world = Config().create_runtime()
    world.sensor_view = lambda sensor: SimpleNamespace(data=world.observe().sensor_data[sensor], sensor=sensor)
    def check(observation, context):
        order.append("hook")
        assert observation.sensor_data["front"].min() == 180
        return observation
    log = ClosedLoopRunner().run(world, Driver(), RunRequest(
        1, hooks=HookSet([Hook("observation", check)]), attacks=(Runtime(),),
        environment={"sensors": [{"type": "Brightness", "options": {"severity": 1.0}, "camera": "front"}]},
        recording=RecordingPolicy(("front",), ("raw", "model_input")),
    ))
    assert order == ["attack", "hook", "avstack"]
    assert log.steps[0].raw_observation.sensor_data["front"].max() == 10


def test_carla_native_environment_uses_layer_lighting_and_vehicle_physics_apis(monkeypatch):
    import sys
    from unittest.mock import Mock

    class Lights(int):
        # CARLA bitwise combinations return int, while set_light_state requires its enum.
        NONE = 0
        Position = 1
        LowBeam = 2

    monkeypatch.setitem(sys.modules, "carla", SimpleNamespace(
        MapLayer=SimpleNamespace(Buildings="buildings", Foliage="foliage"),
        LightGroup=SimpleNamespace(Street="street"), VehicleLightState=Lights,
        Color=lambda *channels: channels,
    ))
    manager = Mock()
    manager.get_all_lights.return_value = ["lamp"]
    native_world = Mock()
    native_world.get_map.return_value.name = "/Game/Carla/Maps/Town10HD_Opt"
    native_world.get_lightmanager.return_value = manager
    ego, lead = Mock(type_id="vehicle.ego"), Mock(type_id="vehicle.lead")
    def require_enum(state):
        assert isinstance(state, Lights)
    ego.set_light_state.side_effect = lead.set_light_state.side_effect = require_enum
    physics = SimpleNamespace(wheels=[SimpleNamespace(tire_friction=3.5)], mass=1600)
    ego.get_physics_control.return_value = physics
    simulator_runtime = CarlaSimulator({"client": {}, "ego": {}}).create_runtime()
    simulator_runtime.client = SimpleNamespace(world=native_world)
    simulator_runtime.ego = SimpleNamespace(actor=ego)
    simulator_runtime.lead = lead
    simulator_runtime._apply_map_layers({"unload": ["Foliage"], "load": ["Buildings"]})
    native_world.unload_map_layer.assert_called_once_with("foliage")
    native_world.load_map_layer.assert_called_once_with("buildings")
    simulator_runtime._apply_lighting({
        "street_lights": {"enabled": True, "intensity": 25, "color": [255, 180, 100]},
        "vehicle_lights": {"all": ["Position"], "ego": ["Position", "LowBeam"]},
    })
    manager.set_day_night_cycle.assert_called_once_with(False)
    manager.turn_on.assert_called_once_with(["lamp"])
    manager.set_intensity.assert_called_once_with(["lamp"], 25)
    manager.set_color.assert_called_once_with(["lamp"], (255, 180, 100))
    ego.set_light_state.assert_called_once_with(Lights.Position | Lights.LowBeam)
    lead.set_light_state.assert_called_once_with(Lights.Position)
    simulator_runtime._apply_road_friction({"tire_friction": 0.6, "actors": ["ego"]})
    ego.apply_physics_control.assert_called_once_with(physics)
    assert physics.wheels[0].tire_friction == 0.6 and physics.mass == 1600
    lead.apply_physics_control.assert_not_called()
    native_world.get_map.return_value.name = "Town10HD"
    with pytest.raises(ValueError, match="_Opt"):
        simulator_runtime._apply_map_layers({"unload": ["Foliage"]})


@pytest.mark.parametrize("world, message", [
    ({"weather": "not-a-preset"}, "weather preset"),
    ({"weather": {"cloudiness": 101}}, "weather.cloudiness"),
    ({"weather": {"cloudiness": True}}, "weather.cloudiness"),
    ({"weather": {"sun_altitude_angle": -91}}, "weather.sun_altitude_angle"),
    ({"weather": {"fog_distance": float("nan")}}, "weather.fog_distance"),
    ({"weather": {"not_a_weather_field": 1}}, "weather supports only"),
    ({"lighting": {"street_lights": {"enabled": 1}}}, "enabled"),
    ({"lighting": {"street_lights": {"intensity": -1}}}, "intensity"),
    ({"lighting": {"street_lights": {"intensity": "25"}}}, "intensity"),
    ({"lighting": {"street_lights": {"color": [255, 256, 0]}}}, "color"),
    ({"lighting": {"street_lights": {"group": "Vehicle"}}}, "group"),
    ({"lighting": {"vehicle_lights": {"ego": ["not-a-flag"]}}}, "vehicle_lights"),
    ({"lighting": {"vehicle_lights": {"ego": "LowBeam"}}}, "vehicle_lights"),
    ({"map_layers": {"unload": ["not-a-layer"]}}, "map_layers.unload"),
    ({"map_layers": {"load": "Buildings"}}, "map_layers.load"),
    ({"road_friction": {"tire_friction": 0}}, "positive"),
    ({"road_friction": {"tire_friction": float("inf")}}, "finite"),
    ({"road_friction": {"tire_friction": 1, "actors": [1]}}, "actors"),
])
def test_carla_world_values_are_rejected_before_a_runtime_is_created(monkeypatch, world, message):
    simulator = CarlaSimulator({"client": {}, "ego": {}})
    monkeypatch.setattr(simulator, "create_runtime", lambda *args, **kwargs: pytest.fail("world started"))
    with pytest.raises(ValueError, match=message):
        validate_environment({"world": world}, simulator)


def test_carla_world_value_validation_needs_no_native_world_and_keeps_input_independent(monkeypatch):
    import builtins

    original_import = builtins.__import__
    def without_carla(name, *args, **kwargs):
        if name == "carla" or name.startswith("avcarla"):
            pytest.fail("Pure setting validation imported a native SDK")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_carla)
    environment = {"world": {
        "weather": {"sun_altitude_angle": -30, "fog_distance": 50, "precipitation": 40},
        "map_layers": {"load": ["Buildings"], "unload": ["Foliage"]},
        "lighting": {"street_lights": {"enabled": False, "intensity": 0, "color": [255, 0, 0]},
                     "vehicle_lights": {"not-spawned-yet": ["LowBeam", "Brake"]}},
        "road_friction": {"tire_friction": .6, "actors": ["not-spawned-yet"]},
    }}
    simulator = CarlaSimulator({"client": {}, "ego": {}})
    result = validate_environment(environment, simulator)
    assert result["world"] == environment["world"]
    result["world"]["lighting"]["vehicle_lights"]["not-spawned-yet"].append("Position")
    assert environment["world"]["lighting"]["vehicle_lights"]["not-spawned-yet"] == ["LowBeam", "Brake"]
    for preset in ("ClearNoon", "SoftRainSunset", "WetCloudyNight", "DustStorm"):
        validate_environment({"world": {"weather": preset}}, simulator)


def test_carla_configured_world_defaults_are_checked_before_preparation():
    simulator = CarlaSimulator({"client": {}, "ego": {},
                                "lighting": {"street_lights": {"enabled": "yes"}}})
    with pytest.raises(ValueError, match="enabled"):
        validate_environment({}, simulator)
    # An explicit setting replaces its configured default before the value check.
    validate_environment({"world": {"lighting": {"street_lights": {"enabled": True}}}}, simulator)


def test_carla_recipe_world_values_are_checked_before_spawn_resolution(monkeypatch):
    from avsectester.closed_loop.simulators import carla
    from avsectester.datasets.carla import CarlaDataset, CarlaRecipe, PrescribedMotion

    scenario = {"client": {"synchronous": True, "rate": 10}, "ego": {},
                "weather": {"precipitation": -1}}
    recipe = CarlaRecipe(scenario, 1, .1, {"ego": PrescribedMotion((Control(),))})
    dataset = CarlaDataset({"recipe": recipe})
    start = dataset.scenario_start("recipe", dataset.frames("recipe")[0])
    runtime = CarlaSimulator({"client": {"rate": 10}, "ego": {}}).create_runtime()
    monkeypatch.setattr(carla, "prepare_scenario", lambda *_: pytest.fail("spawn resolution contacted a server"))
    with pytest.raises(ValueError, match="precipitation"):
        runtime._resolve_scenario(start, None, {})
