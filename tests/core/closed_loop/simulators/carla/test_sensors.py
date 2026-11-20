"""CARLA capture transport, depth decoding and idempotent pinhole camera registration."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.closed_loop.simulators.carla.sensors import SensorCapture, decode_carla_depth
from avsectester.datasets.carla import CarlaDataset
from avsectester.serialization import DEFAULT_REGISTRY
from tests.fakes import CarlaNativeBoundary


def test_decode_carla_depth():
    for metres in (0., 17.25):
        encoded = round(metres / 1000 * (2**24 - 1))
        bgra = np.array([[[encoded >> 16, (encoded >> 8) & 255, encoded & 255, 255]]], np.uint8)
        image = SimpleNamespace(height=1, width=1, raw_data=bgra.tobytes())
        assert decode_carla_depth(image)[0, 0] == pytest.approx(metres, abs=.00003)


def test_sensor_capture():
    source = SensorCapture("carla.LidarMeasurement", np.array([[1., 2., 3., .5]], np.float32).tobytes(),
                           7, .7, {"channels": 16})
    captured = DEFAULT_REGISTRY.snapshot(source)
    source.metadata["channels"] = 32
    assert isinstance(captured, SensorCapture) and captured.metadata == {"channels": 16}
    np.testing.assert_array_equal(np.frombuffer(captured.raw_data, np.float32), [1., 2., 3., .5])
    decoded = DEFAULT_REGISTRY.decode(DEFAULT_REGISTRY.encode(captured))
    assert isinstance(decoded, dict)
    assert decoded["native_type"] == "carla.LidarMeasurement"
    assert decoded["frame"] == 7 and decoded["timestamp"] == .7
    assert decoded["metadata"] == {"channels": 16}
    assert decoded["raw_bytes"] == source.raw_bytes


def test_depth_camera(monkeypatch, carla_dataset):
    CarlaRgbCamera = pytest.importorskip("avcarla.sensors").CarlaRgbCamera
    from tests.fakes import CarlaNativeBoundary

    from avsectester.datasets.carla import CarlaDataset

    recipe = carla_dataset.native("scene")
    scenario = {**recipe.scenario, "ego": {
        **recipe.scenario["ego"], "sensors": [{"type": "CarlaRgbCamera", "name": "front"}]}}
    carla_dataset = CarlaDataset({"scene": replace(recipe, scenario=scenario)})
    boundary = CarlaNativeBoundary().install(monkeypatch, CarlaRgbCamera)
    ordinary = boundary.get("CarlaRgbCamera")
    assert ordinary is CarlaRgbCamera
    start = carla_dataset.scenario_start("scene", carla_dataset.frames("scene")[0])
    registered = []
    for _ in range(2):
        with CarlaSimulator({"rate": 10}, depth_camera="front", settle_iters=0).create_runtime() as runtime:
            runtime.reset(start)
            camera = boundary.get("PinholeRgbCamera")
            registered.append(camera)
            assert issubclass(camera, CarlaRgbCamera)
            assert boundary.get("CarlaRgbCamera") is ordinary
            specification = runtime.native["ego"].specification
            assert next(item for item in specification["sensors"]
                        if item.get("name", "front") == "front")["type"] == "PinholeRgbCamera"
            view = runtime.sensor_view("front")
            assert view.depth.shape == view.data.shape[:2]
            np.testing.assert_allclose(view.depth, 10., atol=.0001, rtol=0)
    assert registered[0] is registered[1]


def test_recipe_blueprint_attributes(monkeypatch, carla_dataset):
    CarlaNativeBoundary().install(monkeypatch, None)
    recipe = carla_dataset.native("scene")
    scenario = deepcopy(recipe.scenario)
    scenario["ego"]["blueprint_attributes"] = {"role_name": "ego", "ros_name": "ego",
                                                "ros2_ackermann_control": True}
    scenario["ego"]["sensors"] = [
        {"type": "CarlaRgbCamera", "name": "front",
         "blueprint_attributes": {"post_process_profile": "Cinematic"}, "ros": True},
        {"type": "CarlaLidar", "name": "lidar",
         "blueprint_attributes": {"ros_name": "lidar"}, "ros": True},
    ]

    def start():
        dataset = CarlaDataset({"scene": replace(recipe, scenario=scenario)})
        return dataset.scenario_start("scene", dataset.frames("scene")[0])

    simulator = CarlaSimulator({"rate": 10}, settle_iters=0, camera_warmup_s=0.)
    with simulator.create_runtime() as runtime:
        runtime.reset(start())
        ego = runtime.native["ego"]
        assert ego.actor.attributes["role_name"] == "ego"
        assert ego.actor.attributes["ros_name"] == "ego"
        assert ego.actor.attributes["ros2_ackermann_control"] == "true"
        camera = next(sensor.object for sensor in ego.sensors.values()
                      if sensor.object.type_id == "sensor.camera.rgb")
        assert camera.attributes["post_process_profile"] == "Cinematic"
        assert camera.ros_enabled is True
        lidar = next(sensor.object for sensor in ego.sensors.values()
                     if sensor.object.type_id == "sensor.lidar.ray_cast")
        assert lidar.attributes["ros_name"] == "lidar"
        assert lidar.ros_enabled is True
        scenario["ego"]["sensors"][0]["blueprint_attributes"] = {"unknown_native_attribute": "x"}
        with pytest.raises(ValueError, match="unknown_native_attribute"):
            runtime.reset(start())
        scenario["ego"]["sensors"][0]["blueprint_attributes"] = {"name": "other"}
        with pytest.raises(ValueError, match=r"constructor options \['name'\].*Use sensor options"):
            runtime.reset(start())

    scenario["ego"]["sensors"][0]["blueprint_attributes"] = {"post_process_profile": "Cinematic"}
    CarlaNativeBoundary(ego_attribute_overrides={"ros_name": "different"}).install(monkeypatch, None)
    with simulator.create_runtime() as runtime:
        with pytest.raises(ValueError, match=r"ego attribute 'ros_name'.*'ego'.*'different'"):
            runtime.reset(start())
        assert not any(actor.is_alive for actor in runtime.native["world"].get_actors()
                       if actor.type_id.startswith("vehicle."))
