"""Shipped CARLA recipes satisfy the required native mobile-actor constructor contract."""
from inspect import signature
from pathlib import Path

import pytest

from avsectester.experiment import load_configuration


CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def native_actor_contract(spawn, vehicle, sensors, pipeline, autopilot, destination, client,
                          reference_to_spawn=None, spawn_transform=None):
    """Required avcarla CarlaMobileActor parameters, checked without starting a world.

    The simulator supplies client and its empty native pipeline; the recipe supplies the
    vehicle, spawn, sensor configuration, autopilot flag and optional destination value.
    """


@pytest.mark.parametrize("filename", ["carla_phantom.yaml", "carla_patch.yaml",
                                     "carla_object_insertion.yaml", "carla_patch_optimized.yaml"])
def test_recipe_ego_binds_required_native_actor_parameters(filename):
    configuration = load_configuration(CONFIGS / filename)
    for scenario in configuration.dataset.scenarios():
        recipe = configuration.dataset.native(scenario.id)
        ego = dict(recipe.scenario["ego"])
        assert ego.pop("type") == "CarlaMobileActor"
        assert ego["autopilot"] is False
        assert "destination" in ego and ego["destination"] is None
        signature(native_actor_contract).bind(
            **ego, client=object(), pipeline={"type": "SerialPipeline", "modules": []})
        for sensor in ego["sensors"]:
            assert sensor["type"] in {"CarlaRgbCamera", "CarlaLidar"}
            assert sensor["name"] and sensor["sensor_tick"] == 0.0
