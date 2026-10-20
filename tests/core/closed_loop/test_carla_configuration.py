"""Native connection and random-seed settings fail before allocating a world."""
import pytest

from avsectester.closed_loop.simulators.carla import CarlaSimulator


@pytest.mark.parametrize('client', [
    {'seed': True}, {'seed': -1}, {'seed': 1.2}, {'traffic_manager_seed': '7'},
    {'connect_port': True}, {'connect_port': 0}, {'connect_port': 65536},
    {'traffic_manager_port': 2.5}, {'traffic_manager_port': '8000'},
])
def test_invalid_seed_or_port_is_rejected_without_a_runtime(client):
    with pytest.raises(ValueError):
        CarlaSimulator(client)


def test_run_seed_overrides_client_seed_and_explicit_traffic_manager_seed_stays_fixed():
    simulator = CarlaSimulator({'seed': 4, 'traffic_manager_seed': 9})
    simulator_runtime = simulator.create_runtime(seed=7)
    assert simulator.client_config['seed'] == 4
    assert simulator_runtime.client_config['seed'] == 7
    assert simulator_runtime.client_config['traffic_manager_seed'] == 9
    with pytest.raises(ValueError, match='seed'):
        simulator.create_runtime(seed=True)


def test_world_overrides_merge_nested_fields_with_recipe_defaults(monkeypatch):
    from avsectester.closed_loop.datatypes import Control
    from avsectester.datasets.carla import CarlaDataset, CarlaRecipe, PrescribedMotion

    recipe = CarlaRecipe({
        'client': {'map_name': 'Town03'}, 'ego': {},
        'lighting': {'street_lights': {'enabled': True, 'intensity': 20}},
        'weather': {'cloudiness': 40, 'precipitation': 0},
    }, 1, .1, {'ego': PrescribedMotion((Control(),))})
    dataset = CarlaDataset({'scenario': recipe})
    start = dataset.scenario_start('scenario', dataset.frames('scenario')[0])
    monkeypatch.setattr('avsectester.closed_loop.simulators.carla.prepare_scenario', lambda value: value)
    simulator_runtime = CarlaSimulator({'rate': 10}).create_runtime()
    resolved = simulator_runtime._resolve_scenario(start, None, {
        'lighting': {'street_lights': {'intensity': 5}}, 'weather': {'precipitation': 60},
    })
    assert resolved['lighting']['street_lights'] == {'enabled': True, 'intensity': 5}
    assert resolved['weather'] == {'cloudiness': 40, 'precipitation': 60}
    assert recipe.scenario['lighting']['street_lights']['intensity'] == 20
