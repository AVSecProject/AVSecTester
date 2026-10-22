"""Preparation models define the canonical inputs used by selection and offline work."""

from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter
from avsectester.experiment import ExperimentConfiguration, RunsConfiguration
from avsectester.experiment.models import preparation_record
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator, MinimalAttack


def test_preparation_record():
    color = (255, 200, 100)
    attack = MinimalAttack()
    attack.environment = {"world": {"lighting": {"intensity": .5}}}
    configuration = ExperimentConfiguration(
        MemoryDataset(), MemorySimulator(), DrivingAdapter(lambda seed: ConstantAVStack()),
        {"probe": attack}, runs=RunsConfiguration(seeds=(3, 7), environment={"world": {
            "weather": "rain", "lighting": {"street_lights": {"color": color}},
        }}),
    )
    assert preparation_record(configuration) == {
        "seed": 3,
        "offline_environments": {"probe": {
            "world": {"weather": "rain", "lighting": {
                "street_lights": {"color": list(color)}, "intensity": .5,
            }}, "sensors": [],
        }},
    }
    actual_color = configuration.runs.environment["world"]["lighting"]["street_lights"]["color"]
    assert actual_color == color and isinstance(actual_color, tuple)
