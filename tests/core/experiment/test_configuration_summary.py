"""Configuration summaries describe public inputs and mark unverifiable attack parameters."""
import pytest

from avsectester.closed_loop.avstacks import DrivingAdapter
from avsectester.experiment import ExperimentConfiguration
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator, MinimalAttack


class UnverifiedAttack(MinimalAttack):
    def configuration_parameters(self):
        return {"unsupported": object()}


def test_configuration_summary():
    configuration = ExperimentConfiguration(
        MemoryDataset(), MemorySimulator(), DrivingAdapter(lambda seed: ConstantAVStack()),
        {"probe": MinimalAttack(assets={"value": 3})})
    summary = configuration.configuration_summary
    assert summary["dataset"] == {"type": "tests.fakes:MemoryDataset", "scenarios": ["scene"]}
    assert summary["attacks"]["probe"]["parameters"]["assets"] == {"value": 3}
    unverified = ExperimentConfiguration(
        MemoryDataset(), MemorySimulator(), configuration.avstack, {"probe": UnverifiedAttack()})
    with pytest.warns(RuntimeWarning, match="parameter compatibility is unverified"):
        description = unverified.configuration_summary["attacks"]["probe"]
    assert description["parameters"] is None and description["parameters_verified"] is False
