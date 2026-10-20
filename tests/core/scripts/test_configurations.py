"""Every supplied experiment loads and validates without a world, model or dataset read."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from avsectester.experiment import load_configuration
from avsectester.experiment.engine import validate_configuration_environment
from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.closed_loop.simulators.nurec import NuRecSimulator
from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator
from avsectester.datasets.nurec import NuRecDataset
from avsectester.datasets.nuscenes import NuScenesDataset
from scripts import REPO_ROOT

CONFIGURATIONS = tuple(sorted((REPO_ROOT / "configs").glob("*.yaml")))
EXPECTED = {
    "carla_phantom.yaml", "carla_patch.yaml", "carla_object_insertion.yaml",
    "carla_patch_optimized.yaml", "nurec_speed_feedback.yaml", "nurec_object_insertion.yaml", "nurec_patch.yaml",
    "nuscenes_object_insertion.yaml",
}


def test_supplied_experiment_set_is_complete():
    assert {path.name for path in CONFIGURATIONS} == EXPECTED


@pytest.mark.parametrize("path", CONFIGURATIONS, ids=lambda path: path.name)
def test_experiment_configuration_preflight_needs_no_native_resources(path, monkeypatch):
    forbidden = Mock(side_effect=AssertionError("preflight must not create a world or load records"))
    for simulator in (CarlaSimulator, NuRecSimulator, RecordedReplaySimulator):
        monkeypatch.setattr(simulator, "create_runtime", forbidden)
    monkeypatch.setattr(NuRecDataset, "scenarios", forbidden)
    monkeypatch.setattr(NuScenesDataset, "scenarios", forbidden)
    configuration = load_configuration(path)
    validate_configuration_environment(configuration)
    assert len(configuration.attacks) == 1
    assert configuration.output == (REPO_ROOT / "results" / path.stem).resolve()
    assert {t.name for t in configuration.treatments} >= {"clean"}
    assert all(t.attack_name is None or t.attack_name in configuration.attacks
               for t in configuration.treatments)
    for attack in configuration.attacks.values():
        assert attack.decision_steps > 0
    forbidden.assert_not_called()


@pytest.mark.parametrize("path", CONFIGURATIONS, ids=lambda path: path.name)
def test_dataset_and_model_paths_are_relative_to_configuration(path):
    specification = yaml.safe_load(path.read_text())
    options = specification["dataset"]["options"]
    for key in ("usdz_paths", "dataroot"):
        value = options.get(key)
        if value is None:
            continue
        for item in value if isinstance(value, list) else [value]:
            assert not Path(item).is_absolute()
    checkpoint = specification["avstack"].get("options", {}).get("checkpoint_path")
    assert checkpoint is None or not Path(checkpoint).is_absolute()
    forbidden = {"backend", "stack", "runtime", "clean_runtime", "patches", "interventions",
                 "selection", "conditions", "frames", "mode", "treatment"}
    assert not forbidden.intersection(specification)
