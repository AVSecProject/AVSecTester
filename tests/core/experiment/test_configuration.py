"""Configuration is loaded through its public constructors and validation API."""

from copy import deepcopy
from pathlib import Path

from PIL import Image
import pytest
import yaml

from avsectester.attack.placement import WorldPlacement
from avsectester.experiment import load_configuration
from avsectester.experiment.config import resolve_type
from avsectester.experiment.engine import validate_configuration_environment
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator, MinimalAttack


def specification():
    return {
        "dataset": {"type": "tests.fakes:MemoryDataset"},
        "simulator": {"type": "tests.fakes:MemorySimulator"},
        "avstack": {"type": "tests.fakes:ConstantAVStack"},
        "attacks": {"probe": {"type": "tests.fakes:MinimalAttack"}},
    }


def test_load_configuration(tmp_path):
    spec = specification()
    loaded = load_configuration(spec)
    assert isinstance(loaded.dataset, MemoryDataset)
    assert isinstance(loaded.simulator, MemorySimulator)
    assert isinstance(loaded.avstack.factory(0), ConstantAVStack)
    assert isinstance(loaded.attacks["probe"], MinimalAttack)
    assert [(item.name, item.attack_name) for item in loaded.runs.treatments] == [
        ("clean", None), ("probe", "probe"),
    ]
    assert load_configuration(loaded) is loaded

    Image.new("RGBA", (3, 2), (255, 255, 255, 255)).save(tmp_path / "texture.png")
    (tmp_path / "attack.yaml").write_text(yaml.safe_dump({
        "type": "tests.fakes:MinimalAttack", "options": {"assets": {
            "type": "avsectester.attack.insertion.assets:ImageTexture", "options": {"image": "texture.png"},
        }},
    }))
    spec["attacks"] = {"probe": "attack.yaml"}
    spec["runs"] = {"output": "results"}
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(spec))
    for source in (str(path), path):
        loaded = load_configuration(source)
        assert loaded.runs.output == tmp_path / "results"
        assert Path(loaded.attacks["probe"].assets.image) == tmp_path / "texture.png"

    spec["dataset"] = {"type": "CarlaDataset", "options": {"recipes": {}, "map_directory": "maps"}}
    path.write_text(yaml.safe_dump(spec))
    assert load_configuration(path).dataset.map_directory == tmp_path / "maps"

    for bundled in (Path(__file__).resolve().parents[3] / "configs").glob("*.yaml"):
        configuration = load_configuration(bundled)
        validate_configuration_environment(configuration)
        assert configuration.attacks

    unknown = deepcopy(specification())
    unknown["unexpected"] = True
    with pytest.raises(ValueError, match="Unknown experiment keys"):
        load_configuration(unknown)
    missing = specification()
    missing["attacks"] = {}
    with pytest.raises(ValueError, match="exactly one attack"):
        load_configuration(missing)


def test_resolve_type():
    assert resolve_type("avsectester.attack.placement:WorldPlacement") is WorldPlacement
    assert resolve_type("tests.fakes:ConstantAVStack") is ConstantAVStack
    assert resolve_type("PatchAttack").__name__ == "PatchAttack"
    with pytest.raises(ValueError, match="Available attacks"):
        resolve_type("PatchAtack")
    with pytest.raises(ValueError, match="package.module:name"):
        resolve_type("tests:fakes:ConstantAVStack")


def test_validate_configuration_environment():
    spec = specification()
    spec["runs"] = {"environment": {"sensors": [
        {"type": "GaussianNoise", "options": {"severity": .1}, "camera": "front"},
    ]}}
    configuration = load_configuration(spec)
    validate_configuration_environment(configuration)
    spec["runs"]["environment"] = {"world": {"unsupported_setting": True}}
    with pytest.raises(ValueError, match="does not support"):
        validate_configuration_environment(load_configuration(spec))
