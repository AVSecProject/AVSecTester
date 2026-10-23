"""Configuration is loaded through its public constructors and validation API."""

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from avsectester.attack import Attack
from avsectester.experiment import load_configuration
from avsectester.experiment.engine import validate_configuration_environment
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator, MinimalAttack

if TYPE_CHECKING:
    import pathlib as annotation_paths
    from collections.abc import Hashable as AnnotationOption


class NestedPathOptions:
    """A user component distinguishes ordinary strings from filesystem paths."""

    def __init__(self, path: str, source: str | Path):
        self.path = path
        self.source = source


class UserPathAttack(Attack):
    """A user attack declares its own path parameters without framework key names."""

    def __init__(self, source: Path, sources: Sequence[str | Path] = (),
                 untyped=None, payload: Any = None, settings: dict[str, Any] | None = None,
                 nested: NestedPathOptions | None = None, **options):
        self.source = source
        self.sources = sources
        self.untyped = untyped
        self.payload = payload
        self.settings = settings
        self.nested = nested
        self.options = options

    def build_metadata(self, scenario):
        return ()


class UnresolvedPathAttack(Attack):
    """An unresolved path annotation cannot safely identify filesystem inputs."""

    def __init__(self, source: "annotation_paths.Path"):
        self.source = source

    def build_metadata(self, scenario):
        return ()


class UnresolvedOptionAttack(Attack):
    """An unrelated forward reference does not hide a valid path annotation."""

    def __init__(self, source: str | Path, setting: "AnnotationOption" = None):
        self.source = source
        self.setting = setting

    def build_metadata(self, scenario):
        return ()


class SeededAVStack(ConstantAVStack):
    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed


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
    seeded = specification()
    seeded["avstack"] = {"type": f"{__name__}:SeededAVStack"}
    assert load_configuration(seeded).avstack.factory(5).seed == 5
    seeded["avstack"]["options"] = {"seed": 1}
    with pytest.raises(ValueError, match="reserved"):
        load_configuration(seeded)
    assert isinstance(loaded.attacks["probe"], MinimalAttack)
    assert [(item.name, item.attack_name) for item in loaded.runs.treatments] == [
        ("clean", None), ("probe", "probe"),
    ]
    assert load_configuration(loaded) is loaded

    attack_directory = tmp_path / "definitions"
    attack_directory.mkdir()
    (attack_directory / "attack.yaml").write_text(yaml.safe_dump({
        "type": f"{__name__}:UserPathAttack", "options": {"source": "texture.png"},
    }))
    spec["attacks"] = {"probe": "definitions/attack.yaml"}
    spec["runs"] = {"output": "results"}
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(spec))
    for source in (str(path), path):
        loaded = load_configuration(source)
        assert loaded.runs.output == tmp_path / "results"
        assert Path(loaded.attacks["probe"].source) == attack_directory / "texture.png"

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


def test_constructor_paths(tmp_path):
    absolute = str(tmp_path / "keep" / ".." / "weights.bin")
    uri = "https://example.test/weights.bin"
    options = {
        "source": "weights.bin",
        "sources": ["first.bin", uri, absolute],
        "untyped": "untyped.bin",
        "payload": "payload.bin",
        "settings": {"image": "ordinary.bin", "nested": {"path": "unchanged.bin"}},
        "nested": {
            "type": f"{__name__}:NestedPathOptions",
            "options": {"path": "ordinary-string.bin", "source": "nested.bin"},
        },
        "image": "extra.bin",
        "path": "extra-path.bin",
    }
    spec = specification()
    spec["attacks"] = {"probe": {"type": f"{__name__}:UserPathAttack", "options": options}}
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(spec))
    attack = load_configuration(path).attacks["probe"]
    assert Path(attack.source) == tmp_path / "weights.bin"
    assert Path(attack.sources[0]) == tmp_path / "first.bin"
    assert attack.sources[1:] == [uri, absolute]
    assert attack.untyped == "untyped.bin"
    assert attack.payload == "payload.bin"
    assert attack.settings == options["settings"]
    assert attack.nested.path == "ordinary-string.bin"
    assert Path(attack.nested.source) == tmp_path / "nested.bin"
    assert attack.options == {"image": "extra.bin", "path": "extra-path.bin"}

    spec["attacks"] = {"probe": {
        "type": f"{__name__}:UnresolvedOptionAttack",
        "options": {"source": "valid.bin", "setting": "unchanged.bin"},
    }}
    path.write_text(yaml.safe_dump(spec))
    attack = load_configuration(path).attacks["probe"]
    assert Path(attack.source) == tmp_path / "valid.bin"
    assert attack.setting == "unchanged.bin"

    spec["attacks"] = {"probe": {
        "type": f"{__name__}:UnresolvedPathAttack", "options": {"source": "unknown.bin"},
    }}
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="path annotation") as error:
        load_configuration(path)
    assert "at runtime" in str(error.value)


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
