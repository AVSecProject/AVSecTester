"""Configuration is loaded through its public constructors and validation API."""

from collections.abc import Sequence
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import yaml

from avsectester.attack import Attack
from avsectester.experiment import MetadataConfiguration, SplitConfiguration, load_configuration
from avsectester.experiment.engine import validate_configuration_environment
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator, MinimalAttack, MetadataOnlyAttack

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


class DeviceOptions:
    """A user component with an explicit execution device."""

    def __init__(self, device="fallback"):
        self.device = device


class DeviceAttack(Attack):
    def __init__(self, device="fallback", nested=None):
        self.device = device
        self.nested = nested

    def build_metadata(self, scenario):
        return ()


class DeviceAVStack(ConstantAVStack):
    def __init__(self, device="fallback"):
        super().__init__()
        self.device = device


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
    with pytest.raises(ValueError, match="at least one attack"):
        load_configuration(missing)
    several = specification()
    several["attacks"]["other"] = {"type": "tests.fakes:MinimalAttack"}
    assert set(load_configuration(several).attacks) == {"probe", "other"}


def test_configuration_device():
    spec = specification()
    spec["device"] = "cpu"
    spec["avstack"] = {"type": f"{__name__}:DeviceAVStack"}
    spec["attacks"] = {"probe": {"type": f"{__name__}:DeviceAttack", "options": {
        "nested": {
            "inherited": {"type": f"{__name__}:DeviceOptions"},
            "explicit": {"type": f"{__name__}:DeviceOptions", "options": {"device": "cuda:1"}},
        },
    }}}
    configuration = load_configuration(spec)
    attack = configuration.attacks["probe"]
    assert attack.device == "cpu"
    assert attack.nested["inherited"].device == "cpu"
    assert attack.nested["explicit"].device == "cuda:1"
    assert configuration.avstack.factory(0).device == "cpu"
    assert isinstance(configuration.dataset, MemoryDataset)
    assert isinstance(configuration.simulator, MemorySimulator)

    explicit = deepcopy(spec)
    explicit.pop("device")
    explicit["avstack"]["options"] = {"device": "cpu"}
    explicit["attacks"]["probe"]["options"]["device"] = "cpu"
    explicit["attacks"]["probe"]["options"]["nested"]["inherited"]["options"] = {"device": "cpu"}
    assert configuration.configuration_summary == load_configuration(explicit).configuration_summary
    assert load_configuration({**spec, "device": "cuda:2"}).attacks["probe"].nested["explicit"].device == "cuda:1"
    assert load_configuration({**spec, "attacks": {"probe": attack}}).attacks["probe"] is attack
    without = deepcopy(spec)
    without.pop("device")
    assert load_configuration(without).attacks["probe"].device == "fallback"
    with pytest.raises(TypeError, match="device.*string"):
        load_configuration({**spec, "device": 0})
    with pytest.raises(ValueError, match="device.*nonempty"):
        load_configuration({**spec, "device": ""})


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

    # Local probe files resolve beside the YAML; server paths stay in the server namespace.
    directory = tmp_path / "configs"
    directory.mkdir()
    probe_spec = specification()
    probe_spec["simulator"] = {
        "type": "NuRecSimulator",
        "options": {
            "depth": "none", "physics_update_mode": "none",
            "route_generator": "none",
            "renderer": {
                "type": "avsectester.closed_loop.simulators.nurec:NuRecRenderer",
                "options": {"probe_directory": "../results/nre-probes",
                            "server_probe_directory": "../server-probes"},
            },
        },
    }
    probe_config = directory / "nurec.yaml"
    probe_config.write_text(yaml.safe_dump(probe_spec))
    renderer = load_configuration(probe_config).simulator.renderer
    assert Path(renderer.probe_directory) == (tmp_path / "results/nre-probes").resolve()
    assert renderer.server_probe_directory == "../server-probes"


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


def test_split_configuration():
    spec = specification()
    assert load_configuration(spec).split is None
    spec["split"] = {"train_segments": 2}
    assert load_configuration(spec).split == SplitConfiguration(2)
    spec["split"] = {"train_segments": 2, "validation_segments": 1, "seed": 7,
                     "execute": ["train", "validation", "test"]}
    assert load_configuration(spec).split == SplitConfiguration(2, 1, 7, ("train", "validation", "test"))
    invalid = (
        ({}, ValueError, "train_segments"),
        ({"train_segments": 0}, ValueError, "train_segments"),
        ({"train_segments": True}, ValueError, "train_segments"),
        ({"train_segments": 1, "validation_segments": -1}, ValueError, "validation_segments"),
        ({"train_segments": 1, "seed": -1}, ValueError, "seed"),
        ({"train_segments": 1, "execute": ["test", "test"]}, ValueError, "duplicates"),
        ({"train_segments": 1, "execute": ["invalid"]}, ValueError, "accepts"),
        ({"train_segments": 1, "execute": ["validation"]}, ValueError, "zero"),
        ({"train_segments": 1, "execute": "test"}, TypeError, "sequence"),
        ({"train_segments": 1, "extra": 0}, ValueError, "Unknown split keys"),
    )
    for value, error, message in invalid:
        spec["split"] = value
        with pytest.raises(error, match=message):
            load_configuration(spec)


def test_metadata_configuration(tmp_path):
    spec = specification()
    assert load_configuration(spec).metadata == MetadataConfiguration()
    spec["metadata"] = {"save": "../metadata/selected.json"}
    directory = tmp_path / "configs"
    directory.mkdir()
    path = directory / "experiment.yaml"
    path.write_text(yaml.safe_dump(spec))
    configuration = load_configuration(path)
    destination = tmp_path / "metadata" / "selected.json"
    assert configuration.metadata == MetadataConfiguration(save=destination)
    assert replace(configuration, metadata=MetadataConfiguration(load=destination)).metadata.load == destination
    spec["metadata"] = {"load": "../metadata/selected.json"}
    path.write_text(yaml.safe_dump(spec))
    assert load_configuration(path).metadata.load == destination
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_configuration({**spec, "metadata": {"save": "first.json", "load": "second.json"}})
    with pytest.raises(ValueError, match="Unknown metadata keys"):
        load_configuration({**spec, "metadata": {"attack": "first.json"}})
    with pytest.raises(TypeError, match="filesystem path"):
        load_configuration({**spec, "metadata": {"load": 7}})

    # Missing selectors are configuration errors before any experiment preparation.
    without_selector = specification()
    without_selector["attacks"] = {"supplied": {"type": "tests.fakes:MetadataOnlyAttack"}}
    with pytest.raises(ValueError, match="metadata.load.*supplied|supplied.*metadata.load"):
        load_configuration(without_selector)
    supplied = replace(configuration, attacks={"probe": MetadataOnlyAttack()})
    with pytest.raises(ValueError, match="metadata.load.*probe|probe.*metadata.load"):
        load_configuration(supplied)
    supplied = replace(supplied, metadata=MetadataConfiguration(load=destination))
    assert load_configuration(supplied) is supplied
    without_selector["metadata"] = {"load": str(destination)}
    assert isinstance(load_configuration(without_selector).attacks["supplied"], MetadataOnlyAttack)


class TimeStepOptions:
    def __init__(self, time_step=7.):
        self.time_step = time_step


def test_decision_time_step():
    spec = specification()
    spec["dataset"]["options"] = {"time_step": .2}
    spec["simulator"]["options"] = {"extra_sensors": {
        "nested": {"type": f"{__name__}:TimeStepOptions"},
    }}
    configuration = load_configuration(spec)
    assert configuration.simulator.time_step == .2
    assert configuration.simulator.extra_sensors["nested"].time_step == 7.
    spec["simulator"]["options"]["time_step"] = .1
    with pytest.raises(ValueError, match="dataset.time_step"):
        load_configuration(spec)
    with pytest.raises(ValueError, match="dataset.time_step"):
        replace(configuration, simulator=MemorySimulator(time_step=.1))
