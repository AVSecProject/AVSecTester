"""The new configuration schema constructs reusable objects without starting driving runs."""

from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType

import pytest
import yaml

from avsectester.attack import Attack
from avsectester.closed_loop.avstack import AVStack
from avsectester.closed_loop.datatypes import Control, StateEstimate
from avsectester.closed_loop.environment import merge_environments
from avsectester.closed_loop.hooks import HookSet
from avsectester.closed_loop.services import ComponentDescriptor, DrivingAdapter
from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator
from avsectester.datasets import Dataset
from avsectester.experiment.config import (
    ExperimentConfiguration, TreatmentConfiguration, load_configuration,
)


@pytest.fixture
def definitions(monkeypatch):
    module = ModuleType("avsec_test_experiment_config")
    module.created = []

    class Data(Dataset):
        def __init__(self, directory=""):
            self.directory = directory

        def scenarios(self):
            return iter(())

        def frames(self, scenario_id):
            return ()

        def frame_data(self, scenario_id, frame):
            raise KeyError(scenario_id)

        def native(self, scenario_id):
            raise KeyError(scenario_id)

        def scenario_start(self, scenario_id, frame):
            raise KeyError(scenario_id)

    class Method(Attack):
        def __init__(self, texture_path=None):
            self.assets = texture_path

        def build_metadata(self, scenario):
            return ()

    class Driver(AVStack):
        def __init__(self, checkpoint_path=None):
            self.checkpoint_path = checkpoint_path
            self.closed = False
            module.created.append(self)

        @property
        def command_modes(self):
            return frozenset({"actuator"})

        def __call__(self, observation):
            return Control()

        def close(self):
            self.closed = True

    class Localizer:
        def __call__(self, value, context):
            return StateEstimate(value.vehicle_state, value.ego_speed)

    class Handler:
        def __call__(self, value, context):
            return value

    def adapter():
        return DrivingAdapter(lambda seed: Driver())

    def bindings(factory, *, components=()):
        return DrivingAdapter(factory, components=tuple(
            (ComponentDescriptor(name, {"value": "input"}, {"value": "output"}, {"value": "none"}),
             lambda avstack, inputs: inputs)
            for name in components
        ))

    module.Data, module.Method, module.Driver = Data, Method, Driver
    module.Localizer, module.Handler, module.adapter = Localizer, Handler, adapter
    module.bindings = bindings
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def basic_config(definitions):
    return {
        "dataset": {"type": definitions.__name__ + ":Data"},
        "simulator": {"type": "RecordedReplaySimulator"},
        "avstack": {"type": definitions.__name__ + ":Driver"},
        "attacks": {"example": {"type": definitions.__name__ + ":Method"}},
    }


def test_python_objects_and_named_treatments(definitions):
    data, simulator, driver, attack = (
        definitions.Data(), RecordedReplaySimulator(), definitions.adapter(), definitions.Method(),
    )
    config = load_configuration({
        "dataset": data, "simulator": simulator, "avstack": driver,
        "attacks": {"example": attack}, "seeds": [3, 5], "decision_steps": 4,
        "treatments": {
            "clean": {}, "noisy": {"attack_name": "example", "environment": {
                "sensors": [{"type": "GaussianNoise", "options": {"severity": 0.2}, "camera": "front"}],
            }},
        },
        "recording": {"sensor_ids": ["front"], "representations": ["raw", "model_input"]},
    })
    assert (config.dataset, config.simulator, config.avstack) == (data, simulator, driver)
    assert config.attacks["example"] is attack
    assert config.seeds == (3, 5)
    assert config.recording.sensor_ids == ("front",)
    assert [(item.name, item.attack_name) for item in config.treatments] == [
        ("clean", None), ("noisy", "example"),
    ]
    assert load_configuration(config) is config


def test_referenced_configs_resolve_paths_from_each_containing_file(tmp_path, definitions):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "driving.yaml").write_text(yaml.safe_dump({
        "type": definitions.__name__ + ":Driver",
        "options": {"checkpoint_path": "../weights/model.pt"},
        "vehicle_io": {"type": "VehicleIO", "options": {
            "localizer": {"type": definitions.__name__ + ":Localizer"},
        }},
        "command_modes": ["actuator"],
    }))
    (configs / "attack.yaml").write_text(yaml.safe_dump({
        "type": definitions.__name__ + ":Method",
        "options": {"texture_path": "assets/texture.png"},
        "environment": {"sensors": [{"type": "GaussianNoise", "options": {"severity": 0.2},
                                      "camera": "front"}]},
    }))
    experiment = basic_config(definitions)
    experiment.update({
        "avstack": "configs/driving.yaml", "attacks": {"example": "configs/attack.yaml"},
        "output": "results", "dataset": {
            "type": definitions.__name__ + ":Data", "options": {"directory": "data"},
        },
    })
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(experiment))
    config = load_configuration(path)
    assert not definitions.created
    assert config.output == tmp_path / "results"
    assert config.dataset.directory == str(tmp_path / "data")
    assert config.attacks["example"].assets == str(configs / "assets/texture.png")
    assert config.attacks["example"].environment["sensors"][0]["options"]["severity"] == 0.2
    first, second = config.avstack.factory(7), config.avstack.factory(11)
    assert first is not second  # A fresh AVStack, and driving history, for every run.
    assert first.checkpoint_path == str(tmp_path / "weights/model.pt")
    assert first.vehicle_io.localizer is not second.vehicle_io.localizer


def test_hooks_construct_fresh_handlers_and_keep_explicit_native_registration(definitions):
    config = basic_config(definitions)
    config["hooks"] = [
        {"stage": "command", "type": definitions.__name__ + ":Handler"},
        {"stage": "perception.post", "imports": [definitions.__name__],
         "native_hook": {"type": "ExampleHook"}},
    ]
    loaded = load_configuration(config)
    first, second = loaded.hooks(3), loaded.hooks(5)
    assert isinstance(first, HookSet)
    assert first.seed == 3 and second.seed == 5
    assert [item.stage for item in first.hooks] == ["command", "perception.post"]
    assert all(a.handler is not b.handler for a, b in zip(first.hooks, second.hooks))
    assert first.hooks[1].handler.config == {"type": "ExampleHook"}


@pytest.mark.parametrize("key,value,match", [
    ("simulator", {"type": "nurec"}, "Unknown built-in type"),
    ("attacks", [], "attacks must map"),
    ("seeds", [True], "nonnegative integers"),
    ("seeds", [1, 1], "duplicates"),
    ("decision_steps", 0, "positive integer"),
    ("strict", "yes", "strict must be boolean"),
    ("treatments", [{"name": "broken", "attack_name": "missing"}], "unknown attack"),
    ("treatments", [{"name": "same"}, {"name": "same"}], "unique"),
    ("treatments", [{"name": "two", "attack_name": ["example", "other"]}], "Treatment attack"),
    ("hooks", [{"stage": "command", "type": "not-a-module"}], "Unknown built-in type"),
    ("hooks", [{"stage": "command", "type": "a:b:c"}], "package.module:name"),
    ("hooks", [{"stage": "command"}], "exactly one of type and native_hook"),
    ("hooks", [{"stage": "perception.post", "native_hook": {"type": "X"}, "options": {}}],
     "without options"),
    ("dataset", {"factory": "package.module:Data"}, "Unknown Dataset keys"),
    ("avstack", {"type": "RecordedReplaySimulator"}, "AVStack class"),
    ("avstack", {"type": "ModularAVStack", "bindings": "names"}, "bindings must be a mapping"),
])
def test_invalid_new_configuration_fails_early(definitions, key, value, match):
    config = basic_config(definitions)
    config[key] = value
    with pytest.raises((ValueError, TypeError), match=match):
        load_configuration(config)


@pytest.mark.parametrize("key", ["backend", "stack", "runtime", "clean_runtime", "selection",
                               "patches", "interventions"])
def test_legacy_experiment_keys_are_not_silently_accepted(definitions, key):
    config = basic_config(definitions)
    config[key] = {}
    with pytest.raises(ValueError, match="Unknown experiment keys"):
        load_configuration(config)


def test_default_treatments_do_not_share_a_reserved_name(definitions):
    config = basic_config(definitions)
    loaded = load_configuration(config)
    assert loaded.treatments == (
        TreatmentConfiguration("clean"), TreatmentConfiguration("example", "example"),
    )
    config["attacks"] = {"clean": definitions.Method()}
    with pytest.raises(ValueError, match="unique"):
        load_configuration(config)


def test_dataclass_configuration_revalidates_replace(definitions):
    config = ExperimentConfiguration(definitions.Data(), RecordedReplaySimulator(),
                                     definitions.adapter(), output=Path("results"))
    assert config.treatments == (TreatmentConfiguration("clean"),)
    with pytest.raises(ValueError, match="positive integer"):
        replace(config, decision_steps=-1)


def test_omitted_sensor_overrides_preserve_base_environment(definitions):
    config = basic_config(definitions)
    noise = {"type": "GaussianNoise", "options": {"severity": 0.2}, "camera": "front"}
    config["environment"] = {"sensors": [noise]}
    config["attacks"]["example"]["environment"] = {"world": {}}
    config["treatments"] = [
        {"name": "clean"}, {"name": "inherited", "attack_name": "example"},
        {"name": "removed", "attack_name": "example", "environment": {"sensors": []}},
    ]
    loaded = load_configuration(config)
    assert loaded.treatments[0].environment == {}
    assert loaded.attacks["example"].environment == {"world": {}}
    for treatment in loaded.treatments[:2]:
        assert merge_environments(loaded.environment, loaded.attacks["example"].environment,
                                  treatment.environment)["sensors"] == [noise]
    assert merge_environments(loaded.environment, loaded.treatments[2].environment)["sensors"] == []


def test_driving_wrapper_releases_created_avstack_when_input_setup_fails(definitions):
    config = basic_config(definitions)

    def broken_io():
        raise RuntimeError("localizer configuration failed")

    definitions.broken_io = broken_io
    config["avstack"]["vehicle_io"] = {"type": definitions.__name__ + ":broken_io"}
    loaded = load_configuration(config)
    with pytest.raises(RuntimeError, match="localizer configuration failed"):
        loaded.avstack.factory(0)
    assert len(definitions.created) == 1 and definitions.created[0].closed


def test_avstack_bindings_expose_components_over_the_configured_factory(definitions):
    config = basic_config(definitions)
    config["avstack"]["bindings"] = {
        "type": definitions.__name__ + ":bindings", "options": {"components": ["policy"]},
    }
    loaded = load_configuration(config)
    assert list(loaded.avstack.component_bindings) == ["policy"]
    first, second = loaded.avstack.factory(0), loaded.avstack.factory(1)
    assert isinstance(first, definitions.Driver) and first is not second


def test_configuration_summary_is_lazy_and_detached_from_callers(definitions):
    class Described(definitions.Method):
        def configuration_parameters(self):
            return {"texture_path": self.assets}

    attack = Described(texture_path="texture.png")
    config = load_configuration({
        "dataset": definitions.Data(), "simulator": RecordedReplaySimulator(),
        "avstack": definitions.adapter(), "attacks": {"example": attack},
    })
    summary = config.configuration_summary
    summary["attacks"]["example"]["parameters"]["texture_path"] = "changed by caller"
    assert config.configuration_summary["attacks"]["example"]["parameters"] == {
        "texture_path": "texture.png",
    }
    attack.assets = "explicitly changed input"
    assert config.configuration_summary["attacks"]["example"]["parameters"] == {
        "texture_path": "explicitly changed input",
    }


def test_configuration_summary_uses_resolved_options_and_discards_stale_constructor_specs(definitions):
    spec = basic_config(definitions)
    spec["attacks"]["example"]["options"] = {"texture_path": "texture.png"}
    config = load_configuration(spec)
    assert config.configuration_summary["attacks"]["example"]["parameters"] == {
        "texture_path": str(Path.cwd() / "texture.png"),
    }
    replaced = replace(config, attacks={"example": definitions.Method("new.png")})
    with pytest.warns(RuntimeWarning, match="parameters cannot be described"):
        summary = replaced.configuration_summary
    assert summary["attacks"]["example"]["parameters"] is None
    assert summary["attacks"]["example"]["parameters_verified"] is False
    copied = replace(config, output=Path("another-output"))
    assert copied.configuration_summary == config.configuration_summary



def test_configuration_does_not_request_attack_parameters_or_enumerate_scenarios(definitions, monkeypatch):
    from inspect import signature
    from unittest.mock import Mock

    data = definitions.Data()
    attack = definitions.Method("texture.png")  # No texture_path attribute or descriptor override.
    scenarios = Mock(side_effect=AssertionError("scenarios should be read only on demand"))
    describe = Mock(side_effect=AssertionError("parameters should be read only on demand"))
    monkeypatch.setattr(data, "scenarios", scenarios)
    monkeypatch.setattr(attack, "configuration_parameters", describe)
    loaded = load_configuration({
        "dataset": data, "simulator": RecordedReplaySimulator(),
        "avstack": definitions.adapter(), "attacks": {"example": attack},
    })
    copied = replace(loaded, output=Path("different-output"))
    assert copied.attacks["example"] is attack
    scenarios.assert_not_called()
    describe.assert_not_called()
    assert not any(name.startswith("_") for name in signature(ExperimentConfiguration).parameters)
