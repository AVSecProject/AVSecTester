"""Metadata JSON preserves supported records and rejects values it cannot reload."""

import json

import numpy as np
import pytest

from avsectester.attack import AttackSegment
from avsectester.datasets import ActorState, FrameReference, SensorDataReference
from avsectester.experiment.persistence import _decode, _encode
from avsectester.geometry import ActorPose
from avsectester.attack.placement import AttachedPlacement


def test_metadata_json_round_trip_preserves_roles_placement_and_array_shapes():
    start, stop = (FrameReference("s", i, str(i), i * 0.1) for i in (1, 3))
    segment = AttackSegment(
        "s", start, stop, {"hosts": ("a", "b")},
        AttachedPlacement("a", anchor="rear_center"), selection_score=0.75, selection_rank=0,
        user_data={
            "empty": np.empty((0, 4), dtype=np.float32),
            "scalar": np.asarray(2, dtype=np.int16),
            "pose": ActorPose(np.eye(4), (4, 2, 1.5)),
            "state": ActorState(np.eye(4), (1, 0, 0), "vehicle", (4, 2, 1.5)),
            "flags": [True, None, "surface"],
        },
    )
    restored = _decode(json.loads(json.dumps(_encode(segment), allow_nan=False)))
    assert restored.roles == segment.roles and restored.placement == segment.placement
    assert (restored.selection_score, restored.selection_rank) == (0.75, 0)
    for name in ("empty", "scalar"):
        expected, actual = segment.user_data[name], restored.user_data[name]
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(restored.user_data["pose"].transform, np.eye(4))
    assert restored.user_data["state"].extent == (4, 2, 1.5)
    assert restored.user_data["flags"] == [True, None, "surface"]


def test_metadata_rejects_unsupported_records_and_arrays_before_saving():
    # A framework dataclass is not automatically a supported, restorable metadata record.
    with pytest.raises(TypeError, match="Unsupported metadata record"):
        _encode(SensorDataReference("camera.png", "front", 0))
    for array in (np.array([object()]), np.zeros(1, dtype=[("field", "f4")]),
                  np.array([1 + 2j])):
        with pytest.raises(TypeError, match="Metadata arrays"):
            _encode(array)
    with pytest.raises(ValueError, match="Unsupported metadata record"):
        _decode({"__type__": "record", "class": "os:system", "fields": {}})


@pytest.fixture
def preparation_configuration():
    from avsectester.attack import Attack
    from avsectester.closed_loop.services import DrivingAdapter
    from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator
    from avsectester.datasets import Dataset, ScenarioDescriptor
    from avsectester.experiment import ExperimentConfiguration

    class Data(Dataset):
        def __init__(self, names=("scene",)):
            self.names = names

        def scenarios(self):
            return iter(ScenarioDescriptor(name, "NuScenesDataset") for name in self.names)

        def frames(self, scenario_id):
            return tuple(FrameReference(scenario_id, i, i, i * .1) for i in range(3))

        def frame_data(self, scenario_id, frame):
            raise NotImplementedError

        def native(self, scenario_id):
            raise NotImplementedError

        def scenario_start(self, scenario_id, frame):
            raise NotImplementedError

    class ConfiguredAttack(Attack):
        def __init__(self, strength=.2):
            self.strength = strength
            self.calls = 0

        def build_metadata(self, view):
            self.calls += 1
            return ()

    return ExperimentConfiguration(Data(), RecordedReplaySimulator(), DrivingAdapter(lambda seed: None),
                                   attacks={"example": ConfiguredAttack()})


def _prepared(configuration):
    from avsectester.experiment import PreparedExperiment, SelectionDecision

    refs = configuration.dataset.frames("scene")
    return PreparedExperiment(
        configuration, {"example": (AttackSegment("scene", refs[0], refs[2]),)},
        {"example": {"weights": np.arange(3)}},
        (SelectionDecision("example", "scene", "selected"),),
    )


def test_prepared_save_and_load_share_one_class_without_mutating_attack(tmp_path, preparation_configuration):
    from avsectester.experiment import PreparedExperiment

    configuration = preparation_configuration
    prepared = _prepared(configuration)
    attack = configuration.attacks["example"]
    # Run state changes after configuration creation do not change preparation identity.
    attack.calls = 10
    before = dict(vars(attack))
    prepared.save(tmp_path)
    manifest = json.loads((tmp_path / "prepared.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["configuration"]["attacks"]["example"]["parameters"] == {"strength": .2}
    assert manifest["selections"][0]["attack_name"] == "example"
    restored = PreparedExperiment.load(configuration, tmp_path)
    assert restored.metadata == prepared.metadata
    assert restored.selections == prepared.selections
    np.testing.assert_array_equal(restored.artifacts["example"]["weights"], np.arange(3))
    assert vars(attack) == before and not hasattr(attack, "artifacts")


@pytest.mark.parametrize("changed", ["dataset_type", "scenarios", "simulator", "attack_type", "parameters"])
def test_saved_preparation_rejects_changed_configuration(tmp_path, preparation_configuration, changed):
    from dataclasses import replace

    from avsectester.experiment import PreparedExperiment

    original = preparation_configuration
    _prepared(original).save(tmp_path)
    if changed == "dataset_type":
        class Other(type(original.dataset)):
            pass

        incompatible = replace(original, dataset=Other())
    elif changed == "scenarios":
        incompatible = replace(original, dataset=type(original.dataset)(("different",)))
    elif changed == "simulator":
        simulator = type(original.simulator)()
        simulator.simulator_version = "different"
        incompatible = replace(original, simulator=simulator)
    elif changed == "attack_type":
        class OtherAttack(type(original.attacks["example"])):
            pass

        incompatible = replace(original, attacks={"example": OtherAttack()})
    else:
        incompatible = replace(original, attacks={"example": type(original.attacks["example"])(.9)})
    with pytest.raises(ValueError, match="configuration differs"):
        PreparedExperiment.load(incompatible, tmp_path)


def test_loading_allows_different_output_recording_and_runtime_counters(tmp_path, preparation_configuration):
    from dataclasses import replace

    from avsectester.closed_loop.runner import RecordingPolicy
    from avsectester.experiment import PreparedExperiment

    original = preparation_configuration
    _prepared(original).save(tmp_path)
    # A fresh object with identical constructor parameters is compatible, regardless of counters.
    fresh = type(original.attacks["example"])()
    fresh.calls = 100
    changed = replace(original, attacks={"example": fresh}, output=tmp_path / "another-output",
                      recording=RecordingPolicy(every_n_steps=7))
    restored = PreparedExperiment.load(changed, tmp_path)
    assert restored.configuration.output == tmp_path / "another-output"
    assert restored.configuration.recording.every_n_steps == 7
    assert fresh.calls == 100


def test_undescribed_parameters_warn_on_save_and_cannot_silently_pass_loading(
    tmp_path, preparation_configuration,
):
    from dataclasses import replace
    from unittest.mock import Mock

    from avsectester.experiment import PreparedExperiment

    class Hidden(type(preparation_configuration.attacks["example"])):
        def __init__(self, texture_path):
            self.assets = texture_path

    attack = Hidden("texture.png")
    config = replace(preparation_configuration, attacks={"example": attack})
    before = dict(vars(attack))
    with pytest.warns(RuntimeWarning, match="parameters cannot be described"):
        _prepared(config).save(tmp_path)
    manifest = json.loads((tmp_path / "prepared.json").read_text())
    description = manifest["configuration"]["attacks"]["example"]
    assert description["type"] == f"{Hidden.__module__}:{Hidden.__qualname__}"
    assert description["parameters"] is None and description["parameters_verified"] is False
    assert (tmp_path / "attacks/example/artifacts.pkl").is_file()
    assert vars(attack) == before
    attack.load_artifacts = Mock(side_effect=AssertionError("unvalidated artifacts must not load"))
    with pytest.raises(ValueError, match="Cannot validate preparation parameters.*example"):
        PreparedExperiment.load(config, tmp_path)
    attack.load_artifacts.assert_not_called()


def test_current_configuration_must_describe_parameters_when_loading(
    tmp_path, preparation_configuration, monkeypatch,
):
    from avsectester.experiment import PreparedExperiment

    _prepared(preparation_configuration).save(tmp_path)
    def hidden():
        raise TypeError("Hidden parameters")
    monkeypatch.setattr(preparation_configuration.attacks["example"], "configuration_parameters", hidden)
    with pytest.warns(RuntimeWarning, match="parameters cannot be described"):
        with pytest.raises(ValueError, match="Cannot validate preparation parameters.*example"):
            PreparedExperiment.load(preparation_configuration, tmp_path)


def test_scenario_enumeration_is_deferred_until_save_or_load(
    tmp_path, preparation_configuration, monkeypatch,
):
    from dataclasses import replace
    from unittest.mock import Mock

    from avsectester.experiment import PreparedExperiment

    dataset = preparation_configuration.dataset
    scenarios = Mock(wraps=dataset.scenarios)
    monkeypatch.setattr(dataset, "scenarios", scenarios)
    config = replace(preparation_configuration, output=tmp_path)
    scenarios.assert_not_called()
    prepared = _prepared(config)
    scenarios.assert_not_called()
    prepared.save(tmp_path)
    assert scenarios.call_count == 1
    PreparedExperiment.load(config, tmp_path)
    assert scenarios.call_count == 2
