"""The user's attack class: one metabuilder and two phases with trivial defaults."""

import pytest
import numpy as np

from avsectester.attack import Attack, AttackSegment, OfflineContext
from avsectester.attack.runtime import AttackRuntime
from avsectester.datasets import FrameReference


class Phantom(Attack):
    def build_metadata(self, scenario):
        return [scenario]


def test_offline_and_online_defaults_are_trivial():
    attack = Phantom()
    assert attack.run_offline(OfflineContext(services=None, metadata=("segment",))) is None
    attack.assets = {"patch": "texture"}
    assert attack.run_offline(OfflineContext(None, ())) == {"patch": "texture"}
    runtime = attack.create_runtime("segment", attack.assets)
    assert type(runtime) is AttackRuntime
    assert runtime.setup(None) == () and runtime.step(None) == ()


def test_every_attack_builds_its_metadata():
    with pytest.raises(TypeError, match="build_metadata"):
        Attack()


def test_segment_selection_fields_are_optional_validated_and_detached():
    start, stop = (FrameReference("scene", index, index, float(index)) for index in (0, 2))
    original = {"hosts": ["a", "b"]}
    data = {"surface": {"offset": [0, 1, 2]}}
    segment = AttackSegment("scene", start, stop, original, selection_score=0.7,
                           selection_rank=0, user_data=data)
    original["hosts"].clear()
    data["surface"]["offset"].clear()
    assert segment.roles == {"hosts": ("a", "b")}
    assert segment.user_data["surface"]["offset"] == [0, 1, 2]
    assert segment.decision_steps == 2
    assert AttackSegment("scene", start, stop).selection_score is None
    for score in (True, "good", float("nan"), float("inf")):
        with pytest.raises(ValueError, match="selection_score"):
            AttackSegment("scene", start, stop, selection_score=score)
    for rank in (True, -1, 0.5):
        with pytest.raises(ValueError, match="selection_rank"):
            AttackSegment("scene", start, stop, selection_rank=rank)
    with pytest.raises(ValueError, match="actor IDs"):
        AttackSegment("scene", start, stop, roles={"hosts": iter(("a", ""))})


def test_attack_artifacts_round_trip_does_not_change_attack_state(tmp_path):
    attack = Phantom()
    assert attack.environment == {} and not hasattr(attack, "artifacts")
    with pytest.raises(TypeError):
        attack.environment["world"] = {"weather": "rain"}
    attack.environment = {"world": {"weather": "rain"}}
    assert Phantom().environment == {}
    original = {"texture": np.arange(24, dtype=np.uint8).reshape(2, 3, 4),
                        "options": {"threshold": 0.3}}
    attack.save_artifacts(original, tmp_path)
    restored = Phantom()
    artifacts = restored.load_artifacts(tmp_path)
    assert not hasattr(restored, "artifacts") and not hasattr(attack, "artifacts")
    np.testing.assert_array_equal(artifacts["texture"], original["texture"])
    assert artifacts["options"] == original["options"]
    payload = tmp_path / "artifacts.pkl"
    payload.write_bytes(payload.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        restored.load_artifacts(tmp_path)


def test_configuration_parameters_excludes_runtime_state_and_requires_explicit_hidden_parameters():
    class Configured(Phantom):
        def __init__(self, decision_steps):
            self.decision_steps = decision_steps
            self.calls = 0

    attack = Configured(3)
    attack.calls += 1
    assert attack.configuration_parameters() == {"decision_steps": 3}

    class Hidden(Phantom):
        def __init__(self, threshold):
            self._threshold = threshold

    with pytest.raises(TypeError, match="configuration_parameters.*threshold"):
        Hidden(.2).configuration_parameters()

    class Explicit(Hidden):
        def configuration_parameters(self):
            return {"threshold": self._threshold}

    assert Explicit(.3).configuration_parameters() == {"threshold": .3}
