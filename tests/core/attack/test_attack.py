"""Public attack contracts and selected intervals."""

import numpy as np
import pytest

from avsectester.attack import Attack, AttackSegment, OfflineContext, OnlineContext
from avsectester.attack.segments import split_segments
from avsectester.execution.services import ClosedLoopServices
from avsectester.datasets import FrameReference
from tests.fakes import MemoryDataset, MemorySimulator


def test_save_and_load_artifacts(tmp_path):
    attack = Attack()
    artifacts = {"image": np.full((2, 3, 4), 120, np.uint8)}
    attack.save_artifacts(artifacts, tmp_path)
    restored = attack.load_artifacts(tmp_path)
    np.testing.assert_array_equal(restored["image"], artifacts["image"])
    (tmp_path / "artifacts.pkl").write_bytes(b"incomplete artifact")
    with pytest.raises(ValueError, match="hash"):
        attack.load_artifacts(tmp_path)


def test_default_phases():
    class SelectionOnlyAttack(Attack):
        def __init__(self, decisions=2):
            self.decisions = decisions

        def build_metadata(self, scenario):
            return split_segments(scenario.frames, lambda frame: True, decision_steps=self.decisions)

    attack = SelectionOnlyAttack()
    with ClosedLoopServices(MemoryDataset(), MemorySimulator()) as services:
        metadata = attack.build_metadata(services.scenario("scene", seed=0))
        assert attack.run_offline(OfflineContext(services, metadata)) is None
    assert metadata[0].decision_steps == 2
    default = Attack()
    default.validate_segment(metadata[0])
    assert default.deployment_ticks(metadata[0]) == 0
    runtime = default.create_runtime(metadata[0], None)
    context = OnlineContext(0, "setup", 0., 0)
    assert runtime.setup(context) == () and runtime.step(context) == ()
    assert runtime.stop(context) is None
    with pytest.raises(NotImplementedError, match="metadata.load"):
        default.build_metadata(None)


def test_attack_segment():
    start = FrameReference("scene", 2, "frame-2", .2)
    stop = FrameReference("scene", 5, "frame-5", .5)
    segment = AttackSegment("scene", start, stop, user_data={"host": "lead", "label": "visible"})
    assert segment.decision_steps == 3
    assert segment.user_data == {"host": "lead", "label": "visible"}
    with pytest.raises(ValueError, match="stop"):
        AttackSegment("scene", stop, start)


def test_configuration_parameters():
    class ConfiguredAttack(Attack):
        def __init__(self, count=2):
            self.count = count

    assert ConfiguredAttack().configuration_parameters() == {"count": 2}

    class VariadicAttack(Attack):
        def __init__(self, **options):
            pass

    class HiddenParameterAttack(Attack):
        def __init__(self, value=1):
            pass

    with pytest.raises(TypeError, match="variadic"):
        VariadicAttack().configuration_parameters()
    with pytest.raises(TypeError, match="no matching attribute"):
        HiddenParameterAttack().configuration_parameters()
