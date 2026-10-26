"""Public attack contracts and selected intervals."""

import numpy as np
import pytest

from avsectester.attack import Attack, AttackSegment, OfflineContext
from avsectester.attack.segments import split_segments
from avsectester.execution.services import ClosedLoopServices
from avsectester.datasets import FrameReference
from tests.fakes import MemoryDataset, MemorySimulator


def test_attack(tmp_path):
    class SelectionOnlyAttack(Attack):
        def build_metadata(self, scenario):
            return split_segments(scenario.frames, lambda frame: True, decision_steps=2)

    attack = SelectionOnlyAttack()
    with ClosedLoopServices(MemoryDataset(), MemorySimulator()) as services:
        metadata = attack.build_metadata(services.scenario("scene", seed=0))
        artifacts = attack.run_offline(OfflineContext(services, metadata))
    assert metadata and metadata[0].decision_steps == 2
    assert artifacts is None
    supplied = Attack()
    supplied.validate_segment(metadata[0])
    with pytest.raises(NotImplementedError, match="metadata.load"):
        supplied.build_metadata(None)
    artifacts = {"image": np.full((2, 3, 4), 120, np.uint8)}
    attack.save_artifacts(artifacts, tmp_path)
    restored = attack.load_artifacts(tmp_path)
    np.testing.assert_array_equal(restored["image"], artifacts["image"])
    (tmp_path / "artifacts.pkl").write_bytes(b"incomplete artifact")
    with pytest.raises(ValueError, match="hash"):
        attack.load_artifacts(tmp_path)


def test_attack_segment():
    start = FrameReference("scene", 2, "frame-2", .2)
    stop = FrameReference("scene", 5, "frame-5", .5)
    segment = AttackSegment("scene", start, stop, user_data={"host": "lead", "label": "visible"})
    assert segment.decision_steps == 3
    assert segment.user_data == {"host": "lead", "label": "visible"}
    with pytest.raises(ValueError, match="stop"):
        AttackSegment("scene", stop, start)
