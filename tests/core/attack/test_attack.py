"""Public attack contracts, segment selection and placement specifications."""

import numpy as np
import pytest

from avsectester.attack import Attack, AttackSegment, OfflineContext
from avsectester.attack.placement import (
    AttachedPlacement, Orientation, WorldPlacement, orientation_from_spec, placement_from_spec,
)
from avsectester.attack.segments import split_segments
from avsectester.closed_loop.services import ClosedLoopServices
from avsectester.datasets import FrameReference
from tests.fakes import MemoryDataset, MemorySimulator


def test_attack(tmp_path):
    class SelectionOnlyAttack(Attack):
        def build_metadata(self, scenario):
            return split_segments(scenario.frames, lambda frame: True, decision_steps=2)

    attack = SelectionOnlyAttack()
    attack.assets = {"image": np.full((2, 3, 4), 120, np.uint8)}
    with ClosedLoopServices(MemoryDataset(), MemorySimulator()) as services:
        metadata = attack.build_metadata(services.scenario("scene", seed=0))
        artifacts = attack.run_offline(OfflineContext(services, metadata))
    assert metadata and metadata[0].decision_steps == 2
    np.testing.assert_array_equal(artifacts["image"], attack.assets["image"])
    attack.save_artifacts(artifacts, tmp_path)
    restored = attack.load_artifacts(tmp_path)
    np.testing.assert_array_equal(restored["image"], artifacts["image"])
    (tmp_path / "artifacts.pkl").write_bytes(b"incomplete artifact")
    with pytest.raises(ValueError, match="hash"):
        attack.load_artifacts(tmp_path)


def test_attack_segment():
    start = FrameReference("scene", 2, "frame-2", .2)
    stop = FrameReference("scene", 5, "frame-5", .5)
    segment = AttackSegment("scene", start, stop, roles={"host": ("lead",)},
                           placement=AttachedPlacement("lead", "rear_center"),
                           selection_score=.8, selection_rank=1, user_data={"label": "visible"})
    assert segment.decision_steps == 3
    assert segment.roles == {"host": ("lead",)}
    assert segment.selection_score == .8 and segment.selection_rank == 1
    with pytest.raises(ValueError, match="stop"):
        AttackSegment("scene", stop, start)


def test_split_segments():
    frames = tuple(FrameReference("scene", i, i, i * .1) for i in range(80))
    eligible = lambda frame: frame.position >= 5
    full = split_segments(frames, eligible, decision_steps=20)
    assert [(segment.start.position, segment.stop.position) for segment in full] == [
        (5, 25), (25, 45), (45, 65),
    ]
    with_tail = split_segments(frames, eligible, decision_steps=20, keep_tail=True)
    assert [segment.decision_steps for segment in with_tail] == [20, 20, 20, 14]
    separated = split_segments(frames[:8], lambda frame: frame.position != 2,
                               decision_steps=2, keep_tail=True)
    assert [(segment.start.position, segment.stop.position) for segment in separated] == [
        (0, 2), (3, 5), (5, 7),
    ]
    with pytest.raises(ValueError, match="positive integer"):
        split_segments(frames, eligible, decision_steps=0)


def test_placement_from_spec():
    attached = placement_from_spec({"type": "AttachedPlacement", "options": {
        "host": "lead", "anchor": "rear_center", "offset_m": [-.1, 0., .2],
    }})
    np.testing.assert_allclose(attached.local_position((4., 2., 1.5)), [-2.1, 0., .2])
    world = placement_from_spec({"type": "WorldPlacement", "options": {"position_m": [8., 1., 0.]}})
    assert world == WorldPlacement((8., 1., 0.))
    assert orientation_from_spec({"type": "Orientation", "options": {"mode": "face_ego"}}) == Orientation("face_ego")
    with pytest.raises(ValueError, match="Unknown placement"):
        placement_from_spec({"type": "UnknownPlacement"})
