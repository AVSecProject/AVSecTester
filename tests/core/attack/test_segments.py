"""Split eligible decision windows through the public selection helper."""
import pytest

from avsectester.attack.segments import split_segments
from avsectester.datasets import FrameReference


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
