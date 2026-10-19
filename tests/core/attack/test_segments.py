"""split_segments preserves decision boundaries and real terminal frames."""

from dataclasses import replace

import numpy as np
import pytest

from avsectester.attack import split_segments
from avsectester.datasets import FrameReference


def frames(count):
    return tuple(FrameReference("scene", index, index + 100, index * 0.1)
                 for index in range(count))


def intervals(segments):
    return [(segment.start.position, segment.stop.position) for segment in segments]


def test_eighty_frames_split_into_full_windows_and_short_tail():
    timeline = frames(80)
    calls = []

    def eligible(frame):
        calls.append(frame.position)
        return frame.position >= 5

    segments = split_segments(timeline, eligible, decision_steps=20, keep_tail=True)
    assert intervals(segments) == [(5, 25), (25, 45), (45, 65), (65, 79)]
    assert segments[-1].decision_steps == 14
    assert calls == list(range(79))  # Every frame except the terminal frame, in order.
    assert all(left.stop == right.start for left, right in zip(segments, segments[1:]))
    assert intervals(split_segments(timeline, eligible, decision_steps=20)) == [
        (5, 25), (25, 45), (45, 65),
    ]


def test_ineligible_frame_breaks_a_run_and_terminal_frame_is_not_a_decision():
    timeline = frames(10)
    flags = [True, True, True, False, True, True, True, True, True]
    segments = split_segments(timeline, lambda frame: flags[frame.position],
                              decision_steps=3, keep_tail=True)
    assert intervals(segments) == [(0, 3), (4, 7), (7, 9)]
    numpy_flags = np.asarray(flags)
    assert intervals(split_segments(timeline, lambda frame: numpy_flags[frame.position],
                                    decision_steps=3, keep_tail=True)) == intervals(segments)


@pytest.mark.parametrize("timeline", [(), frames(1)])
def test_no_decision_without_a_terminal_frame(timeline):
    assert split_segments(timeline, lambda _: True, decision_steps=1, keep_tail=True) == ()


@pytest.mark.parametrize("arguments,match", [
    ({"decision_steps": 0}, "positive integer"),
    ({"decision_steps": True}, "positive integer"),
    ({"decision_steps": 1, "keep_tail": 1}, "keep_tail must be boolean"),
])
def test_explicit_window_parameters_are_validated(arguments, match):
    with pytest.raises((ValueError, TypeError), match=match):
        split_segments(frames(3), lambda _: True, **arguments)


def test_malformed_eligibility_and_discontinuous_timelines_are_rejected():
    timeline = frames(3)
    with pytest.raises(TypeError, match="eligible must be a function"):
        split_segments(timeline, [True, True], decision_steps=1)
    with pytest.raises(TypeError, match="must return a boolean"):
        split_segments(timeline, lambda frame: frame.position, decision_steps=1)
    for malformed in ((timeline[0], timeline[2]),
                      (timeline[0], replace(timeline[1], scenario_id="other"))):
        with pytest.raises(ValueError, match="consecutive and ordered"):
            split_segments(malformed, lambda _: True, decision_steps=1)
