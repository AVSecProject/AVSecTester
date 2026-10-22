"""Split a scenario timeline into attack segments for an attack's own ``build_metadata``.

The helper only cuts segments. The attack decides which frames qualify through ``eligible``,
which can read anything available to the metabuilder, for example ``ScenarioView.frame_data``
or closed-loop sensor data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from avsectester.attack.attack import AttackSegment
from avsectester.datasets import FrameReference
from avsectester.validation import require_positive_integer


def split_segments(
    frames: Sequence[FrameReference],
    eligible: Callable[[FrameReference], bool],
    *,
    decision_steps: int,
    keep_tail: bool = False,
) -> tuple[AttackSegment, ...]:
    """Cut each run of consecutive eligible decision frames into segments of ``decision_steps``.

    ``frames`` is an ordered, consecutive timeline of one scenario, normally
    ``ScenarioView.frames``, and includes the terminal frame. ``eligible(frame)`` is called once
    for every frame except the last, in order, and returns whether an attack decision may be
    made at that frame. The last frame is never a decision frame; it only ends a segment.

    A segment covers ``decision_steps`` consecutive eligible decision frames and stops at the
    frame after them, so ``segment.decision_steps == decision_steps``. A frame that is not
    eligible ends the current run; no segment crosses it. With ``keep_tail``, the remainder of a
    run that is shorter than ``decision_steps`` becomes one shorter segment instead of being
    dropped.
    """
    validate_segment_options(decision_steps, None, keep_tail)
    if not callable(eligible):
        raise TypeError("eligible must be a function of one FrameReference returning bool")
    frames = tuple(frames)
    if any(not isinstance(frame, FrameReference) for frame in frames):
        raise TypeError("frames must contain FrameReference values")
    if frames and any(frame.scenario_id != frames[0].scenario_id
                      or frame.position != frames[0].position + index
                      for index, frame in enumerate(frames)):
        raise ValueError("frames must be consecutive and ordered within one scenario")
    flags = tuple(eligible(frame) for frame in frames[:-1])
    if any(not isinstance(value, (bool, np.bool_)) for value in flags):
        raise TypeError("eligible must return a boolean for every frame")
    result = []
    run_start = 0
    for index in range(len(flags) + 1):
        if index < len(flags) and flags[index]:
            continue
        # frames[run_start:index] are eligible decision frames; frames[index] ends the run.
        cursor = run_start
        while cursor + decision_steps <= index:
            result.append(AttackSegment(frames[cursor].scenario_id, frames[cursor],
                                        frames[cursor + decision_steps]))
            cursor += decision_steps
        if keep_tail and cursor < index:
            result.append(AttackSegment(frames[cursor].scenario_id, frames[cursor], frames[index]))
        run_start = index + 1
    return tuple(result)


def validate_segment_options(decision_steps: int, max_segments: int | None,
                             keep_tail: bool) -> None:
    """Validate user segment-length and count parameters before any preparation starts."""
    require_positive_integer(decision_steps, "decision_steps")
    if max_segments is not None:
        require_positive_integer(max_segments, "max_segments")
    if not isinstance(keep_tail, bool):
        raise TypeError("keep_tail must be boolean")
