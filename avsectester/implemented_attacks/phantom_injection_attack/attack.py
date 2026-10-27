"""Fabricate a box in an AVStack detector's declared source frame."""
from __future__ import annotations

import math

from avsectester.attack.attack import Attack
from avsectester.attack.segments import validate_segment_options
from .metabuilder import build_metadata
from .online import PhantomInjectionAttackRuntime


class PhantomInjectionAttack(Attack):
    """Append one AVStack BoxDetection, including to empty detection containers.

    ``target_xyz`` is the box center in the perception output's ``source_reference``
    (x forward, y left, z up), independent of the frame of any existing boxes.
    ``extent`` is full (length, width, height), in meters. The default center keeps
    the default 1.6 m tall box's bottom at source-frame z=-1.5 m.
    """

    def __init__(self, *, decision_steps: int, max_segments: int | None = None,
                 target_xyz=(6.0, 0.0, -0.7), object_type="Car", score=0.9,
                 extent=(4.0, 1.8, 1.6), object_id=90002, keep_tail: bool = False):
        validate_segment_options(decision_steps, max_segments, keep_tail)
        self.decision_steps, self.max_segments, self.keep_tail = decision_steps, max_segments, keep_tail
        self.target_xyz = tuple(float(value) for value in target_xyz)
        self.object_type, self.score = object_type, float(score)
        self.extent, self.object_id = tuple(float(value) for value in extent), object_id
        if (len(self.target_xyz) != 3 or len(self.extent) != 3
                or not all(math.isfinite(value) for value in (*self.target_xyz, *self.extent, self.score))
                or any(value <= 0 for value in self.extent) or not 0 <= self.score <= 1):
            raise ValueError("Phantom position/extent must be finite, extent positive and score in [0,1]")
        if not isinstance(object_type, str) or not object_type:
            raise ValueError("object_type must be a nonempty category name")
        if isinstance(object_id, bool) or not isinstance(object_id, int):
            raise TypeError("object_id must be an integer")

    def build_metadata(self, scenario):
        return build_metadata(self, scenario)

    def create_runtime(self, segment, artifacts):
        return PhantomInjectionAttackRuntime(self)


__all__ = [
    'PhantomInjectionAttack',
]
