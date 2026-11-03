"""Select bounded decision windows with a recorded vehicle placement ahead."""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from avsectester.attack import AttackSegment
from avsectester.attack.segments import split_segments
from avsectester.geometry.poses import validated_transform

if TYPE_CHECKING:
    from avsectester.execution.services import ScenarioView
    from .attack import VehicleInsertionAttack


def build_metadata(attack: VehicleInsertionAttack, scenario: ScenarioView) -> tuple[AttackSegment, ...]:
    frames = tuple(scenario.frames)
    if not frames:
        return ()
    recording = scenario.services.dataset.scenario_start(
        scenario.scenario_id, frames[0]).recorded_ego
    if len(recording) != len(frames):
        scenario.record_reason("Vehicle insertion requires a complete recorded ego trajectory")
        return ()
    poses = np.stack([validated_transform(state.pose) for state in recording])
    distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1))]
    targets = np.asarray(np.searchsorted(distances, distances + attack.distance_m), dtype=np.int64)
    with scenario.services.session(scenario.seed) as session:
        segments = split_segments(
            frames, lambda frame: session.can_start_at(frame.position),
            decision_steps=attack.decision_steps)
    # The insertion is fixed at setup. Only the segment start needs a future placement pose.
    segments = tuple(segment for segment in segments
                     if targets[segment.start.position] < len(frames))
    if attack.max_segments is not None:
        segments = segments[:attack.max_segments]
    result = []
    for segment in segments:
        pose = poses[targets[segment.start.position]].copy()
        pose[:3, 3] += pose[:3, 1] * attack.lateral_offset_m
        result.append(replace(segment, user_data={"world_from_object": pose}))
    scenario.record_reason(
        f"Selected {len(result)} vehicle-insertion segments with sufficient forward trajectory "
        "and supported starts")
    return tuple(result)
