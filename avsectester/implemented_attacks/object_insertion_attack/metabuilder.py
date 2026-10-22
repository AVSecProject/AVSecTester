"""Select clean-view visibility windows without applying insertion actions."""
from dataclasses import asdict, replace

import numpy as np

from avsectester.attack.insertion.action import InsertionAttackAction
from avsectester.attack.insertion.placement import AttachedPlacement, WorldPlacement
from avsectester.attack.segments import split_segments
from avsectester.execution.services import RenderRequest


def build_metadata(attack, scenario):
    scores = {}
    frames = tuple(scenario.frames)
    if len(frames) < 2:
        scenario.record_reason("No decision interval has a terminal reference frame")
        return ()
    with scenario.services.session(scenario.seed) as session:
        placement = attack.placement
        if attack.placement_relative_to_ego_origin:
            ego = session.frame_data(scenario.scenario_id, frames[0]).ego
            if ego is None:
                raise ValueError("Ego-origin-relative placement requires a reference ego actor")
            point = np.asarray(ego.pose) @ np.r_[placement.position_m, 1.]
            placement = WorldPlacement(tuple(point[:3]))
        action = InsertionAttackAction(attack.sensor, attack.insertion(attack.assets, placement),
                                      compositor=attack.compositor)

        def eligible(frame):
            result = session.render(RenderRequest(scenario.scenario_id, frame, attack.sensor))
            if isinstance(placement, AttachedPlacement) and placement.host not in result.view.actors:
                scenario.record_reason("Insertion host is absent")
                return False
            evidence = action.evidence(result.view)
            visibility = evidence.visibility
            if visibility is None or visibility.fraction <= attack.min_visible_fraction:
                scenario.record_reason("Insertion is not sufficiently visible")
                return False
            scores[frame.position] = visibility.fraction
            return True

        selected = split_segments(frames, eligible, decision_steps=attack.decision_steps,
                                  keep_tail=attack.keep_tail)
    if attack.max_segments is not None:
        selected = selected[:attack.max_segments]
    placement_data = asdict(placement)
    placement_data["kind"] = "attached" if isinstance(placement, AttachedPlacement) else "world"
    return tuple(replace(segment, user_data={
        "placement": placement_data,
        "score": min(scores[position] for position in range(
            segment.start.position, segment.stop.position)),
    }) for segment in selected)
