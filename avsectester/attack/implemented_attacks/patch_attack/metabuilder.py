"""Score candidate hosts in clean views and rank hosts within each decision window."""
import numpy as np

from avsectester.attack.attack import AttackSegment
from avsectester.attack.insertion.action import InsertionAttackAction
from avsectester.closed_loop.services import RenderRequest
from avsectester.closed_loop.simulators.carla import CarlaSimulator


def _windows(frames, scores, decision_steps, keep_tail):
    cursor = 0
    while cursor < len(frames) - 1:
        common = set(scores.get(frames[cursor].position, {}))
        if not common:
            cursor += 1
            continue
        stop = cursor + 1
        while stop < len(frames) - 1 and stop - cursor < decision_steps:
            shared = common.intersection(scores.get(frames[stop].position, {}))
            if not shared:
                break
            common = shared
            stop += 1
        if stop - cursor == decision_steps or keep_tail:
            yield frames[cursor], frames[stop], common
            cursor = stop
        else:
            cursor += 1


def build_metadata(attack, scenario):
    physical = isinstance(scenario.services.simulator, CarlaSimulator)
    if physical and attack.orientation.mode != "follow_host":
        raise ValueError("CARLA physical patches require follow_host orientation")
    frames, scores = tuple(scenario.frames), {}
    if len(frames) < 2:
        scenario.record_reason("No decision interval has a terminal reference frame")
        return ()
    candidates = attack.candidate_hosts
    discovered = []
    with scenario.services.session(scenario.seed) as session:
        for frame in frames[:-1]:
            result = session.render(RenderRequest(scenario.scenario_id, frame, attack.sensor))
            hosts = candidates if candidates is not None else tuple(sorted(result.view.actors))
            if not hosts:
                scenario.record_reason("No candidate hosts are present")
            per_frame = {}
            for host in hosts:
                if host not in discovered:
                    discovered.append(host)
                if host not in result.view.actors:
                    scenario.record_reason("Candidate host is absent")
                    continue
                actor = result.view.actors[host]
                distance = np.linalg.norm(actor.transform[:3, 3] - result.view.ego.transform[:3, 3])
                if attack.max_distance is not None and distance > attack.max_distance:
                    scenario.record_reason("Candidate host exceeds the maximum distance")
                    continue
                insertion = attack.insertion(attack.assets, host=host)
                action = InsertionAttackAction(attack.sensor, insertion, compositor=attack.compositor)
                evidence = action.evidence(result.view)
                visibility = evidence.visibility
                if visibility is None or visibility.fraction <= attack.min_visible_fraction:
                    scenario.record_reason("Candidate host is not visible enough")
                    continue
                per_frame[host] = visibility.fraction
            scores[frame.position] = per_frame
    order = {host: index for index, host in enumerate(discovered)}
    selected = []
    for index, (start, stop, common) in enumerate(_windows(
            frames, scores, attack.decision_steps, attack.keep_tail)):
        if attack.max_windows is not None and index >= attack.max_windows:
            break
        values = {host: min(scores[position][host] for position in range(start.position, stop.position))
                  for host in common}
        ranked = sorted(common, key=lambda host: (-values[host], order[host]))
        for rank, host in enumerate(ranked[:attack.hosts_per_window]):
            segment = AttackSegment(scenario.scenario_id, start, stop, roles={"host": (host,)},
                placement=attack.placement_for(host), selection_score=values[host], selection_rank=rank,
                user_data={"deployment": "world" if physical else "sensor"})
            selected.append(segment)
    return tuple(selected)
