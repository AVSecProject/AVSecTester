"""Select bounded decision windows with a vehicle placed on ego's navigation route."""
from __future__ import annotations

from dataclasses import replace
import math
from typing import TYPE_CHECKING

import numpy as np

from avsectester.attack import AttackSegment
from avsectester.attack.segments import split_segments
from avsectester.closed_loop.simulators.navigation import RouteGenerator
from avsectester.geometry.poses import validated_transform

if TYPE_CHECKING:
    from avsectester.execution.services import ScenarioView
    from .attack import VehicleInsertionAttack


def build_metadata(attack: VehicleInsertionAttack, scenario: ScenarioView) -> tuple[AttackSegment, ...]:
    route_kind = getattr(scenario.services.simulator, "route_generator", None)
    if route_kind is None:
        raise TypeError("VehicleInsertionAttack requires a simulator with a route_generator "
                        "setting, such as NuRecSimulator")
    frames = tuple(scenario.frames)
    if not frames:
        return ()
    first = scenario.services.dataset.scenario_start(scenario.scenario_id, frames[0])
    recording, scenario_map = first.recorded_ego, first.scenario_map
    if len(recording) != len(frames):
        scenario.record_reason("Vehicle insertion requires a complete recorded ego trajectory")
        return ()
    poses = np.stack([validated_transform(state.pose) for state in recording])
    with scenario.services.session(scenario.seed) as session:
        segments = split_segments(
            frames, lambda frame: session.can_start_at(frame.position),
            decision_steps=attack.decision_steps)
    result = []
    for segment in segments:
        if attack.max_segments is not None and len(result) == attack.max_segments:
            break
        origin = poses[segment.start.position]
        velocity = np.asarray(recording[segment.start.position].velocity, dtype=float)
        if velocity.shape != (3,) or not np.isfinite(velocity).all():
            scenario.record_reason("Vehicle insertion requires a finite recorded starting velocity")
            continue
        distance = max(attack.distance_m, float(np.linalg.norm(velocity[:2])) * attack.time_ahead_s)
        if route_kind == "none":
            yaw = math.atan2(origin[1, 0], origin[0, 0])
            route = np.array([origin[:2, 3], origin[:2, 3] + distance * np.array([
                math.cos(yaw), math.sin(yaw)])])
        else:
            try:
                # Each start gets a fresh generator, as each simulator reset does.
                generated = RouteGenerator(route_kind, poses[:, :3, 3], scenario_map).route(origin)
            except ValueError as error:
                scenario.record_reason(f"Vehicle insertion navigation is unavailable: {error}")
                continue
            assert generated is not None
            route = (generated @ origin[:3, :3].T + origin[:3, 3])[:, :2]
        vectors = np.diff(route, axis=0)
        lengths = np.linalg.norm(vectors, axis=1)
        nonzero = np.flatnonzero(lengths > 1e-9)
        lengths, vectors = lengths[nonzero], vectors[nonzero]
        cumulative = np.r_[0., np.cumsum(lengths)]
        if not len(lengths) or distance > cumulative[-1] + 1e-9:
            scenario.record_reason("Vehicle insertion distance exceeds the available navigation route")
            continue
        index = min(int(np.searchsorted(cumulative, distance, side="right")) - 1, len(lengths) - 1)
        tangent = vectors[index] / lengths[index]
        position = route[nonzero[index]] + tangent * (min(distance, cumulative[-1]) - cumulative[index])
        position += attack.lateral_offset_m * np.array([-tangent[1], tangent[0]])
        nearest = int(np.argmin(np.linalg.norm(poses[:, :2, 3] - position, axis=1)))
        pose = poses[nearest].copy()
        route_yaw = math.atan2(tangent[1], tangent[0])
        recorded_yaw = math.atan2(pose[1, 0], pose[0, 0])
        delta = route_yaw - recorded_yaw
        cosine, sine = math.cos(delta), math.sin(delta)
        correction = np.array([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]])
        pose[:3, :3] = correction @ pose[:3, :3]
        pose[:2, 3] = position
        result.append(replace(segment, user_data={"world_from_object": pose}))
    scenario.record_reason(
        f"Selected {len(result)} vehicle-insertion segments with sufficient navigation route "
        "and supported starts")
    return tuple(result)
