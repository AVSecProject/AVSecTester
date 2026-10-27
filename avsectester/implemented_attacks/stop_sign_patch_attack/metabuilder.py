"""Select recorded, correctly facing stop-sign approaches without scene-specific tuning."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import numpy as np

from avsectester.attack import AttackSegment
from avsectester.data_types import Unavailable
from avsectester.datasets import FrameReference
from avsectester.execution.services import ScenarioView
from avsectester.maps import TrafficControl
from avsectester.validation import (
    require_nonnegative_integer, require_nonnegative_number, require_positive_number,
)


def route_progress(points: np.ndarray, route_points: np.ndarray) -> np.ndarray:
    """Project world points onto the recording's horizontal arc length, in metres.

    The first and last nonzero segments extend beyond recorded support for distance
    measurement only. This lets analysis distinguish crossing beyond a recording that
    ends before its stop line; no observations or trajectory states are manufactured.
    """
    points = np.asarray(points, dtype=float)
    route = np.asarray(route_points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or route.ndim != 2
            or route.shape[1] != 3 or len(route) < 2
            or not np.isfinite(points).all() or not np.isfinite(route).all()):
        raise ValueError("Route projection requires finite (N, 3) world points")
    vectors = np.diff(route[:, :2], axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = np.flatnonzero(lengths > 1e-9)
    if not len(valid):
        raise ValueError("Recorded route has no horizontal motion")
    cumulative = np.r_[0., np.cumsum(lengths)]
    origins, vectors, lengths = route[valid, :2], vectors[valid], lengths[valid]
    fractions = np.einsum("psd,sd->ps", points[:, None, :2] - origins, vectors) / lengths ** 2
    clipped = np.clip(fractions, 0., 1.)
    if len(valid) == 1:
        clipped = fractions
    else:
        clipped[:, 0] = np.minimum(fractions[:, 0], 1.)
        clipped[:, -1] = np.maximum(fractions[:, -1], 0.)
    projections = origins + clipped[..., None] * vectors
    nearest = np.argmin(np.sum((points[:, None, :2] - projections) ** 2, axis=-1), axis=1)
    return cumulative[valid[nearest]] + clipped[np.arange(len(points)), nearest] * lengths[nearest]


def line_distances(points: np.ndarray, line_points: np.ndarray) -> np.ndarray:
    """Horizontal distances from world points to a finite stop-line polyline."""
    points, line = np.asarray(points), np.asarray(line_points)
    vectors = np.diff(line[:, :2], axis=0)
    squared = np.einsum("ij,ij->i", vectors, vectors)
    fractions = np.divide(
        np.einsum("psd,sd->ps", points[:, None, :2] - line[:-1, :2], vectors), squared,
        out=np.zeros((len(points), len(vectors))), where=squared > 1e-12,
    )
    projections = line[:-1, :2] + np.clip(fractions, 0., 1.)[..., None] * vectors
    return np.sqrt(np.min(np.sum((points[:, None, :2] - projections) ** 2, axis=-1), axis=1))


def control_from_user_data(user_data: Mapping[str, Any]) -> TrafficControl:
    """Reconstruct the selected face-centred sign using only persisted attack metadata."""
    required = ("sign_id", "sign_pose", "sign_extent", "sign_native_category")
    missing = [name for name in required if name not in user_data]
    if missing:
        raise ValueError(f"Stop-approach metadata is missing {missing}; required: {list(required)}")
    return TrafficControl(
        str(user_data["sign_id"]), "stop_sign", str(user_data["sign_native_category"]),
        np.asarray(user_data["sign_pose"]), tuple(user_data["sign_extent"]), "face_center",
        lane_ids=tuple(user_data.get("sign_lane_ids", ())),
    )


@dataclass(frozen=True)
class StopApproach:
    """One recorded stop, its governing sign and an executable approach interval."""

    scenario_id: str
    start: FrameReference
    stop: FrameReference
    sign_id: str
    sign_pose: np.ndarray
    sign_extent: tuple[float, float, float]
    sign_native_category: str
    stop_line_id: str
    stop_line_points: np.ndarray
    recorded_stop_frame: FrameReference
    approach_distance_m: float
    route_points: np.ndarray
    line_route_distance_m: float
    sign_lane_ids: tuple[str, ...] = ()
    stop_line_lane_ids: tuple[str, ...] = ()

    def to_segment(self) -> AttackSegment:
        """Persist JSON-like values and NumPy arrays; never put custom records in user_data."""
        return AttackSegment(self.scenario_id, self.start, self.stop, {
            "sign_id": self.sign_id,
            "sign_pose": self.sign_pose.copy(), "sign_extent": list(self.sign_extent),
            "sign_native_category": self.sign_native_category,
            "sign_lane_ids": list(self.sign_lane_ids),
            "stop_line_id": self.stop_line_id, "stop_line_points": self.stop_line_points.copy(),
            "stop_line_lane_ids": list(self.stop_line_lane_ids),
            "recorded_stop_frame": self.recorded_stop_frame.position,
            "recorded_stop_timestamp_s": self.recorded_stop_frame.timestamp,
            "approach_distance_m": self.approach_distance_m,
            "route_points": self.route_points.copy(),
            "line_route_distance_m": self.line_route_distance_m,
        })


def _crossing_distances(route: np.ndarray, line: np.ndarray) -> list[float]:
    vectors = np.diff(route[:, :2], axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = np.flatnonzero(lengths > 1e-9)
    cumulative = np.r_[0., np.cumsum(lengths)]
    distances = []
    for route_index in valid:
        vector = vectors[route_index]
        for first, second in zip(line[:-1, :2], line[1:, :2]):
            transverse = second - first
            determinant = vector[0] * transverse[1] - vector[1] * transverse[0]
            if abs(determinant) < 1e-9:
                continue
            offset = first - route[route_index, :2]
            fraction = (offset[0] * transverse[1] - offset[1] * transverse[0]) / determinant
            line_fraction = (offset[0] * vector[1] - offset[1] * vector[0]) / determinant
            endpoint_distance = abs(line_fraction - np.clip(line_fraction, 0., 1.)) * np.linalg.norm(transverse)
            if endpoint_distance >= 3.:
                continue
            in_support = -1e-8 <= fraction <= 1 + 1e-8
            before_support = route_index == valid[0] and fraction < 0
            after_support = route_index == valid[-1] and fraction > 1
            if in_support or before_support or after_support:
                distances.append(float(cumulative[route_index] + fraction * lengths[route_index]))
    return sorted(set(round(distance, 8) for distance in distances))


def select_stop_approach(
    scenario: ScenarioView, approach_m: float = 50., hold_steps: int = 20,
    min_approach_m: float = 0.,
) -> StopApproach | None:
    """Find one recorded stop on the driven route, checking history through the session.

    Only stop_sign stop lines are eligible. Traffic-light, yield-sign and unknown
    stop lines are excluded. The segment starts at the earliest frame within ``approach_m``
    of the line that has the required history; an approach whose start is closer to the
    line than ``min_approach_m`` is rejected.
    """
    require_positive_number(approach_m, "approach_m")
    require_nonnegative_integer(hold_steps, "hold_steps")
    require_nonnegative_number(min_approach_m, "min_approach_m")
    if min_approach_m > approach_m:
        raise ValueError("min_approach_m must not exceed approach_m")
    scenario_map = scenario.scenario_map()
    if isinstance(scenario_map, Unavailable):
        scenario.record_reason(f"Scenario map unavailable: {scenario_map.reason}")
        return None
    frames = tuple(scenario.frames)
    if len(frames) < 2:
        scenario.record_reason("Recording has fewer than two frames")
        return None
    states = [scenario.frame_data(frame).ego for frame in frames]
    if any(state is None or state.velocity is None for state in states):
        scenario.record_reason("Recording lacks ego pose or velocity")
        return None
    route = np.stack([state.pose[:3, 3] for state in states])
    speeds = np.array([np.linalg.norm(state.velocity) for state in states])
    headings = np.stack([state.pose[:3, 0] for state in states])
    progress = np.r_[0., np.cumsum(np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1))]
    signs = {identity: control for identity, control in scenario_map.traffic_controls.items()
             if control.kind == "stop_sign" and control.pose_reference == "face_center"}
    lines = sorted((line for line in scenario_map.stop_lines.values() if line.kind == "stop_sign"),
                   key=lambda line: line.id)
    if not lines:
        scenario.record_reason("No stop-sign stop lines in the map")
        return None
    accepted: list[StopApproach] = []
    with scenario.services.session(scenario.seed) as session:
        for line in lines:
            if line_distances(route, line.points).min() >= 3.:
                scenario.record_reason("Stop line does not intersect the recorded route within 3 m")
                continue
            linked = signs.get(line.control_id)
            controls = [linked] if linked is not None else sorted(
                (control for control in signs.values()
                 if line_distances(control.pose[None, :3, 3], line.points)[0] <= 20.
                 and (control.pose[:2, 3] - line.points[:, :2].mean(axis=0)) @ control.pose[:2, 0] >= 0.),
                key=lambda control: (line_distances(control.pose[None, :3, 3], line.points)[0], control.id),
            )
            if not controls:
                scenario.record_reason("Stop line has no associated or nearby entering-side stop sign")
                continue
            for crossing in _crossing_distances(route, line.points):
                distances = crossing - progress
                stopped = np.flatnonzero((distances >= -1e-8) & (distances <= 15.) & (speeds < .5))
                if not len(stopped):
                    scenario.record_reason("Recording has no stop below 0.5 m/s within 15 m before the line")
                    continue
                recorded_stop = int(stopped[0])
                for control in controls:
                    normal = control.pose[:3, 0]
                    heading = headings[recorded_stop]
                    if normal @ heading / (np.linalg.norm(normal) * np.linalg.norm(heading)) >= -.5:
                        scenario.record_reason("Stop sign does not face the recorded approach")
                        continue
                    entry_side = (route[recorded_stop] - line.points.mean(axis=0)) @ normal
                    if entry_side < -1e-6:
                        scenario.record_reason("Recorded vehicle approaches the stop line from its exit side")
                        continue
                    starts = [index for index in range(recorded_stop + 1)
                              if -1e-8 <= distances[index] <= approach_m
                              and session.can_start_at(frames[index].position)]
                    if not starts:
                        scenario.record_reason("Approach has insufficient past history or deployment space")
                        continue
                    first = starts[0]
                    if distances[first] < min_approach_m:
                        scenario.record_reason("Approach starts closer to the line than min_approach_m")
                        continue
                    last = min(recorded_stop + hold_steps, len(frames) - 1)
                    if last <= first:
                        scenario.record_reason("Selected approach has no executable decision interval")
                        continue
                    accepted.append(StopApproach(
                        scenario.scenario_id, frames[first], frames[last],
                        control.id, control.pose.copy(), control.extent, control.native_category,
                        line.id, line.points.copy(), frames[recorded_stop], float(distances[first]),
                        route.copy(), float(crossing), control.lane_ids, line.lane_ids,
                    ))
    if not accepted:
        if not scenario.reasons:
            scenario.record_reason("Recording does not cross a candidate stop line in its entering direction")
        return None
    chosen = min(accepted, key=lambda item: (
        item.recorded_stop_frame.position,
        -item.approach_distance_m, item.stop_line_id, item.sign_id,
    ))
    return chosen
