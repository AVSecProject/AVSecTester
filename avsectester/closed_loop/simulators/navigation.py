"""Navigation intentions derived from a scenario's map or recorded ego trajectory.

Map routes follow AlpaSim's lane-consistency and centerline-extension rules. Navigation is
recomputed from the actual ego pose; it never constrains the simulated vehicle's motion.
"""
from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

from avsectester.data_types import Unavailable
from avsectester.geometry.poses import validated_transform
from avsectester.maps import Lane, ScenarioMap


def _project(points: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Closest point, segment index and along-polyline distance, without extrapolation."""
    if len(points) == 1:
        return points[0].copy(), 0, 0.0
    vectors = np.diff(points, axis=0)
    squared = np.einsum("ij,ij->i", vectors, vectors)
    fractions = np.divide(np.einsum("ij,ij->i", point - points[:-1], vectors), squared,
                          out=np.zeros_like(squared), where=squared > 0)
    fractions = np.clip(fractions, 0.0, 1.0)
    projections = points[:-1] + fractions[:, None] * vectors
    index = int(np.argmin(np.sum((projections - point) ** 2, axis=1)))
    lengths = np.sqrt(squared)
    distance = lengths[:index].sum() + fractions[index] * lengths[index]
    return projections[index], index, float(distance)


def _tangent(points: np.ndarray, *, at_end: bool = False) -> np.ndarray:
    vectors = np.diff(points[:, :2], axis=0)
    nonzero = np.flatnonzero(np.linalg.norm(vectors, axis=1) > 1e-9)
    if not len(nonzero):
        return np.array([1.0, 0.0])
    vector = vectors[nonzero[-1] if at_end else nonzero[0]]
    return vector / np.linalg.norm(vector)


def _check_foldback(points: np.ndarray) -> None:
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    for index in range(len(vectors) - 1):
        if lengths[index] < 1e-3 or lengths[index + 1] < 1e-3:
            continue
        cosine = np.dot(vectors[index], vectors[index + 1]) / (lengths[index] * lengths[index + 1])
        if cosine < math.cos(math.radians(150)):
            raise ValueError("Navigation route folds back on itself by more than 150 degrees")


class RouteGenerator:
    """Generate 20 ego-frame navigation points spaced ``80 / 19`` meters apart.

    ``kind`` is ``map``, ``recorded`` or ``none``. ``recorded_points`` contains canonical-world
    positions in recording order. Map routes use only driving lanes, project the recording
    onto a consistent lane sequence and extend along the most aligned successor. Recorded
    routes end at the recorded support. Short routes repeat their final point; disabled
    navigation returns ``None``. Each simulator runtime owns an independent instance.
    """

    def __init__(
        self, kind: str, recorded_points: Sequence[Sequence[float]] | np.ndarray,
        scenario_map: ScenarioMap | Unavailable | None,
    ) -> None:
        if kind not in {"map", "recorded", "none"}:
            raise ValueError("route_generator must be 'map', 'recorded' or 'none'")
        self.kind = kind
        self._lanes: dict[str, Lane] = {}
        self._current_lane: str | None = None
        self._points = np.empty((0, 3), dtype=float)
        if kind == "none":
            return
        points = np.array(recorded_points, dtype=float, copy=True)
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError("Navigation requires at least two finite recorded world positions")
        if kind == "map":
            if isinstance(scenario_map, ScenarioMap):
                self._lanes = {identity: lane for identity, lane in scenario_map.lanes.items()
                               if lane.lane_type == "driving"}
            if not self._lanes:
                raise ValueError("Map navigation requires an available map with driving lanes; "
                                 "choose route_generator='recorded' or 'none'")
            points = self._map_points(points)
        kept = [points[0]]
        for point in points[1:]:
            if np.linalg.norm(point - kept[-1]) >= 0.2:
                kept.append(point)
        self._points = np.asarray(kept)
        _check_foldback(self._points)

    def _map_points(self, points: np.ndarray) -> np.ndarray:
        candidates: list[list[str]] = []
        heading = 0.0
        for index, point in enumerate(points):
            if index + 1 < len(points):
                motion = points[index + 1, :2] - point[:2]
                if np.linalg.norm(motion) > 1e-2:
                    heading = math.atan2(motion[1], motion[0])
            choices: list[str] = []
            nearest = None
            for identity, lane in self._lanes.items():
                projection, segment, _ = _project(lane.centerline, point)
                distance = float(np.linalg.norm(projection - point))
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, identity)
                tangent = lane.centerline[segment + 1, :2] - lane.centerline[segment, :2]
                lane_heading = math.atan2(tangent[1], tangent[0])
                difference = abs(math.atan2(math.sin(heading - lane_heading),
                                            math.cos(heading - lane_heading)))
                if distance <= 4.0 and difference <= math.pi / 8:
                    choices.append(identity)
            assert nearest is not None
            candidates.append(choices or [nearest[1]])
        selected = [self._consistent_lane(candidates)]
        for index in range(1, len(candidates)):
            previous = selected[-1]
            if previous in candidates[index]:
                selected.append(previous)
                continue
            continuations = [identity for identity in candidates[index]
                             if identity in self._lanes[previous].successors]
            selected.append(continuations[0] if len(continuations) == 1
                            else self._consistent_lane(candidates[index:]))
        projected = []
        for index, (point, identity) in enumerate(zip(points, selected)):
            projection, _, _ = _project(self._lanes[identity].centerline, point)
            if np.linalg.norm(projection - point) > 10.0:
                raise ValueError(f"Recorded waypoint {index} projects more than 10 m from the map route")
            projected.append(projection)
        self._current_lane = selected[-1]
        return np.asarray(projected)

    @staticmethod
    def _consistent_lane(candidates: list[list[str]]) -> str:
        potential = candidates[0].copy()
        for upcoming in candidates[1:]:
            if len(potential) == 1:
                break
            for identity in potential.copy():
                if identity not in upcoming:
                    potential.remove(identity)
                    if len(potential) == 1:
                        return potential[0]
        return potential[0]

    def _extend(self) -> bool:
        stalled: set[str] = set()
        while self._current_lane is not None and self._current_lane not in stalled:
            identity = self._current_lane
            stalled.add(identity)
            lane = self._lanes[identity]
            _, segment, _ = _project(lane.centerline, self._points[-1])
            extra = []
            last = self._points[-1]
            for point in lane.centerline[segment + 1:]:
                if np.linalg.norm(point - last) > 1e-9:
                    extra.append(point)
                    last = point
            successors = [self._lanes[value] for value in sorted(lane.successors)
                          if value in self._lanes]
            self._current_lane = None
            if successors:
                direction = _tangent(lane.centerline, at_end=True)
                self._current_lane = min(successors, key=lambda successor: math.acos(float(
                    np.clip(direction @ _tangent(successor.centerline), -1.0, 1.0)))).id
            if extra:
                self._points = np.vstack((self._points, extra))
                _check_foldback(self._points)
                return True
        self._current_lane = None
        return False

    def route(self, world_from_ego: np.ndarray) -> np.ndarray | None:
        """Return forward navigation from the actual ego projection, in ego axes, with z=0."""
        pose = validated_transform(world_from_ego)
        if self.kind == "none":
            return None
        point = pose[:3, 3]
        _, _, start_distance = _project(self._points, point)
        lengths = np.linalg.norm(np.diff(self._points, axis=0), axis=1)
        total = float(lengths.sum())
        while self.kind == "map" and total - start_distance < 80.0:
            if not self._extend():
                break
            _, _, start_distance = _project(self._points, point)
            lengths = np.linalg.norm(np.diff(self._points, axis=0), axis=1)
            new_total = float(lengths.sum())
            if new_total <= total + 1e-9:
                break
            total = new_total
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        distances = start_distance + np.arange(20) * (80.0 / 19.0)
        sampled = np.column_stack([np.interp(distances, cumulative, self._points[:, axis])
                                   for axis in range(3)])
        ego_points = (sampled - pose[:3, 3]) @ pose[:3, :3]
        ego_points[:, 2] = 0.0
        return ego_points


__all__ = [
    'RouteGenerator',
]
