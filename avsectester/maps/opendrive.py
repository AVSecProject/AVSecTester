"""Read canonical semantic layers through CARLA's offline OpenDRIVE client parser."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Any, Literal
import xml.etree.ElementTree as ET

import numpy as np

from avsectester.datasets.base import Unavailable

from .scenario_map import Lane, ScenarioMap, STOP_LINE_KIND_BY_CONTROL_KIND, StopLine, TrafficControl


_FLIP = np.diag([1., -1., 1., 1.])
_LANE_TYPES: dict[str, Literal["driving", "shoulder", "bike", "parking", "other"]] = {
    "driving": "driving", "shoulder": "shoulder", "biking": "bike", "parking": "parking",
}
_CONTROL_KINDS: dict[str, Literal["stop_sign", "yield_sign", "speed_limit", "traffic_light", "other"]] = {
    "206": "stop_sign", "205": "yield_sign", "274": "speed_limit", "1000001": "traffic_light",
}


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise horizontal convex hull (monotone chain); vertices keep their z."""
    order = np.lexsort((points[:, 1], points[:, 0]))
    ordered = points[order]

    def chain(values: np.ndarray) -> list[np.ndarray]:
        hull: list[np.ndarray] = []
        for point in values:
            while len(hull) >= 2:
                first, second = hull[-1][:2] - hull[-2][:2], point[:2] - hull[-2][:2]
                if first[0] * second[1] - first[1] * second[0] > 0:  # counter-clockwise turn
                    break
                hull.pop()
            hull.append(point)
        return hull

    lower, upper = chain(ordered), chain(ordered[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _lane_id(waypoint: Any) -> str:
    return f"{waypoint.road_id}:{waypoint.section_id}:{waypoint.lane_id}"


def _point(location: Any) -> np.ndarray:
    return np.array([location.x, -location.y, location.z], dtype=float)


def _edges(waypoint: Any) -> tuple[np.ndarray, np.ndarray]:
    center = _point(waypoint.transform.location)
    right = _point(waypoint.transform.get_right_vector()) * waypoint.lane_width / 2
    return center - right, center + right


def _ordered_samples(values: list[Any]) -> list[Any]:
    # Preserve distinct native states immediately either side of an s-offset step.
    unique = {value.s: value for value in values}
    return sorted(unique.values(), key=lambda value: value.s, reverse=values[0].lane_id > 0)


def _sample_lane(native: Any, road: int, lane: int, start: float, stop: float) -> list[Any]:
    """Sample every lane type, including disconnected lanes, using native geometry.

    The SDK's bulk generator covers driving lanes only. Explicit road/lane/s queries also
    cover shoulders, sidewalks and isolated sections. Offset curves can be longer than the
    road reference line, so native midpoint queries refine the centre and both boundaries
    until every adjacent pair is at most two metres apart.
    """
    margin = min(1e-4, (stop - start) / 4)
    count = max(1, int(np.ceil((stop - start - 2 * margin) / 2)))
    values = [native.get_waypoint_xodr(road, lane, float(position))
              for position in np.linspace(start + margin, stop - margin, count + 1)]
    values = [value for value in values if value is not None]
    samples = []
    for first, last in zip(values, values[1:]):
        pending = [(first, last)]
        while pending:
            left, right = pending.pop()
            first_points = (_point(left.transform.location), *_edges(left))
            last_points = (_point(right.transform.location), *_edges(right))
            longest = max(np.linalg.norm(last - first)
                          for first, last in zip(first_points, last_points))
            if longest <= 2.000001:
                samples.append(left)
                continue
            middle = native.get_waypoint_xodr(road, lane, (left.s + right.s) / 2)
            if middle is None or not left.s < middle.s < right.s:
                # Native widths/offsets may have a true step. Keep both endpoints;
                # output polylines subdivide their existing straight connector below.
                samples.append(left)
                continue
            pending.extend(((middle, right), (left, middle)))
    if values:
        samples.append(values[-1])
    return samples


def _subdivide_polyline(points: np.ndarray) -> np.ndarray:
    """Add collinear vertices to long native connectors, preserving the polyline exactly."""
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if not (distances > 2.000001).any():
        return points
    pieces = []
    for first, last, distance in zip(points[:-1], points[1:], distances):
        count = max(1, int(np.ceil(distance / 2)))
        pieces.append(np.linspace(first, last, count + 1)[:-1])
    return np.concatenate((*pieces, points[-1:]))


def _waypoints(native: Any, xodr_text: str) -> tuple[dict[str, list[Any]], list[tuple[Any, Any]]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    # XML supplies identities and section intervals only. All coordinates, widths and
    # rotations still come from CARLA's offline OpenDRIVE parser.
    for road in ET.fromstring(xodr_text).findall("road"):
        road_id, length = int(road.attrib["id"]), float(road.attrib["length"])
        sections = road.findall("./lanes/laneSection")
        for index, section in enumerate(sections):
            start = float(section.attrib["s"])
            stop = float(sections[index + 1].attrib["s"]) if index + 1 < len(sections) else length
            if stop <= start:
                continue
            for lane in (*section.findall("./left/lane"), *section.findall("./right/lane")):
                for waypoint in _sample_lane(native, road_id, int(lane.attrib["id"]), start, stop):
                    grouped[_lane_id(waypoint)].append(waypoint)
    return {key: _ordered_samples(values) for key, values in grouped.items()}, list(native.get_topology())


def _neighbor(waypoint: Any, side: str, lane_ids: set[str]) -> str | None:
    adjacent = getattr(waypoint, f"get_{side}_lane")()
    identity = _lane_id(adjacent) if adjacent is not None else None
    if identity not in lane_ids or waypoint.lane_id * adjacent.lane_id <= 0:
        return None
    forward = _point(waypoint.transform.get_forward_vector())[:2]
    other = _point(adjacent.transform.get_forward_vector())[:2]
    return identity if forward @ other > 0 else None


def _lanes(
    samples: dict[str, list[Any]], topology: list[tuple[Any, Any]],
) -> tuple[dict[str, Lane], tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    usable = {key: values for key, values in samples.items() if len(values) >= 2}
    lane_ids = set(usable)
    predecessors: dict[str, set[str]] = defaultdict(set)
    successors: dict[str, set[str]] = defaultdict(set)
    for first, last in topology:
        first_id, last_id = _lane_id(first), _lane_id(last)
        if first_id != last_id and first_id in lane_ids and last_id in lane_ids:
            successors[first_id].add(last_id)
            predecessors[last_id].add(first_id)
    for key, values in usable.items():
        for previous in values[0].previous(.1):
            identity = _lane_id(previous)
            if identity != key and identity in lane_ids:
                predecessors[key].add(identity)
                successors[identity].add(key)
        for following in values[-1].next(.1):
            identity = _lane_id(following)
            if identity != key and identity in lane_ids:
                successors[key].add(identity)
                predecessors[identity].add(key)
    lanes = {}
    for key, values in usable.items():
        middle = values[len(values) // 2]
        edges = [_edges(waypoint) for waypoint in values]
        lanes[key] = Lane(
            key, _subdivide_polyline(np.array([_point(waypoint.transform.location) for waypoint in values])),
            _subdivide_polyline(np.array([left for left, _ in edges])),
            _subdivide_polyline(np.array([right for _, right in edges])),
            _LANE_TYPES.get(str(middle.lane_type).lower(), "other"),
            tuple(sorted(predecessors[key])), tuple(sorted(successors[key])),
            _neighbor(middle, "left", lane_ids), _neighbor(middle, "right", lane_ids),
        )
    driving = {key for key, lane in lanes.items() if lane.lane_type == "driving"}
    drivable, boundaries = [], []
    junction_points: dict[int, list[np.ndarray]] = defaultdict(list)
    for key in sorted(driving):
        lane = lanes[key]
        polygon = np.concatenate((lane.left_boundary, lane.right_boundary[::-1]))
        drivable.append(polygon)
        inside = next((waypoint for waypoint in usable[key] if waypoint.is_junction), None)
        if inside is not None:
            junction_points[inside.get_junction().id].append(polygon)
    # One area per junction, matching datasets that label each intersection once.
    intersections = [_convex_hull(np.concatenate(junction_points[identity]))
                     for identity in sorted(junction_points)]
    # A roadside is the outside of the outermost non-sidewalk lane on each side of a
    # non-junction road section. Junction connector edges and internal lane separators
    # are not physical road boundaries.
    outermost: dict[tuple[int, int, int], str] = {}
    for key, values in usable.items():
        middle = values[len(values) // 2]
        if middle.is_junction or str(middle.lane_type).lower() == "sidewalk":
            continue
        group = middle.road_id, middle.section_id, 1 if middle.lane_id > 0 else -1
        previous = outermost.get(group)
        if previous is None or abs(middle.lane_id) > abs(usable[previous][0].lane_id):
            outermost[group] = key
    boundaries.extend(lanes[key].right_boundary for _, key in sorted(outermost.items()))
    return lanes, tuple(drivable), tuple(boundaries), tuple(intersections)


def _controlled_waypoints(native: Any, landmark: Any, lanes: dict[str, Lane]) -> dict[str, Any]:
    validities = tuple(landmark.get_lane_validities())
    orientation = str(landmark.orientation).lower()
    selected = {}
    for key in lanes:
        road, _, lane = (int(value) for value in key.split(":"))
        if road != landmark.road_id:
            continue
        if validities and not any(min(low, high) <= lane <= max(low, high)
                                  for low, high in validities):
            continue
        if not validities and ((orientation == "positive" and lane > 0)
                               or (orientation == "negative" and lane < 0)):
            continue
        waypoint = native.get_waypoint_xodr(road, lane, landmark.s)
        if waypoint is not None and _lane_id(waypoint) == key:
            selected[key] = waypoint
    return selected


def _crosswalks(native: Any) -> tuple[tuple[np.ndarray, ...], str | None]:
    polygons = []
    current = []
    for location in native.get_crosswalks():
        current.append(_point(location))
        if len(current) >= 4 and np.allclose(current[0], current[-1], atol=1e-6):
            polygons.append(np.asarray(current[:-1]))
            current = []
    reason = "CARLA returned an unclosed crosswalk polygon" if current else None
    return tuple(polygons), reason


def _control_pose(landmark: Any, waypoints: list[Any]) -> np.ndarray:
    pose = _FLIP @ np.asarray(landmark.transform.get_matrix()) @ _FLIP
    directions = [_point(waypoint.transform.get_forward_vector())[:2] for waypoint in waypoints]
    directions = [direction / np.linalg.norm(direction) for direction in directions
                  if np.linalg.norm(direction) > 1e-8]
    direction = np.mean(directions, axis=0) if directions else np.zeros(2)
    length = np.linalg.norm(direction)
    if length >= .5:
        front = -direction / length
        pose[:3, :3] = np.array([[front[0], -front[1], 0.],
                                [front[1], front[0], 0.], [0., 0., 1.]])
    else:
        # CARLA's landmark transform describes the signal asset, not the canonical
        # front normal. Unbound or directionally ambiguous controls use its local
        # heading turned by 180 degrees.
        pose[:3, :3] = pose[:3, :3] @ np.diag([-1., -1., 1.])
    return pose


def read_opendrive(xodr_text: str, *, name: str) -> ScenarioMap | Unavailable:
    """Read OpenDRIVE without a simulator connection, or report a missing CARLA client package.

    Lanes use two-metre waypoint samples and native topology. Drivable polygons are
    driving-lane polygons, whose set union is the drivable area. Each OpenDRIVE junction with
    driving lanes gives one intersection polygon, the convex hull of those lanes. Road
    boundaries exclude junctions and use each side's outermost non-sidewalk lane. References
    to the same signal share one control, with a separate stop line for each controlled lane.
    These are derived lines, not a claim that OpenDRIVE recorded painted stop-line geometry.
    """
    try:
        import carla
    except (ImportError, OSError) as error:
        return Unavailable(f"OpenDRIVE maps require the carla client package: {error}")
    if not isinstance(xodr_text, str) or not xodr_text.strip() or not isinstance(name, str) or not name:
        raise ValueError("OpenDRIVE text and map name must be nonempty strings")
    native = carla.Map(name, xodr_text)
    samples, topology = _waypoints(native, xodr_text)
    lanes, drivable, boundaries, intersections = _lanes(samples, topology)
    controls, stop_lines = {}, {}
    limits: dict[str, set[float]] = defaultdict(set)
    unavailable = {}
    landmarks: dict[str, list[Any]] = defaultdict(list)
    for landmark in native.get_all_landmarks():
        landmarks[str(landmark.id)].append(landmark)
    for identity, references in landmarks.items():
        references.sort(key=lambda value: (value.road_id, value.s, value.t))
        landmark = references[0]
        kind = _CONTROL_KINDS.get(str(landmark.type), "other")
        selected = {}
        directions = []
        line_ids = set()
        for reference in references:
            controlled = _controlled_waypoints(native, reference, lanes)
            selected.update(controlled)
            directions.extend(controlled.values())
            if kind in STOP_LINE_KIND_BY_CONTROL_KIND:
                for key, waypoint in controlled.items():
                    line_id = f"stop-line:{identity}:{waypoint.road_id}:{waypoint.lane_id}"
                    line_ids.add(line_id)
                    stop_lines[line_id] = StopLine(
                        line_id, np.array(_edges(waypoint)),
                        native_category="opendrive_signal_reference", kind=STOP_LINE_KIND_BY_CONTROL_KIND[kind],
                        lane_ids=(key,), control_id=identity,
                    )
        line_id = next(iter(line_ids)) if len(line_ids) == 1 else None
        pose = _control_pose(landmark, directions)
        height = float(landmark.height)
        pose_reference = "signal_position"
        if kind != "traffic_light" and height > 0:
            pose[2, 3] += height / 2
            pose_reference = "face_center"
        controls[identity] = TrafficControl(
            identity, kind, str(landmark.type), pose,
            (0., float(landmark.width), height), pose_reference=pose_reference,
            lane_ids=tuple(sorted(selected)), stop_line_id=line_id,
        )
        if kind == "speed_limit":
            scale = {"km/h": 1 / 3.6, "kmh": 1 / 3.6, "kph": 1 / 3.6,
                     "mph": 1609.344 / 3600, "m/s": 1.}.get(str(landmark.unit).lower())
            if scale is not None and landmark.value >= 0:
                for key in selected:
                    limits[key].add(float(landmark.value) * scale)
    for key, values in limits.items():
        if len(values) == 1:
            lanes[key] = replace(lanes[key], speed_limit_mps=next(iter(values)))
    if any(lane.speed_limit_mps is None for lane in lanes.values()):
        unavailable["lanes.speed_limit_mps"] = (
            "Only unambiguous type-274 landmarks with explicit supported units provide lane "
            "speed limits; road-type limits and varying within-lane limits are unavailable"
        )
    if controls:
        unavailable["traffic_controls.extent"] = (
            "CARLA landmarks provide width and height but no thickness"
        )
    crosswalks, reason = _crosswalks(native)
    if reason is not None:
        unavailable["crosswalks.partial" if crosswalks else "crosswalks"] = reason
    if not lanes:
        unavailable["lanes"] = "CARLA provided no lane with at least two waypoint samples"
        unavailable["drivable_area"] = "No driving-lane polygons could be constructed"
        unavailable["road_boundaries"] = "No driving-lane outer boundaries could be constructed"
    return ScenarioMap(lanes, drivable, boundaries, crosswalks, intersections,
                       controls, stop_lines, unavailable)
