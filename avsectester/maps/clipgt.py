"""Read NuRec clipgt semantic geometry without dataset or simulator dependencies."""

from __future__ import annotations

import io
from typing import Any
import xml.etree.ElementTree as ElementTree
import zipfile

import numpy as np

from avsectester.geometry.poses import make_pose, quaternion_to_rotation
from avsectester.maps.scenario_map import (
    STOP_LINE_KIND_BY_CONTROL_KIND, Lane, ScenarioMap, StopLine, TrafficControl,
)


def _map_points(values):
    """Read one clipgt polyline; placeholder rows contain no geometry."""
    if not values:
        return np.empty((0, 3), dtype=float)
    return np.asarray([[point[axis] for axis in ("x", "y", "z")] for point in values], dtype=float)


def _matched_rails(left, right):
    """Pair rail points by normalized arc length before taking their midpoint."""
    count = max(len(left), len(right))
    fractions = np.linspace(0., 1., count)

    def resample(points):
        distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if distance[-1] == 0:
            return np.repeat(points[:1], count, axis=0)
        return np.column_stack([np.interp(fractions, distance / distance[-1], points[:, axis])
                                for axis in range(3)])

    return (resample(left) + resample(right)) / 2


def _speed_multiplier(archive):
    """Use explicit OpenDRIVE units; bare clipgt speed numbers do not declare a unit."""
    if "map.xodr" not in archive.namelist():
        return None
    root = ElementTree.fromstring(archive.read("map.xodr"))
    units = {element.get("unit") for element in root.findall(".//speed")
             if element.get("max") not in {None, "0", "undefined", "no limit"}}
    if len(units) != 1:
        return None
    return {"mph": .44704, "km/h": 1 / 3.6, "m/s": 1.}.get(next(iter(units)))


def _lane_kind(record):
    uses = set(record.get("use_types") or ())
    for token, kind in (("SHOULDER_LANE", "shoulder"), ("PARKING_LANE", "parking")):
        if token in uses:
            return kind
    vehicles = set(record.get("vehicle_types") or ())
    if "CAR" in vehicles:
        return "driving"
    if vehicles == {"BICYCLE"}:
        return "bike"
    return "other"


def _control_kind(category):
    # These source categories denote MUTCD R1-1, R1-2 and R2-1. Other signs that
    # mention stopping or yielding impose different rules and retain kind="other".
    return {
        "TRAFFIC_SIGN_REGULATORY_R1_STOP": "stop_sign",
        "TRAFFIC_SIGN_REGULATORY_R1_YIELD": "yield_sign",
        "TRAFFIC_SIGN_REGULATORY_R2_SPEED_LIMIT": "speed_limit",
    }.get(category, "other")


def read_clipgt(archive: zipfile.ZipFile) -> ScenarioMap:
    """Read canonical semantic layers from clipgt parquet tables in an open scene archive.

    Source coordinates share the reconstructed rig world frame. Missing or placeholder
    layers retain explicit reasons, and source categories remain attached to controls.
    """
    import pyarrow.parquet as parquet

    unavailable = {}
    missing_tables = {}

    def records(layer):
        name = f"clipgt/{layer}.parquet"
        if name not in archive.namelist():
            missing_tables[layer] = f"NuRec archive has no {name}"
            return ()
        rows = parquet.read_table(io.BytesIO(archive.read(name))).to_pylist()
        return tuple((row.get("key") or {}, row.get(layer) or {}) for row in rows)

    lane_records = {}
    for key, record in records("lane"):
        left, right = _map_points(record.get("left_rail")), _map_points(record.get("right_rail"))
        if key.get("map_id") is not None and len(left) >= 2 and len(right) >= 2:
            lane_records[str(key["map_id"])] = (record, left, right)

    control_records = {}
    for layer in ("traffic_sign", "traffic_light"):
        valid_controls = 0
        for key, record in records(layer):
            center, dimensions = record.get("center") or {}, record.get("dimensions") or {}
            orientation = record.get("orientation") or {}
            values = [*(center.get(axis) for axis in ("x", "y", "z")),
                      *(dimensions.get(axis) for axis in ("x", "y", "z")),
                      *(orientation.get(axis) for axis in ("x", "y", "z", "w"))]
            if all(value is None for value in values):
                continue
            if key.get("map_id") is None or any(value is None for value in values):
                unavailable["traffic_controls.partial"] = (
                    "Some NuRec controls lack an identity, pose or dimensions")
                continue
            category = str(record.get("category") or "")
            pose = make_pose(
                quaternion_to_rotation(*(orientation[axis] for axis in ("w", "x", "y", "z"))),
                np.array([center[axis] for axis in ("x", "y", "z")]),
            )
            extent = tuple(float(dimensions[axis]) for axis in ("x", "y", "z"))
            control_records[str(key["map_id"])] = (
                "traffic_light" if layer == "traffic_light" else _control_kind(category),
                category, pose, extent,
            )
            valid_controls += 1
        if layer == "traffic_light" and not valid_controls:
            unavailable["traffic_controls.traffic_light"] = missing_tables.get(layer) or (
                "NuRec traffic_light source contains no valid light geometry")

    line_records = {}
    source_line_flags = []
    for key, record in records("wait_line"):
        if key.get("map_id") is not None:
            source_line_flags.append(record.get("is_implicit"))
        points = _map_points(record.get("location"))
        if (record.get("is_implicit") is False and key.get("map_id") is not None
                and len(points) >= 2):
            # STOP is a paint label shared by signal-controlled and stop-controlled
            # approaches. The lane's controlling light/sign determines the line kind.
            line_records[str(key["map_id"])] = (points, str(record.get("category") or ""))
    implicit_only = bool(source_line_flags) and all(flag is True for flag in source_line_flags)

    neighbors = {identity: {name: [] for name in ("predecessors", "successors", "left", "right")}
                 for identity in lane_records}
    control_lanes = {identity: set() for identity in control_records}
    line_lanes = {identity: set() for identity in line_records}
    lane_relations = {"PREVIOUS_LANE": "predecessors", "NEXT_LANE": "successors",
                      "LEFT_LANE": "left", "RIGHT_LANE": "right"}
    for key, record in records("association"):
        relation = lane_relations.get(key.get("kind"))
        for subject in record.get("subjects") or ():
            for target in record.get("objects") or ():
                subject, target = str(subject), str(target)
                if relation and subject in neighbors and target in neighbors:
                    neighbors[subject][relation].append(target)
                # Real clipgt SIGN_TO_LANE/LIGHT_TO_LANE/WAIT_LINE_TO_LANE rows put the
                # lane in subjects. Identify endpoints by their layer instead of the label.
                for first, second in ((subject, target), (target, subject)):
                    if first in lane_records and second in control_records:
                        control_lanes[second].add(first)
                    if first in lane_records and second in line_records:
                        line_lanes[second].add(first)

    lane_controls = {identity: set() for identity in lane_records}
    control_lines = {identity: set() for identity in control_records}
    for control_id, lane_ids in control_lanes.items():
        if control_records[control_id][0] in STOP_LINE_KIND_BY_CONTROL_KIND:
            for lane_id in lane_ids:
                lane_controls[lane_id].add(control_id)
    line_controls = {identity: set().union(*(lane_controls[lane_id] for lane_id in lane_ids))
                     for identity, lane_ids in line_lanes.items()}
    for line_id, control_ids in line_controls.items():
        for control_id in control_ids:
            control_lines[control_id].add(line_id)

    multiplier = _speed_multiplier(archive)
    lanes = {}
    for identity, (record, left, right) in lane_records.items():
        speed = None
        value = record.get("speed_limit")
        if value in (None, "", "0", 0):
            unavailable["lanes.speed_limit_mps"] = "Some NuRec lane speed limits are missing or unspecified"
        else:
            if multiplier is None:
                unavailable["lanes.speed_limit_mps"] = "NuRec lane speed values have no confirmed unit"
            else:
                speed = float(value) * multiplier
        relation = neighbors[identity]
        lanes[identity] = Lane(
            id=identity, centerline=_matched_rails(left, right), left_boundary=left,
            right_boundary=right, lane_type=_lane_kind(record),
            predecessors=tuple(dict.fromkeys(relation["predecessors"])),
            successors=tuple(dict.fromkeys(relation["successors"])),
            left_neighbor=next(iter(relation["left"]), None),
            right_neighbor=next(iter(relation["right"]), None), speed_limit_mps=speed,
        )
    controls = {identity: TrafficControl(
        id=identity, kind=kind, native_category=category, pose=pose, extent=extent,
        pose_reference="face_center",
        lane_ids=tuple(sorted(control_lanes[identity])),
        stop_line_id=next(iter(control_lines[identity])) if len(control_lines[identity]) == 1 else None,
    ) for identity, (kind, category, pose, extent) in control_records.items()}
    lines = {}
    for identity, (points, native_category) in line_records.items():
        controller_ids = line_controls[identity]
        kinds = {STOP_LINE_KIND_BY_CONTROL_KIND[control_records[control_id][0]] for control_id in controller_ids}
        lines[identity] = StopLine(
            id=identity, points=points, native_category=native_category,
            kind=next(iter(kinds)) if len(kinds) == 1 else "unknown",
            lane_ids=tuple(sorted(line_lanes[identity])),
            control_id=next(iter(controller_ids)) if len(controller_ids) == 1 else None,
        )

    def geometries(layer, minimum):
        return tuple(points for _key, record in records(layer)
                     if len(points := _map_points(record.get("location"))) >= minimum)

    values: dict[str, Any] = {"lanes": lanes, "traffic_controls": controls, "stop_lines": lines}
    layers: dict[str, tuple[str, ...]] = {
        "lanes": ("lane",), "traffic_controls": ("traffic_sign", "traffic_light"),
        "stop_lines": ("wait_line",),
    }
    for name, layer, minimum in (("drivable_area", "drivable_space", 3),
            ("road_boundaries", "road_boundary", 2), ("crosswalks", "crosswalk", 3),
            ("intersections", "intersection_area", 3)):
        values[name] = geometries(layer, minimum)
        layers[name] = (layer,)
    for name, records_in_layer in values.items():
        if not records_in_layer:
            if name == "stop_lines" and implicit_only:
                continue
            reasons = [missing_tables[layer] for layer in layers[name] if layer in missing_tables]
            unavailable[name] = "; ".join(reasons) or "NuRec source layer contains no valid geometry"
    return ScenarioMap(**values, unavailable_layers=unavailable)


__all__ = [
    'read_clipgt',
]
