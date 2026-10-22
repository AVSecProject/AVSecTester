"""Shared map contract for dataset adapters."""

import numpy as np


def assert_scenario_map(scenario_map, ego):
    """Shared semantic-map contract: ego lies on a lane, controls face approaching traffic."""
    import cv2

    assert any(cv2.pointPolygonTest(np.concatenate((lane.left_boundary[:, :2],
        lane.right_boundary[::-1, :2])).astype(np.float32), tuple(ego.pose[:2, 3]), False) >= 0
        for lane in scenario_map.lanes.values())
    controls = [control for control in scenario_map.traffic_controls.values() if control.lane_ids]
    nearest = {}
    for control in controls:
        candidates = []
        for identity in control.lane_ids:
            points = scenario_map.lanes[identity].centerline
            vectors = np.diff(points, axis=0)
            lengths = np.linalg.norm(vectors, axis=1)
            for index in np.flatnonzero(lengths > 0):
                vector = vectors[index]
                fraction = np.clip((control.pose[:3, 3] - points[index]) @ vector
                                   / lengths[index] ** 2, 0., 1.)
                point = points[index] + fraction * vector
                distance = np.linalg.norm(control.pose[:3, 3] - point)
                candidates.append((distance, vector / lengths[index], point))
        assert candidates, f"Control {control.id!r} has no nonzero governed-lane tangent"
        nearest[control.id] = min(candidates, key=lambda item: item[0])
    for control in (value for value in controls
                    if value.kind in {"stop_sign", "yield_sign", "traffic_light"}):
        _distance, tangent, _point = nearest[control.id]
        assert control.pose[:3, 0] @ tangent < 0, f"Control {control.id!r} faces along traffic"
    for control in (value for value in controls if value.pose_reference == "face_center"):
        height = control.pose[2, 3] - nearest[control.id][2][2]
        assert .3 <= height <= 8., f"Control {control.id!r} has implausible face-centre height {height}"
