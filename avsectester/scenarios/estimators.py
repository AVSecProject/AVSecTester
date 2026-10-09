"""Scene projection adapter for exposing resolved insertions to visual filters."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from avsectester.insertion import ActorPose, ResolvedInsertion
from avsectester.scenarios.scene import CameraCalib, PlacementCandidate
from avsectester.rendering.cameras import camera_from_calibration, transform
from avsectester.rendering.geometry import edge_points
__all__ = ["resolved_to_target"]


def resolved_to_target(
    subject: ResolvedInsertion,
    ego: ActorPose,
    cameras: Mapping[str, CameraCalib],
) -> PlacementCandidate:
    """Expose a resolved insertion in the scene's ego frame for ordinary visual filters.

    Projection boxes are only a field-of-view aid. They are not used as visibility silhouettes.
    Missing or unsupported camera calibration leaves that camera's projection unavailable.
    """
    ego_from_world = np.linalg.inv(ego.transform)
    pose = ego_from_world @ subject.pose
    planes = subject.planes()
    local_corners = np.concatenate([surface.corners for surface in subject.asset.planes()])
    extent = np.ptp(local_corners, axis=0)
    box2d = {}
    for name, calib in cameras.items():
        if calib.cam_to_ego is None:
            continue
        try:
            camera = camera_from_calibration(calib)
        except (ValueError, TypeError, NotImplementedError):
            continue
        cam_from_world = np.linalg.inv(np.asarray(calib.cam_to_ego)) @ ego_from_world
        pixels = []
        for surface in planes:
            corners = transform(cam_from_world, surface.corners)
            if np.all(corners[:, 2] > 1e-5):
                projected = camera.project(edge_points(corners))
                if np.isfinite(projected).all():
                    pixels.append(projected)
        if pixels:
            pixels = np.concatenate(pixels)
            box2d[name] = tuple(np.concatenate((pixels.min(axis=0), pixels.max(axis=0))))
    return PlacementCandidate(
        track_id=subject.id,
        category="insertion",
        center=tuple(pose[:3, 3]),
        extent=tuple(extent),
        yaw=float(np.arctan2(pose[1, 0], pose[0, 0])),
        box2d=box2d,
        pose=pose,
    )
