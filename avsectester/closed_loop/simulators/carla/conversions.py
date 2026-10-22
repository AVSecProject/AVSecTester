"""Convert native CARLA actors, signal states and rotations to canonical world records."""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from avsectester.geometry.poses import mirror_y_points, mirror_y_transform
from avsectester.data_types import ActorState


def rotation_to_carla(rotation: ArrayLike) -> tuple[float, float, float]:
    """Return native CARLA (pitch, yaw, roll) degrees from a native rotation matrix.

    CARLA uses Rz(yaw) @ Ry(-pitch) @ Rx(-roll). At either pitch singularity,
    choose roll=0 and retain the observable yaw/roll combination.
    """
    matrix = np.asarray(rotation, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-6)):
        raise ValueError("rotation must be a finite right-handed 3x3 matrix")
    pitch = np.arcsin(np.clip(matrix[2, 0], -1.0, 1.0))
    if np.hypot(matrix[0, 0], matrix[1, 0]) > 1e-8:
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
        roll = np.arctan2(-matrix[2, 1], matrix[2, 2])
    else:
        yaw = np.arctan2(-matrix[0, 1], matrix[1, 1])
        roll = 0.0
    angles = np.rad2deg([pitch, yaw, roll])
    return float(angles[0]), float(angles[1]), float(angles[2])


def _traffic_light_state(state: Any) -> str:
    value = str(state).rsplit(".", 1)[-1].lower()
    return value if value in {"red", "yellow", "green", "off"} else "unknown"


def _actor_state(actor, type_id, box=None):
    """Canonical world state of a CARLA actor or actor snapshot, with its class (``vehicle``)."""
    velocity = actor.get_velocity()
    box = box if box is not None else getattr(actor, "bounding_box", None)
    extent = None if box is None else (2 * box.extent.x, 2 * box.extent.y, 2 * box.extent.z)
    pose = mirror_y_transform(np.asarray(actor.get_transform().get_matrix()))
    center = (tuple(mirror_y_points((box.location.x, box.location.y, box.location.z)))
              if box is not None and hasattr(box, "location") else (0., 0., 0.))
    rotation = ((box.rotation.roll, -box.rotation.pitch, -box.rotation.yaw)
                if box is not None and hasattr(box, "rotation") else (0., 0., 0.))
    return ActorState(pose, tuple(mirror_y_points((velocity.x, velocity.y, velocity.z))),
                      type_id.split(".")[0], extent, center, rotation)
