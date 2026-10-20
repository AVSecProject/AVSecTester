"""Canonical world/actor poses in metres and right-handed x-forward, y-left, z-up axes."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


def validated_vector(value: ArrayLike, name: str) -> np.ndarray:
    """Return a finite three-dimensional vector, rejecting invalid shapes or values."""
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return array


def validated_transform(value: ArrayLike) -> np.ndarray:
    """Return an independent, read-only right-handed rigid transform."""
    array = np.array(value, dtype=float, copy=True)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    rotation = array[:3, :3]
    if not (
        np.allclose(array[3], [0, 0, 0, 1], atol=1e-8)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise ValueError("transform must be a right-handed rigid pose")
    array.setflags(write=False)
    return array


def rotation_matrix(rotation_deg: ArrayLike) -> np.ndarray:
    """Return Rz(yaw) @ Ry(pitch) @ Rx(roll) for right-handed degree angles."""
    roll, pitch, yaw = np.deg2rad(validated_vector(rotation_deg, "rotation_deg"))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


@dataclass(frozen=True)
class ActorPose:
    """World box-centre pose and optional full (length, width, height) dimensions.

    None means dimensions are unknown; explicitly zero dimensions describe a known
    degenerate box. Consumers requiring a cuboid must reject unknown dimensions.
    """

    transform: np.ndarray
    extent: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform", validated_transform(self.transform))
        if self.extent is not None:
            extent = validated_vector(self.extent, "extent")
            if (extent < 0).any():
                raise ValueError("extent must be nonnegative")
            object.__setattr__(self, "extent", tuple(float(v) for v in extent))


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
