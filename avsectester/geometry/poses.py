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
    """World pose of a bounding-box centre and its full (length, width, height)."""

    transform: np.ndarray
    extent: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        extent = validated_vector(self.extent, "extent")
        if (extent < 0).any():
            raise ValueError("extent must be nonnegative")
        object.__setattr__(self, "transform", validated_transform(self.transform))
        object.__setattr__(self, "extent", tuple(float(v) for v in extent))
