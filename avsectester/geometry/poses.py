"""Canonical world/actor poses in metres and right-handed x-forward, y-left, z-up axes."""

import math

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


MIRROR_Y = np.diag([1., -1., 1., 1.])
MIRROR_Y.setflags(write=False)
"""Reflection between left-handed x-forward/y-right/z-up and canonical y-left axes."""


def mirror_y_points(points: ArrayLike) -> np.ndarray:
    """Mirror the y component of points or vectors with trailing coordinate dimension three."""
    values = np.asarray(points, dtype=float)
    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValueError("points must have a trailing coordinate dimension of three")
    return values * np.array([1., -1., 1.])


def mirror_y_transform(matrix: ArrayLike) -> np.ndarray:
    """Express a homogeneous transform in the opposite y-axis convention on both sides."""
    return MIRROR_Y @ np.asarray(matrix, dtype=float) @ MIRROR_Y


def quaternion_to_rotation(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Return a rotation matrix from a finite, nonzero quaternion in w/x/y/z order."""
    length = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(length) or length == 0:
        raise ValueError("quaternion must be finite and nonzero")
    w, x, y, z = w / length, x / length, y / length, z / length
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rotation_to_quaternion(rotation: ArrayLike) -> tuple[float, float, float, float]:
    """Return a normalized w/x/y/z quaternion, including for rotations close to pi.

    The eigenvector formulation avoids division by a near-zero quaternion component.
    Quaternions have a sign ambiguity; either sign represents the same rotation.
    """
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not (np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-6)
            and np.isclose(np.linalg.det(matrix), 1., atol=1e-6)):
        raise ValueError("rotation must be right-handed and orthonormal")
    m = matrix
    k = np.array([
        [m[0, 0]-m[1, 1]-m[2, 2], m[1, 0]+m[0, 1], m[2, 0]+m[0, 2], m[2, 1]-m[1, 2]],
        [m[1, 0]+m[0, 1], m[1, 1]-m[0, 0]-m[2, 2], m[2, 1]+m[1, 2], m[0, 2]-m[2, 0]],
        [m[2, 0]+m[0, 2], m[2, 1]+m[1, 2], m[2, 2]-m[0, 0]-m[1, 1], m[1, 0]-m[0, 1]],
        [m[2, 1]-m[1, 2], m[0, 2]-m[2, 0], m[1, 0]-m[0, 1], np.trace(m)],
    ]) / 3
    x, y, z, w = np.linalg.eigh(k)[1][:, -1]
    return float(w), float(x), float(y), float(z)


def interpolate_transform(first: ArrayLike, second: ArrayLike, fraction: float) -> np.ndarray:
    """Interpolate rigid poses with linear translation and shortest-path quaternion slerp."""
    first_pose, second_pose = validated_transform(first), validated_transform(second)
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("fraction must be finite and between zero and one")
    q0 = np.asarray(rotation_to_quaternion(first_pose[:3, :3]))
    q1 = np.asarray(rotation_to_quaternion(second_pose[:3, :3]))
    dot = float(q0 @ q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        quaternion = (1 - fraction) * q0 + fraction * q1
    else:
        theta = math.acos(np.clip(dot, -1, 1))
        quaternion = (math.sin((1 - fraction) * theta) * q0
                      + math.sin(fraction * theta) * q1) / math.sin(theta)
    return make_pose(quaternion_to_rotation(*quaternion),
                     (1 - fraction) * first_pose[:3, 3] + fraction * second_pose[:3, 3])


def make_pose(rotation: np.ndarray, translation: ArrayLike) -> np.ndarray:
    """Build a homogeneous transform from a rotation matrix and translation vector."""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Transform an array of three-dimensional points into the destination frame."""
    pts = np.asarray(pts, dtype=np.float64)
    return (T[:3, :3] @ pts.T).T + T[:3, 3]
