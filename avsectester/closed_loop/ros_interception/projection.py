"""Timestamp pairing and rigid transforms for a projected ROS frame pair."""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np


Vector = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


@dataclass(frozen=True)
class Pose:
    """A position and its orientation as an xyzw quaternion."""

    position: Vector
    rotation: Quaternion


def _rotation(value: Quaternion) -> np.ndarray:
    quaternion = np.array(value, dtype=float)
    magnitude = np.linalg.norm(quaternion)
    if not np.isfinite(magnitude) or magnitude == 0:
        raise ValueError("Cannot project a non-finite or zero pose quaternion")
    return quaternion / magnitude


def _vector(value: np.ndarray) -> Vector:
    return float(value[0]), float(value[1]), float(value[2])


def _multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a, b = left[:3], right[:3]
    return np.append(left[3] * b + right[3] * a + np.cross(a, b),
                     left[3] * right[3] - a @ b)


def _rotate(rotation: np.ndarray, point: Vector) -> np.ndarray:
    value = np.array(point, dtype=float)
    twice_cross = 2 * np.cross(rotation[:3], value)
    return value + rotation[3] * twice_cross + np.cross(rotation[:3], twice_cross)


def _same_rotation(left: Quaternion, right: Quaternion) -> bool:
    return left == right or all(a == -b for a, b in zip(left, right))


@dataclass(frozen=True)
class PoseDelta:
    """Rigid displacement from an original pose to its replacement."""

    original: Pose
    replacement: Pose
    rotation: np.ndarray
    translation: np.ndarray

    @classmethod
    def between(cls, original: Pose, replacement: Pose) -> PoseDelta | None:
        """Compute replacement times inverse original. None is an exact identity."""
        if (original.position == replacement.position
                and _same_rotation(original.rotation, replacement.rotation)):
            return None
        inverse = _rotation(original.rotation) * (-1, -1, -1, 1)
        rotation = _multiply(_rotation(replacement.rotation), inverse)
        translation = np.array(replacement.position) - _rotate(rotation, original.position)
        if not np.isfinite(translation).all():
            raise ValueError("Cannot project a non-finite pose translation")
        return cls(original, replacement, rotation, translation)

    def apply(self, pose: Pose) -> Pose:
        """Apply the displacement without re-rounding an exact copy of the original."""
        if (pose.position == self.original.position
                and _same_rotation(pose.rotation, self.original.rotation)):
            return self.replacement
        position = _vector(_rotate(self.rotation, pose.position) + self.translation)
        x, y, z, w = _multiply(self.rotation, _rotation(pose.rotation))
        return Pose(position, (float(x), float(y), float(z), float(w)))


T = TypeVar("T")


@dataclass(frozen=True)
class Projected(Generic[T]):
    """A queued TF message and the matched or fallback pose correction for each target."""

    message: T
    pairs: tuple[tuple[PoseDelta | None, bool], ...]


class ProjectionQueue(Generic[T]):
    """Preserve TF order and pair each selected transform with the pose of its header stamp.

    Poses arrive in stamp order. A transform waits for its pose, or for a later pose that shows
    its own will not arrive, and then uses the latest correction. ``timeout_s`` only bounds the
    wait when the pose topic stops.
    """

    def __init__(self, timeout_s: float) -> None:
        self.timeout_s = timeout_s
        self._poses: OrderedDict[int, PoseDelta | None] = OrderedDict()
        self._pending: deque[tuple[tuple[int, ...], float, T]] = deque()
        self._latest: PoseDelta | None = None
        self._latest_stamp: int | None = None

    def pose(self, stamp_ns: int, delta: PoseDelta | None) -> None:
        """Supply a pose correction at its original ROS header timestamp in nanoseconds."""
        self._poses[stamp_ns] = delta
        self._poses.move_to_end(stamp_ns)
        self._latest = delta
        self._latest_stamp = stamp_ns if self._latest_stamp is None else max(self._latest_stamp, stamp_ns)
        while len(self._poses) > 128:
            self._poses.popitem(last=False)

    def enqueue(self, message: T, stamps: Sequence[int], now: float) -> None:
        """Queue a TF message along with the timestamps of selected transforms."""
        self._pending.append((tuple(stamps), now + self.timeout_s, message))

    def _resolved(self, stamp: int) -> bool:
        return stamp in self._poses or (self._latest_stamp is not None and self._latest_stamp > stamp)

    @property
    def next_deadline(self) -> float | None:
        """Monotonic deadline of the first queued TF message, if any."""
        return self._pending[0][1] if self._pending else None

    @property
    def waiting(self) -> bool:
        """Whether any TF messages still need to be published."""
        return self.next_deadline is not None

    def drain(self, now: float) -> list[Projected[T]]:
        """Return queued messages ready for publication at the supplied monotonic time."""
        ready = []
        while self._pending:
            stamps, expires, message = self._pending[0]
            if now < expires and not all(self._resolved(stamp) for stamp in stamps):
                break
            self._pending.popleft()
            pairs = tuple((self._poses.get(stamp, self._latest), stamp in self._poses)
                          for stamp in stamps)
            ready.append(Projected(message, pairs))
        return ready


__all__ = ["Pose", "PoseDelta", "Projected", "ProjectionQueue"]
