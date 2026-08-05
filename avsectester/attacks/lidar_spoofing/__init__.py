"""LiDAR-spoofing attack vector.

One delivery mechanism (:class:`LidarSpoofingVector`), multiple methods that share it:

- :class:`ObjectSpoofingAttack` — inject a phantom obstacle (false positive).
- :class:`ObjectRemovalAttack`  — erase a real obstacle (false negative).

``LidarSpoofAttack`` is a backward-compatible alias of :class:`ObjectSpoofingAttack`.
"""

from .object_removal import ObjectRemovalAttack
from .object_spoofing import LidarSpoofAttack, ObjectSpoofingAttack
from .vector import LidarSpoofingVector

__all__ = [
    "LidarSpoofAttack",
    "LidarSpoofingVector",
    "ObjectRemovalAttack",
    "ObjectSpoofingAttack",
]
