"""LiDAR-spoofing attack vector.

One delivery mechanism (:class:`LidarSpoofingVector`), multiple methods that share it:

- :class:`ObjectSpoofingAttack` — inject a phantom obstacle (false positive).
- :class:`ObjectRemovalAttack`  — erase a real obstacle (false negative).

"""

from .object_removal import ObjectRemovalAttack
from .object_spoofing import ObjectSpoofingAttack
from .vector import LidarSpoofingVector

__all__ = [
    "LidarSpoofingVector",
    "ObjectRemovalAttack",
    "ObjectSpoofingAttack",
]
