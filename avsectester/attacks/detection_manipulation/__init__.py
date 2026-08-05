"""Detection-manipulation attack vector.

One mechanism (:class:`DetectionManipulationVector`, at the ``perception_out`` seam),
multiple methods that share it:

- :class:`PhantomDetectionAttack` — inject a phantom detection (false positive).
- :class:`DetectionRemovalAttack` — suppress a real detection (false negative).
"""

from .detection_injection import PhantomDetectionAttack
from .detection_removal import DetectionRemovalAttack
from .vector import DetectionManipulationVector

__all__ = [
    "DetectionManipulationVector",
    "DetectionRemovalAttack",
    "PhantomDetectionAttack",
]
