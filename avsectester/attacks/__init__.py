"""Attack plugins, organized by **attack vector** (the shared delivery mechanism).

Attacks register with the ATTACKS registry and implement core.Attack (offline prepare +
runtime apply). They are grouped by *vector* — the mechanism and toolkit they share — with one
or more concrete *methods* per vector (see attacks/vector.py):

  lidar_spoofing/         - LiDAR-spoofing vector; methods: object spoofing (false positive),
                            object removal (false negative). Shared tools in ``vector.py``.
  detection_manipulation/ - detector-output vector; methods: phantom detection injection
                            (false positive), detection removal (false negative).

Planned vectors: camera adversarial patch, GPS/localization spoofing, V2X message injection
— each a package with its own ``vector.py`` toolkit + method classes.
"""

from .detection_manipulation import (
    DetectionManipulationVector,
    DetectionRemovalAttack,
    PhantomDetectionAttack,
)
from .lidar_spoofing import (
    LidarSpoofingVector,
    ObjectRemovalAttack,
    ObjectSpoofingAttack,
)

__all__ = [
    "DetectionManipulationVector",
    "DetectionRemovalAttack",
    "LidarSpoofingVector",
    "ObjectRemovalAttack",
    "ObjectSpoofingAttack",
    "PhantomDetectionAttack",
]
