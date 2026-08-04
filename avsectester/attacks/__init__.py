"""Attack plugins + execution engine (PLAN.md Phase 3).

Attacks register with the ATTACKS registry and are hook-shaped (see core.interfaces.
AttackBase): they attach to avstack modules as pre/post hooks at a declared insertion
point (SensorData boundary or an inter-module edge).

Initial categories (PROJECT.md 12.3):
  sensor/   - LiDAR spoofing/point manipulation, camera adversarial perturbation
  physical/ - object-level physical attacks
  localization/ - localization / map manipulation
  v2x/      - collaborative-perception / V2X attacks (stretch)
"""

from .perception.phantom_detection import PhantomDetectionAttack
from .sensor.lidar_spoof import LidarSpoofAttack

__all__ = ["LidarSpoofAttack", "PhantomDetectionAttack"]
