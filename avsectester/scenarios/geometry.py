"""Shared geometric measurements for scenario filters. No simulator dependencies."""

from functools import lru_cache
import math


@lru_cache(maxsize=1024)
def clipped_box_area(box: tuple, width: int, height: int) -> float:
    """Image-clipped box area, cached by values rather than mutable scene identity."""
    x1, y1, x2, y2 = box
    if width <= 0 or height <= 0 or not all(math.isfinite(v) for v in box):
        raise ValueError("Projection requires finite coordinates and positive image dimensions")
    return max(0.0, min(x2, width) - max(x1, 0)) * max(0.0, min(y2, height) - max(y1, 0))


def rear_view_angle(center, yaw, camera_position) -> float | None:
    """Horizontal angle in degrees between the rear direction and target-to-camera ray."""
    dx, dy = camera_position[0] - center[0], camera_position[1] - center[1]
    if math.hypot(dx, dy) < 1e-9:
        return None
    bearing = math.atan2(dy, dx)
    delta = bearing - (yaw + math.pi)
    return abs(math.degrees(math.atan2(math.sin(delta), math.cos(delta))))
