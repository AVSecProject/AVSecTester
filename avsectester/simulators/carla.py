"""CARLA-specific view adapters for the generic pipeline in :mod:`avsectester.simulators.viz`.

The generic ``viz`` interface consumes a *view* (``Observation -> RGB ndarray``); CARLA's sensor
payloads are avstack ``ImageData`` / ``LidarData``, so their unwrapping lives here rather than in the
simulator-agnostic ``viz``. These functions duck-type the avstack/CARLA data objects (reading
``.rgb_image`` / ``.data.raw_data``), so this module imports without ``carla`` and tests offline.

Use with the generic pipeline, e.g.::

    from avsectester.simulators import carla as carla_view
    from avsectester.simulators.viz import record_run, detections_view
    record_run(backend, stack, n, visualize=detections_view(detect, base=carla_view.camera_view))
"""

from __future__ import annotations

from typing import Any

from avsectester.plane import Observation
from avsectester.simulators.viz import as_rgb


def camera_view(observation: Observation, camera: str | None = None) -> Any:
    """Extract an RGB frame from a CARLA ``CarlaRgbCamera`` payload (avstack ``ImageData``)."""
    import numpy as np

    data = observation.sensor_data
    if not data:
        return None
    key = camera if camera in data else next(iter(data))
    frame = data[key]
    rgb = getattr(frame, "rgb_image", None)  # avstack ImageData -> channel-correct RGB
    if rgb is not None:
        arr = np.asarray(rgb)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return arr[:, :, :3].astype("uint8")
    return as_rgb(frame)  # already a raw ndarray


def lidar_bev(observation: Observation, size: int = 800, meters: float = 60.0) -> Any:
    """A top-down bird's-eye view of a CARLA ego lidar cloud (x forward = up, y left = left)."""
    import numpy as np

    data = next(iter(observation.sensor_data.values()), None)
    try:  # avstack LidarData wraps a CARLA measurement whose raw_data is float32 [x,y,z,intensity]
        raw = data.data.raw_data
        pts = np.frombuffer(bytes(raw), dtype=np.float32).reshape(-1, 4)[:, :3]
    except Exception:  # noqa: BLE001 - best-effort: any non-lidar/unparseable payload -> skip
        return None
    img = np.zeros((size, size, 3), dtype=np.uint8)
    scale = size / (2 * meters)
    u = (size / 2 - pts[:, 0] * scale).astype(int)  # forward -> up
    v = (size / 2 - pts[:, 1] * scale).astype(int)  # left -> left
    m = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    img[u[m], v[m]] = (0, 255, 0)
    img[size // 2 - 3 : size // 2 + 3, size // 2 - 3 : size // 2 + 3] = (255, 80, 80)  # ego
    return img
