"""Per-simulation scene visualization — save what the ego sees each frame.

Each simulator exposes a different observable, so the ``visualize(Observation) -> image`` differs by
type: NuRec/AlpaSim renders a **camera** frame; CARLA yields a **lidar** cloud we draw as a top-down
bird's-eye view. :func:`record_run` drives the loop like :func:`avsectester.backend.run` but writes
``visualize(obs)`` for every frame. Frames land under ``<repo>/tmp/`` by default.

(This is the *scene* view; the clean-vs-attacked driving-impact plot is the metric view in
:mod:`avsectester.viz`.)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from avsectester.backend import AVStack, WorldBackend
from avsectester.plane import FrameRecord, Observation, Trace

REPO_TMP = Path(__file__).resolve().parents[2] / "tmp"  # <repo>/tmp (gitignored)


def camera_view(observation: Observation, camera: str | None = None) -> Any:
    """Camera simulators (NuRec / AlpaSim): the rendered HWC-uint8 RGB frame from the Observation."""
    data = observation.sensor_data
    if not data:
        return None
    key = camera if camera in data else next(iter(data))
    frame = data[key]
    return frame if hasattr(frame, "shape") else None  # skip non-image payloads (e.g. stub dicts)


def lidar_bev(observation: Observation, size: int = 800, meters: float = 60.0) -> Any:
    """CARLA: a top-down bird's-eye view of the ego lidar cloud (x forward = up, y left = left)."""
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


def save_image(image: Any, path: Path) -> None:
    from matplotlib import image as mpimg

    path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(str(path), image)


def record_run(
    backend: WorldBackend,
    stack: AVStack,
    frames: int,
    *,
    out_dir: str | Path | None = None,
    visualize: Callable[[Observation], Any] = camera_view,
    perturb: Callable[[Observation], Observation] | None = None,
    prefix: str = "frame",
) -> Trace:
    """Drive ``stack`` in ``backend`` for ``frames`` steps, saving ``visualize(obs)`` per frame.

    Same loop and Trace as :func:`avsectester.backend.run`; the only addition is writing each frame's
    scene view to ``out_dir`` (default ``<repo>/tmp/frames``). Returns the driving Trace.
    """
    out = Path(out_dir) if out_dir is not None else REPO_TMP / "frames"
    out.mkdir(parents=True, exist_ok=True)
    obs = backend.reset()
    stack.reset(obs)
    trace = Trace()
    for i in range(frames):
        seen = perturb(obs) if perturb is not None else obs
        control = stack(seen)
        image = visualize(obs)
        if image is not None:
            save_image(image, out / f"{prefix}_{i:04d}.png")
        obs = backend.step(control)
        trace.records.append(
            FrameRecord(
                frame=i,
                t=obs.t,
                speed=obs.ego_speed,
                throttle=control.throttle,
                brake=control.brake,
                steer=control.steer,
            )
        )
    return trace
