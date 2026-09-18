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


def _as_rgb(frame: Any) -> Any:
    """Extract an ``(H, W, 3)`` uint8 RGB ndarray from a camera payload, or None if it is not one.

    Handles a raw ndarray (NuRec stub) and an avstack ``ImageData`` (CARLA ``CarlaRgbCamera``), whose
    ``.rgb_image`` gives channel-correct RGB. Non-image payloads (e.g. lidar, stub dicts) -> None.
    """
    import numpy as np

    rgb = getattr(frame, "rgb_image", frame)  # ImageData -> RGB array; ndarray passes through
    arr = np.asarray(rgb) if hasattr(rgb, "shape") else None
    if arr is None or arr.ndim != 3 or arr.shape[2] < 3:
        return None
    return arr[:, :, :3].astype("uint8")


def camera_view(observation: Observation, camera: str | None = None) -> Any:
    """Camera simulators (NuRec / AlpaSim / CARLA RGB): the rendered HWC-uint8 RGB frame."""
    data = observation.sensor_data
    if not data:
        return None
    key = camera if camera in data else next(iter(data))
    return _as_rgb(data[key])


def annotate(image: Any, detections: Any, threshold: float = 0.3) -> Any:
    """Draw detection boxes on an RGB frame. ``detections`` = iterable of ``(xyxy, score, label)``."""
    import numpy as np
    from PIL import Image, ImageDraw

    im = Image.fromarray(np.asarray(image).astype("uint8")).convert("RGB")
    d = ImageDraw.Draw(im)
    for box, score, label in detections:
        col = (40, 200, 40) if score >= threshold else (230, 60, 60)
        if score >= threshold:
            d.rectangle([box[0], box[1], box[2], box[3]], outline=col, width=4)
        d.text((box[0] + 3, max(box[1] - 14, 2)), f"{label} {score:.2f}", fill=col)
    return np.asarray(im)


def detections_view(
    detect: Callable[[Any], Any], camera: str | None = None, threshold: float = 0.3
) -> Callable[[Observation], Any]:
    """A ``visualize`` that overlays detections. ``detect(rgb) -> [(xyxy, score, label)]`` is injected
    (run your model in it), so this module stays free of any perception dependency."""

    def _view(observation: Observation) -> Any:
        rgb = camera_view(observation, camera)
        return None if rgb is None else annotate(rgb, detect(rgb), threshold=threshold)

    return _view


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


def filmstrip(images: list, cols: int = 4, pad: int = 6, bg=(20, 20, 20)) -> Any:
    """Composite a list of RGB frames into a single grid image (a contact sheet) for quick viewing."""
    import numpy as np
    from PIL import Image

    tiles = [Image.fromarray(np.asarray(im).astype("uint8")) for im in images]
    w, h = tiles[0].size
    cols = min(cols, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w + (cols + 1) * pad, rows * h + (rows + 1) * pad), bg)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet.paste(t, (pad + c * (w + pad), pad + r * (h + pad)))
    return np.asarray(sheet)


def save_gif(images: list, path: str | Path, fps: int = 5) -> None:
    """Save a sequence of RGB frames as an animated GIF."""
    import numpy as np
    from PIL import Image

    frames = [Image.fromarray(np.asarray(im).astype("uint8")) for im in images]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(str(path), save_all=True, append_images=frames[1:],
                   duration=int(1000 / max(fps, 1)), loop=0)


def record_run(
    backend: WorldBackend,
    stack: AVStack,
    frames: int,
    *,
    out_dir: str | Path | None = None,
    visualize: Callable[[Observation], Any] = camera_view,
    perturb: Callable[[Observation], Observation] | None = None,
    prefix: str = "frame",
    collect: bool = False,
) -> Trace:
    """Drive ``stack`` in ``backend`` for ``frames`` steps, saving ``visualize(obs)`` per frame.

    Same loop and Trace as :func:`avsectester.backend.run`; the only addition is writing each frame's
    scene view to ``out_dir`` (default ``<repo>/tmp/frames``). With ``collect=True`` the RGB frames are
    also kept on ``trace.frames`` for :func:`filmstrip` / :func:`save_gif`. Returns the driving Trace.
    """
    out = Path(out_dir) if out_dir is not None else REPO_TMP / "frames"
    out.mkdir(parents=True, exist_ok=True)
    obs = backend.reset()
    stack.reset(obs)
    trace = Trace()
    collected: list = []
    for i in range(frames):
        seen = perturb(obs) if perturb is not None else obs
        control = stack(seen)
        image = visualize(obs)
        if image is not None:
            save_image(image, out / f"{prefix}_{i:04d}.png")
            if collect:
                collected.append(image)
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
    if collect:
        trace.frames = collected  # the saved RGB frames, for filmstrip()/save_gif()
    return trace
