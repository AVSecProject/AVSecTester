"""Scene visualization — the **simulator-agnostic** interface.

A *view* is a ``Callable[[Observation], ndarray|None]`` that turns one frame's observation into an RGB
image (or None to skip). The pipeline here — :func:`record_run` (drive + save per frame),
:func:`detections_view` (overlay boxes), :func:`filmstrip` / :func:`save_gif` (assemble a sequence) —
is generic and works for **any** backend. The default :func:`camera_view` handles the *canonical*
payload (a raw RGB ndarray, e.g. NuRec/AlpaSim); a simulator whose sensor payload is its own type
provides its own view adapter in ``simulators/<sim>.py`` (e.g. :mod:`avsectester.simulators.carla`
extracts an avstack ``ImageData`` and draws a CARLA lidar BEV). Pass whichever view fits the backend.

(This is the *scene* view; the clean-vs-attacked driving-impact plot is the metric view in
:mod:`avsectester.viz`.)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from avsectester.backend import AVStack, WorldBackend
from avsectester.plane import FrameRecord, Observation, Trace

View = Callable[[Observation], Any]  # Observation -> HWC-uint8 RGB ndarray, or None to skip
REPO_TMP = Path(__file__).resolve().parents[2] / "tmp"  # <repo>/tmp (gitignored)


def as_rgb(frame: Any) -> Any:
    """Canonical camera payload -> ``(H, W, 3)`` uint8 RGB ndarray, or None if it is not a raw image.

    Only the canonical form (a numpy array) is handled here; simulator-specific payloads (e.g. avstack
    ``ImageData``) are unwrapped by that simulator's view adapter before calling this.
    """
    import numpy as np

    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3:
        return None
    return frame[:, :, :3].astype("uint8")


def camera_view(observation: Observation, camera: str | None = None) -> Any:
    """Default view: the rendered RGB frame when the sensor payload is already a raw ndarray."""
    data = observation.sensor_data
    if not data:
        return None
    key = camera if camera in data else next(iter(data))
    return as_rgb(data[key])


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
    detect: Callable[[Any], Any], base: View = camera_view, threshold: float = 0.3
) -> View:
    """Wrap any camera ``base`` view to overlay detections. ``detect(rgb) -> [(xyxy, score, label)]``
    is injected (run your model in it), so this module needs no perception dependency; ``base`` is the
    backend's camera view (e.g. :func:`avsectester.simulators.carla.camera_view`)."""

    def _view(observation: Observation) -> Any:
        rgb = base(observation)
        return None if rgb is None else annotate(rgb, detect(rgb), threshold=threshold)

    return _view


def detector_quad(detect: Callable[[Any], Any], base: View = camera_view, width_frac: float = 0.7,
                  height_frac: float = 0.55, v_center: float = 0.5, yaw: float = 0.0,
                  central: float = 0.25) -> Callable[[Observation], Any]:
    """Return ``quad_of(observation) -> (4,2) | None``: the rear-face quad of the lead vehicle,
    approximated from a 2-D detector box (image-space, no depth). Runs ``detect(rgb) -> [(xyxy, score,
    label)]`` on the ``base`` view, picks the largest box near the image centre (the lead), and turns it
    into a planar-warp quad via :func:`avsectester.simulators.patch_insertion.box_to_quad`. Feeds
    :func:`composite_view` — the NuRec/real-imagery analog of the geometric ``carla.lead_rear_quad``."""
    from avsectester.simulators.patch_insertion import box_to_quad

    def _quad_of(observation: Observation) -> Any:
        rgb = base(observation)
        if rgb is None:
            return None
        w = rgb.shape[1]
        best = None
        for box, _score, _label in detect(rgb):
            cx = (box[0] + box[2]) / 2.0
            area = (box[2] - box[0]) * (box[3] - box[1])
            if abs(cx - w / 2) < central * w and (best is None or area > best[1]):
                best = (box, area)
        if best is None:
            return None
        return box_to_quad(best[0], width_frac=width_frac, height_frac=height_frac,
                           v_center=v_center, yaw=yaw)

    return _quad_of


def composite_view(
    compositor: Any, patch_rgba: Any, quad_of: Callable[[Observation], Any], base: View = camera_view
) -> View:
    """Wrap a camera ``base`` view to insert a harmonized patch onto the rendered frame.

    The mirror of :func:`detections_view`, at the pixel-insertion layer: ``quad_of(observation)`` gives
    the target surface as a ``(4, 2)`` image-space quad (TL, TR, BR, BL) — the backend-specific
    projection is injected (e.g. :func:`avsectester.simulators.carla.lead_rear_quad`), so this module
    stays simulator-agnostic — and ``compositor`` (a
    :class:`avsectester.simulators.patch_insertion.PatchCompositor`) warps + harmonizes ``patch_rgba``
    onto it. Returns the clean frame unchanged when ``quad_of`` yields None (target not in view)."""

    def _view(observation: Observation) -> Any:
        rgb = base(observation)
        if rgb is None:
            return None
        quad = quad_of(observation)
        return rgb if quad is None else compositor.apply(rgb, quad, patch_rgba)

    return _view


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


def save_sequence(frames: list, out_dir: str | Path, name: str = "sequence",
                  cols: int = 4, fps: int = 4) -> tuple[str, str] | None:
    """Save a captured frame sequence as both a filmstrip PNG and an animated GIF — the one place
    demos turn ``trace.frames`` into output, so no script re-implements the filmstrip+gif dance.

    Writes ``<out_dir>/<name>_filmstrip.png`` and ``<out_dir>/<name>.gif``; returns their paths (or
    None when there are no frames). ``collect=True`` on :func:`record_run` fills ``trace.frames``.
    """
    if not frames:
        return None
    out = Path(out_dir)
    strip = out / f"{name}_filmstrip.png"
    gif = out / f"{name}.gif"
    save_image(filmstrip(frames, cols=cols), strip)
    save_gif(frames, gif, fps=fps)
    return str(strip), str(gif)


def record_run(
    backend: WorldBackend,
    stack: AVStack,
    frames: int,
    *,
    out_dir: str | Path | None = None,
    visualize: View = camera_view,
    perturb: Callable[[Observation], Observation] | None = None,
    prefix: str = "frame",
    collect: bool = False,
) -> Trace:
    """Drive ``stack`` in ``backend`` for ``frames`` steps, saving ``visualize(seen)`` per frame.

    Same loop and Trace as :func:`avsectester.backend.run`; the only addition is writing each frame's
    scene view to ``out_dir`` (default ``<repo>/tmp/frames``). ``visualize`` is any backend's view.
    With ``collect=True`` the RGB frames are also kept on ``trace.frames`` for :func:`filmstrip` /
    :func:`save_gif`. Returns the driving Trace.
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
        image = visualize(seen)  # visualize what the stack perceives (== obs when no perturb)
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
