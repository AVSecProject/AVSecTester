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
from copy import deepcopy

from avsectester.backend import AVStack, WorldBackend, run
from avsectester.plane import Observation, Trace
from avsectester.runtime import Runtime

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
    """Draw boxes on an RGB frame — the one box-drawing primitive, shared by detections and GT labels.

    ``detections`` = iterable of ``(xyxy, score, label)``, or ``(xyxy, score, label, color)`` to force the
    box colour. A 3-tuple is a *detection*: coloured green/red by ``score`` vs ``threshold`` and labelled
    ``"label score"``. A 4-tuple is a *label* (e.g. from :func:`scene_labels`): drawn in its own colour
    with the score omitted from the text. Boxes below ``threshold`` are labelled but not outlined.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    im = Image.fromarray(np.asarray(image).astype("uint8")).convert("RGB")
    d = ImageDraw.Draw(im)
    for det in detections:
        box, score, label = det[0], det[1], det[2]
        explicit = len(det) > 3
        col = det[3] if explicit else ((40, 200, 40) if score >= threshold else (230, 60, 60))
        if score >= threshold:
            d.rectangle([box[0], box[1], box[2], box[3]], outline=col, width=4)
        d.text((box[0] + 3, max(box[1] - 14, 2)), label if explicit else f"{label} {score:.2f}", fill=col)
    return np.asarray(im)


def scene_labels(scene: Any, camera: str | None = None, target: Any = None,
                 color=(40, 200, 40), target_color=(235, 64, 52)) -> list:
    """Ground-truth :class:`SceneGT` objects -> :func:`annotate` tuples, so scene labels reuse the same
    drawing primitive as detections.

    Returns ``(xyxy, 1.0, "category dist", color)`` for every object with a 2-D box in ``camera``
    (default: the scene's first camera). ``target`` (a ``ScenarioMatch.target`` / ``ObjectGT``, or
    anything with a ``track_id``) is drawn in ``target_color`` and last, so its box sits on top.
    Duck-typed — reads ``.objects``/``.cameras``/``.box2d``/``.category``/``.distance``/``.track_id`` — so
    this module keeps no dependency on the scenarios package.
    """
    cam = camera if (camera and camera in scene.cameras) else next(iter(scene.cameras))
    tid = getattr(target, "track_id", None)
    others, hit = [], []
    for o in scene.objects:
        box = o.box2d.get(cam)
        if box is None:
            continue
        text = f"{o.category} {o.distance:.0f}m"
        if tid is not None and o.track_id == tid:
            hit.append((box, 1.0, text, target_color))
        else:
            others.append((box, 1.0, text, color))
    return others + hit


# the 12 edges of a cuboid whose 8 corners are ordered by (sx, sy, sz) bits (see :func:`box3d_corners`)
_BOX_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if (i ^ j).bit_count() == 1]


def box3d_corners(center: Any, extent: Any, yaw: float) -> Any:
    """The 8 corners of a 3-D box in the ego frame ``(8, 3)``. ``center`` (x fwd, y left, z up),
    ``extent`` (length, width, height), ``yaw`` about +z. Corners are ordered by the sign bits
    ``(sx, sy, sz)`` so ``_BOX_EDGES`` connects the ones differing in exactly one axis."""
    import numpy as np

    length, width, height = extent
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    local = np.array([[sx * length / 2, sy * width / 2, sz * height / 2]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return (rot @ local.T).T + np.asarray(center, dtype=float)


def ego_projector(model: Any) -> Any:
    """A ``points_ego (N,3) -> (pixels (N,2), valid (N,))`` projector derived from a camera ``model``, or
    None if it cannot project. Reuses a model that already projects (nuRec's ``FThetaCamera.project``,
    fisheye-correct); for a raw 3x3 pinhole ``K`` (nuScenes) it maps ego (x fwd, y left, z up) -> camera
    (x right, y down, z fwd) = ``(-y, -z, x)`` then applies ``K``. Duck-typed — no dataset imports."""
    import numpy as np

    if hasattr(model, "project"):
        return model.project
    k = np.asarray(model) if model is not None else None
    if k is None or k.shape != (3, 3):
        return None

    def _pinhole(pts: Any):
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        cam = np.stack([-pts[:, 1], -pts[:, 2], pts[:, 0]], axis=1)
        valid = cam[:, 2] > 1e-3
        z = np.where(cam[:, 2] == 0, 1.0, cam[:, 2])
        px = (k @ cam.T)[:2] / z
        return px.T, valid

    return _pinhole


def draw_boxes3d(image: Any, scene: Any, camera: str | None = None, target: Any = None,
                 subdiv: int = 8, max_distance: float | None = None,
                 color=(40, 200, 40), target_color=(235, 64, 52)) -> Any:
    """Draw ground-truth **3-D bounding boxes** (projected cuboid wireframes) on an RGB frame — the right
    representation for a 3-D scene. Each of the 12 edges is subdivided in 3-D and projected point-by-point,
    so a straight edge renders as the correct **curve** under a fisheye (f-theta) camera, not an
    over-covering axis-aligned rectangle. ``target`` is highlighted and drawn last. ``max_distance`` (m)
    drops far actors so the frame is not swamped by tiny distant boxes. Falls back to :func:`annotate` +
    :func:`scene_labels` (2-D boxes) when the camera model cannot project 3-D points."""
    import numpy as np
    from PIL import Image, ImageDraw

    cam = camera if (camera and camera in scene.cameras) else next(iter(scene.cameras))
    project = ego_projector(scene.cameras[cam].model)
    if project is None:
        return annotate(image, scene_labels(scene, cam, target, color, target_color))

    im = Image.fromarray(np.asarray(image).astype("uint8")).convert("RGB")
    draw = ImageDraw.Draw(im)
    tid = getattr(target, "track_id", None)
    ts = np.linspace(0.0, 1.0, subdiv + 1)[:, None]
    for obj in sorted(scene.objects, key=lambda o: o.track_id == tid):  # target last (drawn on top)
        if max_distance is not None and obj.distance > max_distance and obj.track_id != tid:
            continue
        corners = box3d_corners(obj.center, obj.extent, obj.yaw)
        corner_px, corner_valid = project(corners)
        # Only draw a fully-projectable box. A box with any corner behind the camera or outside the
        # fisheye field of view (e.g. an actor closer than a few metres) would otherwise project to
        # nonsense edges spanning the whole frame, so it is skipped rather than drawn wrong.
        if not corner_valid.all():
            continue
        col = target_color if (tid is not None and obj.track_id == tid) else color
        for a, b in _BOX_EDGES:  # subdivide each edge so it curves correctly under the fisheye
            px, _ = project(corners[a] + ts * (corners[b] - corners[a]))
            draw.line([(float(x), float(y)) for x, y in px], fill=col, width=2)
        top = corner_px[np.argmin(corner_px[:, 1])]
        draw.text((float(top[0]) + 2, max(float(top[1]) - 12, 2)),
                  f"{obj.category} {obj.distance:.0f}m", fill=col)
    return np.asarray(im)


def labels_view(get_scene: Callable[[Observation], Any], base: View = camera_view,
                camera: str | None = None, target: Any = None, boxes3d: bool = True,
                max_distance: float | None = None) -> View:
    """Wrap any camera ``base`` view to overlay ground-truth scene labels — the GT counterpart of
    :func:`detections_view`. ``boxes3d=True`` draws projected 3-D bounding boxes (:func:`draw_boxes3d`,
    honouring ``max_distance``); ``boxes3d=False`` draws 2-D boxes (:func:`annotate` + :func:`scene_labels`).
    ``get_scene(obs) -> SceneGT`` (or None) supplies the frame's ground truth; returns the plain frame when
    it is None."""

    def _view(observation: Observation) -> Any:
        rgb = base(observation)
        if rgb is None:
            return None
        scene = get_scene(observation)
        if scene is None:
            return rgb
        return (draw_boxes3d(rgb, scene, camera, target, max_distance=max_distance) if boxes3d
                else annotate(rgb, scene_labels(scene, camera, target)))

    return _view


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
    runtime: Runtime | None = None,
) -> Trace:
    """Drive ``stack`` in ``backend`` for ``frames`` steps, saving ``visualize(seen)`` per frame.

    Same loop and Trace as :func:`avsectester.backend.run`; the only addition is writing each frame's
    scene view to ``out_dir`` (default ``<repo>/tmp/frames``). ``visualize`` is any backend's view.
    With ``collect=True`` the RGB frames are also kept on ``trace.frames`` for :func:`filmstrip` /
    :func:`save_gif`. Returns the driving Trace.
    """
    out = Path(out_dir) if out_dir is not None else REPO_TMP / "frames"
    out.mkdir(parents=True, exist_ok=True)
    collected: list = []

    def capture(i, seen, control):
        image = visualize(seen)  # visualize what the stack perceives (== obs when no perturb)
        if image is not None:
            save_image(image, out / f"{prefix}_{i:04d}.png")
            if collect:
                collected.append(deepcopy(image))

    trace = run(backend, stack, frames, perturb=perturb, on_step=capture, runtime=runtime)
    if collect:
        trace.frames = collected  # the saved RGB frames, for filmstrip()/save_gif()
    return trace
