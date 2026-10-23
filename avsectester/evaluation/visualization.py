"""Images, overlays and animations exported from RunLogs after simulators have closed."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from avsectester.data_types import Observation
from avsectester.closed_loop.simulators.base import SensorView
from avsectester.geometry.cameras import BODY_FROM_OPTICAL, FThetaCamera, PinholeCamera, WindshieldDistortion
from avsectester.geometry.poses import transform
from avsectester.data_types import ActorState
from avsectester.runlog import RunLog, StepRecord

from .metric import executed_brakes, speeds

View = Callable[[StepRecord], Any]
_BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))


def _get(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def as_rgb(frame: Any) -> np.ndarray | None:
    """Read canonical pixels or a logged native camera payload without importing its SDK."""
    rgb = _get(frame, "rgb_image")
    if rgb is not None:
        frame = rgb
    elif isinstance(frame, Mapping):
        pixels = frame.get("data")
        if pixels is not None:
            decoded = as_rgb(pixels)
            calibration = frame.get("calibration", {})
            return decoded[:, :, ::-1] if decoded is not None and _get(calibration, "channel_order", "rgb").lower() == "bgr" else decoded
        raw = frame.get("raw_bytes")
        metadata = frame.get("metadata", {})
        if raw is not None and "width" in metadata and "height" in metadata:
            pixels = np.frombuffer(raw, dtype=np.uint8).reshape(metadata["height"], metadata["width"], 4)
            return pixels[:, :, [2, 1, 0]].copy()
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] < 3:
        return None
    return frame[:, :, :3].astype(np.uint8)


def _observation(record: StepRecord | Observation) -> Observation | None:
    if isinstance(record, Observation):
        return record
    return record.model_observation if record.model_observation is not None else record.raw_observation


def camera_view(record: StepRecord | Observation, camera: str | None = None) -> np.ndarray | None:
    """Read a recorded model-visible camera image (or an explicit Observation)."""
    observation = _observation(record)
    if observation is None or not observation.sensor_data:
        return None
    if camera is not None:
        return as_rgb(observation.sensor_data.get(camera))
    for payload in observation.sensor_data.values():
        image = as_rgb(payload)
        if image is not None:
            return image
    return None


def lidar_bev(record: StepRecord | Observation, size: int = 800, meters: float = 60.0) -> np.ndarray | None:
    """Read a saved lidar cloud and draw x-forward up, y-left left. Non-lidar payloads skip."""
    if size <= 0 or meters <= 0:
        raise ValueError("LiDAR view size and range must be positive")
    observation = _observation(record)
    if observation is None:
        return None
    points = None
    for payload in observation.sensor_data.values():
        data = payload if isinstance(payload, np.ndarray) else _get(payload, "data", payload)
        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 3:
            points = data[:, :3]
            break
        raw = _get(data, "raw_data", _get(data, "raw_bytes"))
        if raw is not None and "lidar" in str(_get(data, "native_type", type(data).__name__)).lower():
            points = np.frombuffer(bytes(raw), dtype=np.float32).reshape(-1, 4)[:, :3]
            break
    if points is None:
        return None
    image = np.zeros((size, size, 3), dtype=np.uint8)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    u = (size / 2 - points[:, 0] * size / (2 * meters)).astype(int)
    v = (size / 2 - points[:, 1] * size / (2 * meters)).astype(int)
    inside = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    image[u[inside], v[inside]] = (0, 255, 0)
    image[max(0, size // 2 - 3):size // 2 + 3, max(0, size // 2 - 3):size // 2 + 3] = (255, 80, 80)
    return image


def annotate(image: Any, detections: Sequence[Any], threshold: float = 0.3) -> np.ndarray:
    """Draw explicit 2-D labels ``(xyxy, score, text[, color])`` on saved image pixels."""
    from PIL import Image, ImageDraw

    result = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(result)
    for detection in detections:
        box, score, label = detection[:3]
        color = detection[3] if len(detection) > 3 else (40, 200, 40)
        if score >= threshold:
            draw.rectangle(list(box), outline=color, width=2)
            draw.text((box[0] + 2, max(box[1] - 12, 2)), str(label), fill=color)
    return np.asarray(result)


def box3d_corners(center: Any, extent: Any, yaw: float) -> np.ndarray:
    """Eight cuboid corners from full x/y/z extents and right-handed yaw, in radians."""
    signs = np.array([[-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1],
                      [-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1]])
    points = signs * np.asarray(extent) / 2
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return points @ rotation.T + np.asarray(center)


def _actor_corners(actor: ActorState) -> np.ndarray | None:
    if actor.extent is None:
        return None
    return transform(actor.box_transform(), box3d_corners((0, 0, 0), actor.extent, 0))


def _draw_projected(image: Any, view: SensorView, boxes: Sequence[tuple[Any, str, Any]],
                    subdiv: int) -> np.ndarray:
    from PIL import Image, ImageDraw

    if subdiv < 1:
        raise ValueError("subdiv must be positive")
    if view.camera is None or view.cam_from_world is None:
        raise ValueError("Drawing projected boxes requires camera geometry")
    result = Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(result)
    interpolation = np.linspace(0, 1, subdiv + 1)[:, None]
    for world_corners, label, color in boxes:
        corners = transform(view.cam_from_world, world_corners)
        if np.any(corners[:, 2] <= 1e-3):
            continue
        visible = getattr(view.camera, "in_view", None)
        if visible is not None and not np.all(visible(corners)):
            continue
        projected = view.camera.project(corners)
        if not np.isfinite(projected).all():
            continue
        for first, second in _BOX_EDGES:
            pixels = view.camera.project(corners[first] + interpolation * (corners[second] - corners[first]))
            if np.isfinite(pixels).all():
                draw.line([tuple(map(float, pixel)) for pixel in pixels], fill=color, width=2)
        top = projected[np.argmin(projected[:, 1])]
        draw.text((float(top[0]) + 2, max(float(top[1]) - 12, 2)), label, fill=color)
    return np.asarray(result)


def draw_boxes3d(image: Any, view: SensorView, *, actors: Mapping[str, Any] | None = None,
                 target: str | None = None, subdiv: int = 8, max_distance: float | None = None,
                 color: tuple[int, int, int] = (40, 200, 40),
                 target_color: tuple[int, int, int] = (235, 64, 52)) -> np.ndarray:
    """Project ActorState cuboids through a SensorView, without SceneGT.

    This also supports dataset inspection: pass FrameData.actors explicitly. Ordinary run
    overlays use the physical_pre actors stored in each StepRecord.
    """
    boxes = []
    ego_pose = view.ego.box_transform()
    origin = np.asarray(ego_pose)[:3, 3]
    for identity, actor in sorted((view.actors if actors is None else actors).items(), key=lambda item: item[0] == target):
        corners = _actor_corners(actor)
        if corners is None:
            continue
        distance = float(np.linalg.norm(corners.mean(axis=0) - origin))
        if max_distance is not None and distance > max_distance and identity != target:
            continue
        label = f"{_get(actor, 'category', None) or identity} {distance:.0f}m"
        boxes.append((corners, label, target_color if identity == target else color))
    return _draw_projected(image, view, boxes, subdiv)


def _camera_model(value: Any, width: int, height: int) -> Any:
    if callable(getattr(value, "project", None)):
        return value
    if isinstance(value, Mapping):
        if "angle_to_pixeldist" in value:
            options = {name: item for name, item in value.items() if name in FThetaCamera.__dataclass_fields__}
            if isinstance(options.get("windshield"), Mapping):
                options["windshield"] = WindshieldDistortion(**options["windshield"])
            return FThetaCamera(**options)
        if "K" in value:
            return PinholeCamera(np.asarray(value["K"]), int(value.get("width", width)), int(value.get("height", height)))
    matrix = np.asarray(value)
    if matrix.shape == (3, 3):
        return PinholeCamera(matrix, width, height)
    raise ValueError("Recorded camera calibration does not contain a supported lens")


def sensor_view(record: StepRecord, camera: str | None = None) -> SensorView | None:
    """Rebuild projection geometry from recorded canonical sensor-view metadata/calibration.

    No simulator or native camera class is loaded. Unrecorded geometry returns None rather
    than drawing boxes with an invented extrinsic or an incorrect coordinate system.
    """
    observation = _observation(record)
    if observation is None or record.physical_pre is None or record.physical_pre.ego is None:
        return None
    sensors = [camera] if camera is not None else list(observation.sensor_data)
    for name in sensors:
        image = camera_view(record, name)
        if image is None:
            continue
        geometry = record.sensor_geometry.get(name)
        canonical = ({"camera": geometry.camera, "cam_from_world": geometry.cam_from_world}
                     if geometry is not None else None)
        calibration = observation.calibration.get(name, {})
        if canonical is None and isinstance(calibration, Mapping) and "camera" in calibration and "ego_from_sensor" in calibration:
            canonical = {"camera": calibration["camera"], "cam_from_world": np.linalg.inv(
                np.asarray(record.physical_pre.ego.pose) @ np.asarray(calibration["ego_from_sensor"]))}
        if canonical is None:
            continue
        lens = _camera_model(canonical["camera"], image.shape[1], image.shape[0])
        cam_from_world = np.asarray(canonical["cam_from_world"])
        return SensorView(name, image, lens, cam_from_world,
                          record.physical_pre.actors, record.physical_pre.ego,
                          observation.t, observation.frame,
                          sensor_from_world=BODY_FROM_OPTICAL @ cam_from_world)
    return None


def labels_view(record: StepRecord, camera: str | None = None, *, target: str | None = None,
                max_distance: float | None = None) -> np.ndarray | None:
    """Overlay the saved physical_pre actors, using its real ego pose and recorded camera."""
    image = camera_view(record, camera)
    view = sensor_view(record, camera)
    return image if image is None or view is None else draw_boxes3d(
        image, view, target=target, max_distance=max_distance,
    )


def detections_view(record: StepRecord, camera: str | None = None, *, threshold: float = 0.3) -> np.ndarray | None:
    """Overlay StepRecord.detections; this never runs a detector or reads components."""
    image, view = camera_view(record, camera), sensor_view(record, camera)
    if image is None or view is None or record.detections is None:
        return image
    boxes = []
    for detection in record.detections:
        if detection.score is not None and detection.score < threshold:
            continue
        local = box3d_corners(detection.position, detection.extent, detection.yaw)
        corners = transform(record.physical_pre.ego.pose, local)
        boxes.append((corners, str(detection.category or "detection"), (40, 200, 40)))
    return _draw_projected(image, view, boxes, 8)


def save_image(image: Any, path: str | Path) -> None:
    """Write an RGB frame without a plotting backend."""
    from PIL import Image

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image).astype(np.uint8)).save(destination)


def _images(source: RunLog | Sequence[Any], visualize: View) -> list[np.ndarray]:
    if isinstance(source, RunLog):
        return [image for step in source.steps if (image := visualize(step)) is not None]
    return [np.asarray(image) for image in source]


def record_run(log: RunLog, out_dir: str | Path, *, visualize: View = camera_view,
               prefix: str = "frame") -> tuple[Path, ...]:
    """Export a frame per recorded decision. Does not run an experiment or simulator."""
    paths = []
    for record in log.steps:
        image = visualize(record)
        if image is not None:
            path = Path(out_dir) / f"{prefix}_{record.step:04d}.png"
            save_image(image, path)
            paths.append(path)
    return tuple(paths)


def filmstrip(source: RunLog | Sequence[Any], cols: int = 4, pad: int = 6,
              bg: tuple[int, int, int] = (20, 20, 20), *, visualize: View = camera_view) -> np.ndarray:
    """Assemble recorded camera frames into one contact sheet."""
    from PIL import Image

    frames = _images(source, visualize)
    if not frames or cols < 1 or pad < 0:
        raise ValueError("Filmstrip needs frames, positive cols and nonnegative padding")
    tiles = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
    width, height = tiles[0].size
    if any(tile.size != (width, height) for tile in tiles):
        raise ValueError("Filmstrip frames must have equal dimensions")
    cols = min(cols, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    result = Image.new("RGB", (cols * width + (cols + 1) * pad, rows * height + (rows + 1) * pad), bg)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, cols)
        result.paste(tile, (pad + column * (width + pad), pad + row * (height + pad)))
    return np.asarray(result)


def save_gif(source: RunLog | Sequence[Any], path: str | Path, fps: int = 5,
             *, visualize: View = camera_view) -> None:
    """Write an animation from recorded frames, without model or simulator imports."""
    from PIL import Image

    frames = [Image.fromarray(image.astype(np.uint8)) for image in _images(source, visualize)]
    if not frames or fps <= 0:
        raise ValueError("GIF needs frames and positive fps")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(destination, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0)


def save_sequence(log: RunLog, out_dir: str | Path, name: str = "sequence", cols: int = 4,
                  fps: int = 4, *, visualize: View = camera_view) -> tuple[str, str] | None:
    """Export a contact sheet and GIF from one recorded run; no frames yields None."""
    frames = _images(log, visualize)
    if not frames:
        return None
    strip, gif = Path(out_dir) / f"{name}_filmstrip.png", Path(out_dir) / f"{name}.gif"
    save_image(filmstrip(frames, cols), strip)
    save_gif(frames, gif, fps)
    return str(strip), str(gif)


def plot_group(runs: Mapping[str, Any], path: str | Path, *, title: str | None = None) -> Path:
    """Plot physical speeds and executed brakes without changing matplotlib's global backend."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 6))
    FigureCanvasAgg(figure)
    speed_axis, brake_axis = figure.subplots(2, 1, sharex=True)
    if title:
        figure.suptitle(title)
    for name, log in runs.items():
        positions = [step.step for step in log.steps]
        physical_speeds, braking = speeds(log), executed_brakes(log)
        if physical_speeds is not None:
            speed_axis.plot(positions, physical_speeds, label=name)
        if braking is not None:
            brake_axis.plot(positions, braking, label=name)
    speed_axis.set_ylabel("ego speed (m/s)")
    brake_axis.set_ylabel("executed brake")
    brake_axis.set_xlabel("decision step")
    for axis in (speed_axis, brake_axis):
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend()
    figure.tight_layout()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=130)
    figure.clear()
    return destination
