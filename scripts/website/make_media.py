"""Regenerate the homepage media from saved AVSecTester experiment outputs.

Every subcommand reads one experiment output directory (the configured ``runs.output``) through
the documented output layout ``runs/segment-<NNNNN>/seed-<S>/<treatment>/`` and the public
``avsectester.runlog`` and ``avsectester.evaluation.visualization`` APIs; ``stop-line`` loads it
with ``ExperimentResult`` from its configuration to read the saved stop line. Nothing is simulated
or inferred again. Run it from the repository root, in the experiment environment
(for example the ``nurec-experiment`` Compose service with ``--entrypoint python``), with
``imageio-ffmpeg`` installed for the videos.

Regenerate the media of a run directory, for example:

    python -m scripts.website.make_media compare OUTPUT --segment 5 --clean clean \\
        --attacked white_detector_pgd --name stop-sign-white-box
    python -m scripts.website.make_media overlay OUTPUT --segment 2 \\
        --treatment white_rollout_nes --name stop-sign-black-box
    python -m scripts.website.make_media frame OUTPUT --segment 0 --treatment object \\
        --step 19 --crop 990 560 1630 920 --width 640 --name image-insertion
    python -m scripts.website.make_media stop-line CONFIG --segment 5 \\
        --attacked white_detector_pgd --name stop-line-approach
    python -m scripts.website.make_media still IMAGE --name evaluation-report

Videos are H.264 MP4 files with a WebP poster; frames and screenshots are WebP, plots are PNG.
Each command prints the written files and the decisions it used, for the media report.
"""

from __future__ import annotations

import argparse
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from avsectester.evaluation.visualization import annotate, camera_view
from avsectester.geometry import FThetaCamera, PinholeCamera
from avsectester.runlog import RunLog, StepRecord

DEFAULT_CAMERA = "camera_front_wide_120fov"
DEFAULT_OUTPUT = Path("website/content/assets/media")
MAX_VIDEO_WIDTH = 1280
MAX_IMAGE_WIDTH = 1600
MAX_VIDEO_SECONDS = 15.0


# Reading saved runs -----------------------------------------------------------------------------

def read_run(output: Path, segment: int, treatment: str, seed: int = 0) -> RunLog:
    """Read one saved treatment run from an experiment output directory."""
    directory = Path(output) / "runs" / f"segment-{segment:05d}" / f"seed-{seed}" / treatment
    if not (directory / "manifest.json").is_file():
        raise FileNotFoundError(f"No saved run for segment {segment}, seed {seed}, treatment {treatment!r}")
    log = RunLog.read(directory)
    if not log.steps:
        raise ValueError(f"Run {directory} has no recorded decisions")
    return log


def _get(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def frame_of(record: StepRecord, camera: str) -> np.ndarray:
    """The model-visible camera image of one recorded decision."""
    image = camera_view(record, camera)
    if image is None:
        raise ValueError(f"Step {record.step} has no recorded {camera!r} image")
    return np.asarray(image, dtype=np.uint8)


def times_of(log: RunLog) -> np.ndarray:
    """Recorded decision times relative to the first decision, in seconds."""
    stamps = [_get(step.physical_pre, "t") for step in log.steps]
    if any(stamp is None for stamp in stamps):
        return np.arange(len(log.steps), dtype=float)
    return np.asarray(stamps, dtype=float) - float(stamps[0])


def physical_speeds(log: RunLog) -> np.ndarray:
    """True ego speed at each recorded decision, in m/s."""
    return np.asarray([float(_get(step.physical_pre, "ego_speed", np.nan)) for step in log.steps])


# Image helpers ----------------------------------------------------------------------------------

def _font(size: int) -> Any:
    from PIL import ImageFont

    return ImageFont.load_default(size=size)


def resize_width(image: np.ndarray, width: int) -> np.ndarray:
    from PIL import Image

    image = np.asarray(image, dtype=np.uint8)
    if image.shape[1] == width:
        return image
    height = round(image.shape[0] * width / image.shape[1])
    return np.asarray(Image.fromarray(image).resize((width, height), Image.LANCZOS))


def even_size(image: np.ndarray, multiple: int = 16) -> np.ndarray:
    """Pad to a multiple of ``multiple`` pixels, as H.264 encoders require."""
    height, width = image.shape[:2]
    pad_h, pad_w = (-height) % multiple, (-width) % multiple
    if not pad_h and not pad_w:
        return image
    return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=255)


def label(image: np.ndarray, text: str, *, size: int = 26) -> np.ndarray:
    """Draw a readable label in the top-left corner."""
    from PIL import Image, ImageDraw

    result = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(result, "RGBA")
    font = _font(size)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    pad = size // 3
    draw.rectangle((0, 0, right - left + 2 * pad, bottom - top + 2 * pad), fill=(0, 0, 0, 170))
    draw.text((pad - left, pad - top), text, fill=(255, 255, 255), font=font)
    return np.asarray(result)


def save_still(image: np.ndarray, path: Path, *, max_width: int = MAX_IMAGE_WIDTH) -> Path:
    from PIL import Image

    image = np.asarray(image, dtype=np.uint8)
    if image.shape[1] > max_width:
        image = resize_width(image, max_width)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".webp":
        Image.fromarray(image).save(path, quality=82, method=6)
    else:
        Image.fromarray(image).save(path, optimize=True)
    return path


def write_video(frames: Sequence[np.ndarray], path: Path, fps: float, *, crf: int = 28) -> Path:
    """Encode RGB frames as a muted, web-streamable H.264 MP4."""
    import imageio_ffmpeg

    if not frames:
        raise ValueError("A video needs at least one frame")
    if len(frames) / fps > MAX_VIDEO_SECONDS:
        raise ValueError(f"Video would last {len(frames) / fps:.1f} s; raise fps or select fewer decisions")
    height, width = frames[0].shape[:2]
    if width > MAX_VIDEO_WIDTH:
        raise ValueError(f"Video width {width} exceeds {MAX_VIDEO_WIDTH} px")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(path), (width, height), fps=fps, codec="libx264", pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p", quality=None, macro_block_size=16,
        output_params=["-crf", str(crf), "-preset", "slow", "-movflags", "+faststart", "-an"],
    )
    writer.send(None)
    for frame in frames:
        writer.send(np.ascontiguousarray(frame, dtype=np.uint8))
    writer.close()
    return path


# Plots -----------------------------------------------------------------------------------------

def _figure(width_px: int, height_px: int, dpi: int = 100) -> tuple[Any, Any]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    FigureCanvasAgg(figure)
    return figure, figure.add_subplot(1, 1, 1)


def _rgb(figure: Any) -> np.ndarray:
    figure.canvas.draw()
    return np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()


def speed_strip(curves: Mapping[str, tuple[np.ndarray, np.ndarray]], width: int, height: int
                ) -> tuple[np.ndarray, Any]:
    """Plot speed curves and return the image plus a time-to-pixel mapping for a cursor."""
    figure, axis = _figure(width, height)
    for (name, (times, values)), color in zip(curves.items(), ("#1f77b4", "#d62728")):
        axis.plot(times, values, label=name, color=color, linewidth=2)
    axis.set_xlabel("time (s)")
    axis.set_ylabel("speed (m/s)")
    axis.grid(alpha=0.3)
    axis.legend(loc="upper right", fontsize=9)
    figure.tight_layout(pad=0.4)
    image = _rgb(figure)

    def to_pixel(time: float) -> int:
        x, _ = axis.transData.transform((time, 0.0))
        return int(round(x))

    top = int(round(height - axis.get_window_extent().y1))
    bottom = int(round(height - axis.get_window_extent().y0))
    return image, (to_pixel, top, bottom)


def with_cursor(strip: np.ndarray, mapping: Any, time: float) -> np.ndarray:
    to_pixel, top, bottom = mapping
    result = strip.copy()
    x = min(max(to_pixel(time), 0), result.shape[1] - 2)
    result[top:bottom, x:x + 2] = (40, 40, 40)
    return result


# Overlays --------------------------------------------------------------------------------------

def detections_of(record: StepRecord, scale: float = 1.0
                  ) -> list[tuple[Any, float, str, tuple[int, int, int]]]:
    """Recorded 2-D detector boxes from the driving system's component log, scaled to a view."""
    boxes = []
    for detection in _get(record.components, "detections", None) or ():
        box, score = _get(detection, "box_xyxy"), _get(detection, "score")
        if box is not None and score is not None:
            boxes.append((tuple(float(v) * scale for v in box), float(score),
                          f"stop sign {float(score):.2f}", (255, 210, 0)))
    return boxes


def _lens(value: Any) -> Any:
    if callable(getattr(value, "project", None)):
        return value
    if isinstance(value, Mapping) and "angle_to_pixeldist" in value:
        fields = {name: item for name, item in value.items() if name in FThetaCamera.__dataclass_fields__}
        windshield = fields.get("windshield")
        if isinstance(windshield, Mapping):
            from avsectester.geometry import WindshieldDistortion

            fields["windshield"] = WindshieldDistortion(**windshield)
        return FThetaCamera(**fields)
    if isinstance(value, Mapping) and "K" in value:
        return PinholeCamera(np.asarray(value["K"]), int(value["width"]), int(value["height"]))
    raise ValueError("Recorded calibration has no supported lens")


def planned_path_pixels(record: StepRecord, camera: str) -> np.ndarray | None:
    """Project the produced rig-frame trajectory into the recorded camera image."""
    trajectory = _get(record.produced_command, "trajectory")
    observation = record.model_observation or record.raw_observation
    calibration = None if observation is None else observation.calibration.get(camera)
    if not trajectory or calibration is None:
        return None
    lens = _lens(_get(calibration, "camera"))
    rig_from_camera = np.asarray(_get(calibration, "rig_from_camera"), dtype=float)
    points = np.asarray([point[0] for point in trajectory], dtype=float)
    homogeneous = np.c_[points, np.ones(len(points))]
    optical = (np.linalg.inv(rig_from_camera) @ homogeneous.T).T[:, :3]
    optical = optical[optical[:, 2] > 0.5]
    if len(optical) < 2:
        return None
    pixels = np.asarray(lens.project(optical), dtype=float)
    return pixels[np.isfinite(pixels).all(axis=1)]


def reasoning_of(record: StepRecord) -> str | None:
    policy = _get(record.components, "policy")
    text = _get(policy, "reasoning_text") if policy is not None else None
    if isinstance(text, (list, tuple)):
        text = " ".join(str(item) for item in np.ravel(np.asarray(text, dtype=object)) if item)
    return str(text).strip() if text else None


def draw_overlay(image: np.ndarray, path: np.ndarray | None, reasoning: str | None,
                 *, text_size: int = 30, width_chars: int = 70) -> np.ndarray:
    from PIL import Image, ImageDraw

    result = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(result, "RGBA")
    if path is not None and len(path) >= 2:
        draw.line([tuple(point) for point in path], fill=(0, 230, 255, 255), width=8, joint="curve")
    if reasoning:
        font = _font(text_size)
        lines = textwrap.wrap(f"Reasoning: {reasoning}", width_chars)[:3]
        line_height = int(text_size * 1.3)
        top = result.height - line_height * len(lines) - 24
        draw.rectangle((0, top - 12, result.width, result.height), fill=(0, 0, 0, 175))
        for index, line in enumerate(lines):
            draw.text((20, top + index * line_height), line, fill=(255, 255, 255), font=font)
    return np.asarray(result)


# Commands --------------------------------------------------------------------------------------

def compare(arguments: argparse.Namespace) -> None:
    """Side-by-side clean and attacked camera frames with detector boxes over a speed strip."""
    clean = read_run(arguments.output, arguments.segment, arguments.clean, arguments.seed)
    attacked = read_run(arguments.output, arguments.segment, arguments.attacked, arguments.seed)
    count = min(len(clean.steps), len(attacked.steps))
    panel = MAX_VIDEO_WIDTH // 2
    curves = {arguments.clean_label: (times_of(clean)[:count], physical_speeds(clean)[:count]),
              arguments.attacked_label: (times_of(attacked)[:count], physical_speeds(attacked)[:count])}
    strip, mapping = speed_strip(curves, 2 * panel, arguments.strip_height)
    frames = []
    for index in range(count):
        panels = []
        for log, name in ((clean, arguments.clean_label), (attacked, arguments.attacked_label)):
            record = log.steps[index]
            image = frame_of(record, arguments.camera)
            scaled = resize_width(image, panel)
            boxes = detections_of(record, scale=panel / image.shape[1])
            panels.append(label(annotate(scaled, boxes, threshold=0.0), name, size=20))
        top = np.concatenate(panels, axis=1)
        frames.append(even_size(np.concatenate(
            [top, with_cursor(strip, mapping, float(curves[arguments.clean_label][0][index]))], axis=0)))
    out = Path(arguments.out)
    video = write_video(frames, out / f"{arguments.name}.mp4", arguments.fps)
    poster_index = count // 2 if arguments.poster_step is None else arguments.poster_step
    poster = save_still(frames[poster_index], out / f"{arguments.name}.webp")
    print(f"{video} ({video.stat().st_size} bytes, {count} decisions at {arguments.fps} fps)")
    print(f"{poster} (poster from decision {poster_index})")


def overlay(arguments: argparse.Namespace) -> None:
    """One run's camera frames with the planned trajectory and recorded reasoning text."""
    log = read_run(arguments.output, arguments.segment, arguments.treatment, arguments.seed)
    frames = []
    for record in log.steps:
        image = draw_overlay(frame_of(record, arguments.camera),
                             planned_path_pixels(record, arguments.camera), reasoning_of(record))
        frames.append(even_size(resize_width(label(image, arguments.label, size=34), arguments.width)))
    out = Path(arguments.out)
    video = write_video(frames, out / f"{arguments.name}.mp4", arguments.fps)
    poster_index = len(frames) // 2 if arguments.poster_step is None else arguments.poster_step
    poster = save_still(frames[poster_index], out / f"{arguments.name}.webp")
    print(f"{video} ({video.stat().st_size} bytes, {len(frames)} decisions at {arguments.fps} fps)")
    print(f"{poster} (poster from decision {poster_index})")


def frame(arguments: argparse.Namespace) -> None:
    """One recorded camera frame of a run."""
    log = read_run(arguments.output, arguments.segment, arguments.treatment, arguments.seed)
    record = next(step for step in log.steps if step.step == arguments.step)
    image = frame_of(record, arguments.camera)
    if arguments.crop is not None:
        left, top, right, bottom = arguments.crop
        image = image[top:bottom, left:right]
    path = save_still(resize_width(image, min(arguments.width, image.shape[1])),
                      Path(arguments.out) / f"{arguments.name}.webp")
    crop = "" if arguments.crop is None else f", crop {tuple(arguments.crop)}"
    print(f"{path} (decision {record.step}{crop})")


def stop_line(arguments: argparse.Namespace) -> None:
    """Speed against signed distance to the stop line for the clean and attacked runs of one segment."""
    from avsectester.evaluation import signed_stop_line_distances
    from avsectester.experiment import ExperimentResult, load_configuration

    configuration = load_configuration(arguments.configuration)
    result = ExperimentResult.load(Path(configuration.runs.output), configuration)
    runs = {run.treatment: run for run in result.runs
            if run.segment_index == arguments.segment and run.seed == arguments.seed}
    figure, axis = _figure(1200, 560, dpi=150)
    for treatment, name, color in ((arguments.clean, arguments.clean_label, "#1f77b4"),
                                   (arguments.attacked, arguments.attacked_label, "#d62728")):
        if treatment not in runs:
            raise FileNotFoundError(f"No saved run for segment {arguments.segment}, seed {arguments.seed}, "
                                    f"treatment {treatment!r}")
        run = runs[treatment]
        scenario_map = result.scenario_maps.get(run.segment.scenario_id)
        line = getattr(scenario_map, "stop_lines", {}).get(run.segment.user_data.get("stop_line_id"))
        if line is None:
            raise ValueError(f"No saved stop line for segment {arguments.segment}")
        states = [run.log.steps[0].physical_pre] + [step.physical_post for step in run.log.steps]
        distances, _ = signed_stop_line_distances(np.stack([state.ego.pose[:3, 3] for state in states]),
                                                  line.points)
        axis.plot(distances, [state.ego_speed for state in states], label=name, color=color, linewidth=2)
    axis.axvline(0.0, color="#333333", linestyle=":", linewidth=1.5, label="Stop line")
    axis.invert_xaxis()
    axis.set_xlabel("distance before the stop line (m)")
    axis.set_ylabel("speed (m/s)")
    axis.grid(alpha=0.3)
    axis.legend(loc="lower left")
    figure.tight_layout()
    path = save_still(_rgb(figure), Path(arguments.out) / f"{arguments.name}.png")
    print(f"{path} (segment {arguments.segment}, scene {run.segment.scenario_id})")


def still(arguments: argparse.Namespace) -> None:
    """Resize an existing image, such as a report screenshot or a saved plot."""
    from PIL import Image

    image = np.asarray(Image.open(arguments.image).convert("RGB"))
    suffix = ".png" if arguments.png else ".webp"
    path = save_still(image, Path(arguments.out) / f"{arguments.name}{suffix}", max_width=arguments.width)
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def run_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("output", type=Path, help="Experiment output directory (runs.output)")
        command.add_argument("--segment", type=int, required=True, help="Global segment-unit index")
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--camera", default=DEFAULT_CAMERA)
        command.add_argument("--name", required=True, help="Output file stem")
        command.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)

    command = commands.add_parser("compare", help=compare.__doc__)
    run_arguments(command)
    command.add_argument("--clean", default="clean")
    command.add_argument("--attacked", required=True)
    command.add_argument("--clean-label", default="clean")
    command.add_argument("--attacked-label", default="attacked")
    command.add_argument("--fps", type=float, default=10.0)
    command.add_argument("--strip-height", type=int, default=200)
    command.add_argument("--poster-step", type=int)
    command.set_defaults(handler=compare)

    command = commands.add_parser("overlay", help=overlay.__doc__)
    run_arguments(command)
    command.add_argument("--treatment", required=True)
    command.add_argument("--label", default="attacked")
    command.add_argument("--fps", type=float, default=10.0)
    command.add_argument("--width", type=int, default=960)
    command.add_argument("--poster-step", type=int)
    command.set_defaults(handler=overlay)

    command = commands.add_parser("frame", help=frame.__doc__)
    run_arguments(command)
    command.add_argument("--treatment", required=True)
    command.add_argument("--step", type=int, required=True)
    command.add_argument("--width", type=int, default=1280)
    command.add_argument("--crop", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                         help="Pixel region of the recorded image to keep")
    command.set_defaults(handler=frame)

    command = commands.add_parser("stop-line", help=stop_line.__doc__)
    command.add_argument("configuration", type=Path, help="Experiment configuration of the saved runs")
    command.add_argument("--segment", type=int, required=True, help="Global segment-unit index")
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--clean", default="clean")
    command.add_argument("--attacked", required=True)
    command.add_argument("--clean-label", default="clean")
    command.add_argument("--attacked-label", default="attacked")
    command.add_argument("--name", required=True, help="Output file stem")
    command.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    command.set_defaults(handler=stop_line)

    command = commands.add_parser("still", help=still.__doc__)
    command.add_argument("image", type=Path)
    command.add_argument("--name", required=True)
    command.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    command.add_argument("--width", type=int, default=MAX_IMAGE_WIDTH)
    command.add_argument("--png", action="store_true", help="Write PNG instead of WebP")
    command.set_defaults(handler=still)

    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
