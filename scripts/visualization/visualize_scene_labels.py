"""Project FrameData actor boxes onto recorded NuRec or nuScenes camera images."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from avsectester.closed_loop.simulator import SensorView
from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator
from avsectester.datasets.nurec import NuRecDataset
from avsectester.datasets.nuscenes import NuScenesDataset
from avsectester.evaluation.visualization import draw_boxes3d, filmstrip, save_image


def recorded_video_frame(path: str | Path, timestamps, timestamp: float):
    """Choose the nearest recorded sample by time, independently of decision positions."""
    import cv2

    times = np.asarray(timestamps, dtype=float)
    if not len(times) or timestamp < times[0] or timestamp > times[-1]:
        raise ValueError("Requested timestamp lies outside the recorded camera support")
    index = int(np.argmin(np.abs(times - timestamp)))
    video = cv2.VideoCapture(str(path))
    try:
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = video.read()
    finally:
        video.release()
    if not ok:
        raise ValueError(f"Cannot read video frame {index} from {path}")
    return bgr[:, :, ::-1].copy()


def nurec_examples(usdz: Path, video: Path, output: Path, *, camera_id: str, samples: int) -> None:
    dataset = NuRecDataset([str(usdz)], camera_id=camera_id, camera_alias=camera_id, time_step=.1)
    scenario = next(dataset.scenarios())
    timeline = dataset.frames(scenario.id)
    timestamps = dataset.camera_timestamps(str(usdz))
    camera, ego_from_sensor = dataset.camera_geometry(str(usdz))
    positions = np.unique(np.linspace(0, len(timeline) - 1, samples).astype(int))
    images = []
    for position in positions:
        frame = timeline[int(position)]
        data = dataset.frame_data(scenario.id, frame)
        rgb = recorded_video_frame(video, timestamps, frame.timestamp)
        if data.ego is None:
            raise ValueError("NuRec frame has no true ego state")
        ego = data.ego.box_pose()
        actors = {identity: actor.box_pose() for identity, actor in data.actors.items()}
        view = SensorView(camera_id, rgb, camera, np.linalg.inv(data.ego.pose @ ego_from_sensor),
                          actors, ego, frame.timestamp, frame.position)
        images.append(draw_boxes3d(rgb, view, actors=data.actors, max_distance=40.))
    save_image(filmstrip(images, cols=4), output / "labels_nurec.png")


def nuscenes_examples(dataroot: Path, output: Path, *, version: str, samples: int) -> None:
    dataset = NuScenesDataset(str(dataroot), version=version)
    scenario = next(dataset.scenarios())
    timeline = dataset.frames(scenario.id)
    positions = np.unique(np.linspace(0, len(timeline) - 1, samples).astype(int))
    images = []
    simulator = RecordedReplaySimulator()
    with simulator.create_runtime() as simulator_runtime:
        for position in positions:
            frame = timeline[int(position)]
            data = dataset.frame_data(scenario.id, frame)
            simulator_runtime.reset(dataset.scenario_start(scenario.id, frame))
            view = simulator_runtime.sensor_view("front")
            images.append(draw_boxes3d(view.data, view, actors=data.actors, max_distance=40.))
    save_image(filmstrip(images, cols=4), output / "labels_nuscenes.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usdz", type=Path, help="NuRec reconstruction archive")
    parser.add_argument("--video", type=Path, help="Matching recorded NuRec camera video")
    parser.add_argument("--camera", default="camera_front_wide_120fov")
    parser.add_argument("--nuscenes", type=Path, help="nuScenes dataroot")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("tmp/compare"))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    if args.usdz is None and args.nuscenes is None:
        parser.error("Provide --usdz with --video, or --nuscenes")
    if args.usdz is not None:
        if args.video is None:
            parser.error("--usdz requires --video from the same scene and camera")
        nurec_examples(args.usdz, args.video, args.output, camera_id=args.camera, samples=args.samples)
    if args.nuscenes is not None:
        nuscenes_examples(args.nuscenes, args.output, version=args.version, samples=args.samples)


if __name__ == "__main__":
    main()
