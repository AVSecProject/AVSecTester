#!/usr/bin/env python
"""Inspect host-bound insertions and a fixed-world sign on a recorded NuRec camera sequence.

Requires a NuRec renderer and the matching USDZ. Frames are rendered at recorded camera and actor
poses, without a driving policy or interaction. The sensor perturbation seam supplies the saved
model-input images. Visibility overlays use known cuboids, not exact rendered occlusion.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from avsectester.insertion import (
    AttachedPlacement,
    Insertion,
    Orientation,
    PlaneAsset,
    PlaneSurface,
    WorldPlacement,
)
from avsectester.plane import Observation
from avsectester.scenarios.datasets.nurec import NuRecDataset
from avsectester.scenarios.estimators import CuboidVisibilityEstimator
from avsectester.simulators.nurec import NuRecInsertions
from avsectester.simulators.patch_insertion import (
    Harmonizer,
    PatchCompositor,
    frame_perturbation,
)


class UnchangedColors(Harmonizer):
    """Preserve diagnostic texture colors so motion and masks remain easy to inspect."""

    def __call__(self, composite_rgb, mask, background_rgb):
        return composite_rgb


class MountedSign:
    """A sign panel and narrow supporting pole, in canonical local asset coordinates."""

    def planes(self):
        from avsectester.attacks.object_insertion.sign_spoof import sign_rgba

        face = PlaneAsset(sign_rgba(), 0.9, 0.9).planes()[0]
        pole = PlaneAsset(np.full((4, 4, 4), [140, 140, 140, 255], np.uint8), 0.06, 1.4).planes()[0]
        return (PlaneSurface(pole.corners + [-0.03, 0, -1.1], pole.texture), face)


def diagnostic_texture(color, index):
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (128, 96), (*color, 255))
    draw = ImageDraw.Draw(image)
    for y in range(0, 96, 24):
        for x in range(0, 128, 32):
            if (x // 32 + y // 24) % 2:
                draw.rectangle((x, y, x + 31, y + 23), fill=(20, 20, 20, 255))
    draw.rectangle((45, 28, 85, 66), fill=(255, 255, 255, 255))
    draw.text((59, 40), str(index), fill=(0, 0, 0, 255), stroke_width=1)
    return np.asarray(image)


def build_insertions(context, host, patch_only=False, texture=None, world_position=None):
    from scipy.spatial.transform import Rotation

    pose = context.actors[host].transform
    initial_rear = pose[:3, :3] @ np.diag([-1, -1, 1])
    roll, pitch, yaw = Rotation.from_matrix(initial_rear).as_euler("xyz", degrees=True)
    modes = ("follow_host", "fixed_world", "face_victim")
    colors = ((255, 200, 30), (20, 220, 240), (240, 60, 170))
    insertions = []
    for i, (mode, color, lateral) in enumerate(zip(modes, colors, (0.58, 0, -0.58))):
        angles = (0, 0, 180) if mode == "follow_host" else (roll, pitch, yaw)
        orientation = Orientation(mode) if mode == "face_victim" else Orientation(mode, angles)
        insertions.append(
            Insertion(
                mode,
                PlaneAsset(
                    texture if texture is not None else diagnostic_texture(color, i + 1), 0.48, 0.34
                ),
                AttachedPlacement(host, "rear_center", (-0.10, lateral, 0.15)),
                orientation,
            )
        )
    if not patch_only:
        if world_position is None:
            position = (context.world_from_ego @ [27, -3.4, 1.8, 1])[:3]
        else:
            position = np.asarray(world_position)
        ego_yaw = math.degrees(
            math.atan2(context.world_from_ego[1, 0], context.world_from_ego[0, 0])
        )
        insertions.append(
            Insertion(
                "world_sign",
                MountedSign(),
                WorldPlacement(tuple(position)),
                Orientation("fixed_world", (0, 0, ego_yaw + 180)),
            )
        )
    return tuple(insertions)


def review_frame(clean, attacked, evidence, frame_index, host_box):
    import cv2

    overlay = clean.copy()
    for item in evidence.values():
        overlay[item.reference_mask] = (240, 70, 60)
        overlay[item.visible_mask] = (20, 240, 100)
    x1, y1, x2, y2 = host_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    width = max(200, (x2 - x1) * 1.8)
    x1, x2 = max(0, int(cx - width)), min(clean.shape[1], int(cx + width))
    y1, y2 = max(0, int(cy - width * 0.45)), min(clean.shape[0], int(cy + width * 0.45))
    panels = [clean, attacked, attacked[y1:y2, x1:x2], overlay[y1:y2, x1:x2]]
    panels = [cv2.resize(p, (960, 540)) for p in panels]
    captions = [
        "Recorded NuRec RGB",
        "Model input: 3 attached patches + world-fixed sign",
        "Host detail: 1 follow / 2 fixed-world / 3 face-victim",
        "Cuboid visibility estimate: green visible / red hidden",
    ]
    for panel, caption in zip(panels, captions):
        cv2.rectangle(panel, (0, 0), (959, 40), (15, 15, 15), -1)
        cv2.putText(panel, caption, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    result = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])
    text = f"Frame {frame_index}: " + " | ".join(
        f"{key}: {value.visibility.fraction:.1%}" if value.visibility else f"{key}: unknown"
        for key, value in evidence.items()
    )
    cv2.rectangle(result, (0, 1040), (1919, 1079), (15, 15, 15), -1)
    cv2.putText(result, text, (12, 1067), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return result


def main(patch_only=False):
    import cv2
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usdz", required=True)
    parser.add_argument("--endpoint", default="127.0.0.1:50051")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--stride", type=int, default=3, help="Recorded camera-frame stride")
    parser.add_argument("--keyframe", type=float, default=0.5)
    parser.add_argument("--host", default="15", help="Explicit USDZ track ID, never reselected")
    parser.add_argument("--texture", default=None)
    parser.add_argument(
        "--world-position", type=float, nargs=3, default=None, metavar=("X", "Y", "Z")
    )
    parser.add_argument("--out", type=Path, default=Path("tmp/nurec-insertion"))
    args = parser.parse_args()
    if args.frames < 1 or args.stride < 1:
        parser.error("frames and stride must be positive")
    dataset = NuRecDataset([args.usdz], keyframe=args.keyframe, endpoint=args.endpoint)
    start = next(dataset.scenes())
    with dataset.context(start) as initial:
        if args.host not in initial.actors:
            parser.error(f"Track {args.host!r} is not present at the chosen initial frame")
        items = build_insertions(initial, args.host, patch_only, args.texture, args.world_position)
    window = dataset.initial_sequence(start, args.frames * args.stride)[:: args.stride]
    if len(window) != args.frames:
        parser.error("The recorded clip has fewer frames than requested from this start")
    renderer = dataset.make_renderer(start)
    fps = 1 / (window[1].t - window[0].t) if len(window) > 1 else 10.0
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "model_inputs").mkdir(exist_ok=True)
    observations = {}
    animation, snapshots, rows = [], [], []
    try:
        renderer.load_scene(start.source["scene_id"])
        camera = renderer.camera_model(dataset.sensor)

        def geometry(obs):
            ctx = observations[obs.frame]
            return (
                ctx.actors,
                ctx.victim,
                renderer.camera_transform(ctx.world_from_ego, dataset.sensor),
            )

        adapter = NuRecInsertions(
            items,
            camera,
            geometry,
            PatchCompositor(UnchangedColors()),
            visibility_estimator=CuboidVisibilityEstimator(),
            camera_name=dataset.camera,
        )
        perturb = frame_perturbation(adapter, camera=dataset.camera)
        for index, scene in enumerate(window):
            with dataset.context(scene) as context:
                observations[index] = context
                clean = renderer.render_at(
                    scene.ego.pose, scene.source["timestamp_us"], dataset.sensor
                )
                obs = Observation(scene.t, index, sensor_data={dataset.camera: clean})
                model_input = perturb(obs)
                attacked = model_input.sensor_data[dataset.camera]
                Image.fromarray(attacked).save(args.out / "model_inputs" / f"{index:04d}.png")
                host = next(o for o in scene.objects if o.track_id == args.host)
                box = host.box2d.get(dataset.camera, (640, 550, 1280, 900))
                review = review_frame(clean, attacked, adapter.evidence, index, box)
                animation.append(
                    Image.fromarray(review).resize((1280, 720), Image.Resampling.LANCZOS)
                )
                if index in {0, len(window) // 2, len(window) - 1}:
                    Image.fromarray(review).save(args.out / f"review_{index:04d}.png")
                    snapshots.append(cv2.resize(review, (960, 540)))
                rows.append(
                    {
                        "frame": index,
                        "recorded_frame": scene.frame,
                        "timestamp_us": scene.source["timestamp_us"],
                        "host_id": args.host,
                        "host_pose": context.actors[args.host].transform.tolist(),
                        "insertions": {
                            item.id: {
                                "pose": item.pose.tolist(),
                                "visibility": adapter.evidence[item.id].visibility.fraction,
                                "visibility_source": adapter.evidence[item.id].visibility.source,
                            }
                            for item in adapter.resolved
                        },
                    }
                )
                observations.pop(index)
            print(f"Rendered frame {index + 1}/{len(window)}", flush=True)
    finally:
        renderer.close()
    animation[0].save(
        args.out / "review.gif",
        save_all=True,
        append_images=animation[1:],
        duration=max(10, round(100 / fps) * 10),
        loop=0,
    )
    Image.fromarray(np.vstack(snapshots)).save(args.out / "contact_sheet.png")
    (args.out / "measurements.json").write_text(
        json.dumps(
            {
                "protocol": "recorded_camera_and_actor_poses",
                "scene": start.source,
                "visibility_method": "cuboid_estimate",
                "frames": rows,
            },
            indent=2,
        )
    )
    print(args.out / "review.gif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
