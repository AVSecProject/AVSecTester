"""Render a short road-level CARLA insertion demo on a dedicated server.

Vehicles follow prescribed poses to make attachment and visibility easy to inspect.
This is a geometry demo, not an attack-success or driving-policy evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from avsectester.insertion import (
    AttachedPlacement,
    Insertion,
    Orientation,
    PlaneAsset,
    WorldPlacement,
    resolve_insertion,
)
from avsectester.plane import Observation
from avsectester.rendering.cameras import PinholeCamera
from avsectester.rendering.visibility import DepthVisibilityEstimator
from avsectester.scenarios.carla_gt import carla_actor_pose
from avsectester.scenarios.carla_provider import carla_filter_context
from avsectester.simulators.carla import insertion_perturbation
from validate_carla_visibility import FOCAL, HEIGHT, WIDTH, RoadValidation


def textures():
    patch = Image.new("RGBA", (256, 128), (255, 255, 255, 255))
    draw = ImageDraw.Draw(patch)
    for row in range(4):
        for column in range(8):
            if (row + column) % 2 == 0:
                draw.rectangle((column * 32, row * 32, column * 32 + 31, row * 32 + 31), fill="red")
    sign = Image.new("RGBA", (160, 360))
    draw = ImageDraw.Draw(sign)
    draw.rectangle((75, 145, 85, 359), fill=(130, 130, 130, 255))
    vertices = [(47, 2), (112, 2), (157, 47), (157, 112), (112, 157), (47, 157), (2, 112), (2, 47)]
    draw.polygon(vertices, fill="white")
    draw.polygon([(80 + (x - 80) * .92, 80 + (y - 80) * .92) for x, y in vertices], fill="red")
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 38)
    except OSError:
        font = ImageFont.load_default()
    draw.text((80, 80), "STOP", fill="white", font=font, anchor="mm")
    return np.asarray(patch), np.asarray(sign)


def main():
    import carla

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=2300)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("tmp/carla-insertions"))
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(60)
    world = client.get_world()
    previous = world.get_settings()
    demo = RoadValidation(world, args.output)
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        world.set_weather(carla.WeatherParameters.ClearNoon)
        ego = demo.spawn("vehicle.tesla.model3", world.get_map().get_spawn_points()[0])
        origin = demo.settle(ego)

        def relative(x, y=0, z=0, yaw=0):
            point = origin.transform(carla.Location(x, y, z))
            return carla.Transform(point, carla.Rotation(yaw=origin.rotation.yaw + yaw))

        host = demo.spawn("vehicle.tesla.model3", relative(13, z=.6))
        host_initial = demo.settle(host)
        blocker = demo.spawn("vehicle.carlamotors.carlacola", relative(8, -3.3, .6))
        blocker_initial = demo.settle(blocker)
        demo.cameras(relative(1.5, z=1.6))
        rgb_actor = next(actor for name, actor, _ in demo.sensors if name == "rgb")
        k = np.array([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1]])
        camera = PinholeCamera(k, WIDTH, HEIGHT)
        rgb_sensor = SimpleNamespace(
            object=rgb_actor, P=np.column_stack((k, np.zeros(3))), imsize=(HEIGHT, WIDTH),
            source_identifier="front", frame0=0,
        )
        backend = SimpleNamespace(
            client=SimpleNamespace(world=world), lead=host, npcs=[], scenario={}, replay_scenario={},
            ego=SimpleNamespace(actor=ego, sensors={"front": rgb_sensor},
                                get_object_state=lambda: SimpleNamespace(velocity=SimpleNamespace(norm=lambda: 0.0))),
        )
        patch, sign = textures()
        heading = -origin.rotation.yaw
        location = relative(9.5, 3.0, 1.025).location
        fixed = Insertion("fixed-sign", PlaneAsset(sign, .9, 2.025),
                          WorldPlacement((location.x, -location.y, location.z)),
                          Orientation("fixed_world", (0, 0, heading + 180)))
        modes = ("follow_host", "fixed_world", "face_victim")
        patches = [Insertion(
            "patch", PlaneAsset(patch, 1.2, .6),
            AttachedPlacement("lead", "rear_center", (-.08, 0, .1)),
            Orientation(mode, (0, 0, 180 if mode == "follow_host" else heading + 180))
            if mode != "face_victim" else Orientation(mode),
        ) for mode in modes]
        perturbations = [insertion_perturbation(backend, (patch, fixed)) for patch in patches]
        estimator = DepthVisibilityEstimator(tolerance_m=.0002)
        rows, animation = [], []
        optical_from_unreal = np.eye(4)
        optical_from_unreal[:3, :3] = [[0, 1, 0], [0, 0, -1], [1, 0, 0]]
        flip_y = np.diag([1, -1, 1, 1])
        for index in range(args.frames):
            progress = index / max(args.frames - 1, 1)
            ego_pose = relative(3.0 * progress)
            ego_pose.location.z = origin.location.z
            ego.set_transform(ego_pose)
            host_pose = relative(13 + 2.5 * progress, .9 * np.sin(progress * np.pi), yaw=18 * np.sin(progress * np.pi))
            host_pose.location.z = host_initial.location.z
            host.set_transform(host_pose)
            # A parked van increasingly obscures the host as the viewpoint changes.
            blocker_pose = relative(8, -2.0 + .9 * progress)
            blocker_pose.location.z = blocker_initial.location.z
            blocker.set_transform(blocker_pose)
            camera_pose = relative(1.5 + 3.0 * progress, z=1.6)
            frame, captures = demo.capture(camera_pose)
            clean = captures["rgb"]
            backend.selection_context = lambda: carla_filter_context(backend, depth=captures["depth"])
            observation = Observation(world.get_snapshot().timestamp.elapsed_seconds, frame,
                                      sensor_data={"front": clean})
            Image.fromarray(clean).save(args.output / f"clean-{index:03d}.png")
            cam_from_world = optical_from_unreal @ np.asarray(camera_pose.get_inverse_matrix()) @ flip_y
            actors = {"lead": carla_actor_pose(host)}
            victim = carla_actor_pose(ego)
            if index == 0:
                from avsectester.scenarios import RoleSpec, ScenarioRequirement
                from avsectester.scenarios.filters import InView, MinVisibility

                context = carla_filter_context(backend, depth=captures["depth"])
                requirement = ScenarioRequirement(
                    "attached-patch", roles={"attacker": RoleSpec(ids=("lead",))},
                    insertions=(patches[0],), constraints=[InView("front"), MinVisibility(.1, camera="front")],
                )
                selected = requirement.evaluate(context)
                print(f"Initial scene selection: {selected.status}", flush=True)
                if selected.status != "pass":
                    raise RuntimeError(f"Demo initial scene rejected: {selected.reason}")
            panel = Image.new("RGB", (WIDTH * 3, HEIGHT + 150), "white")
            draw = ImageDraw.Draw(panel)
            record = {"index": index, "carla_frame": frame, "modes": {}}
            overlay = clean.copy()
            for column, (mode, insertion) in enumerate(zip(modes, patches)):
                resolved = [resolve_insertion(spec, actors, victim) for spec in (insertion, fixed)]
                # Save the actual observation perturbation output, not a view-only overlay.
                rendered = perturbations[column](observation).sensor_data["front"]
                evidence_by_id = {}
                for item in resolved:
                    evidence = estimator.estimate(
                        item, camera, cam_from_world, scene_depth=captures["depth"],
                        depth_convention="z", camera_name="front",
                        other_insertions=tuple(other for other in resolved if other.id != item.id),
                    )
                    evidence_by_id[item.id] = {
                        "visibility": evidence.visibility.fraction,
                        "reference_pixels": evidence.reference_pixels,
                        "visible_pixels": int(evidence.visible_mask.sum()),
                        "world_pose": item.pose.tolist(),
                    }
                    if mode == "follow_host":
                        mask = Image.fromarray(evidence.visible_mask.astype(np.uint8) * 255)
                        mask.save(args.output / f"{item.id}-visible-{index:03d}.png")
                        reference = evidence.reference_mask
                        visible = evidence.visible_mask
                        overlay[reference & ~visible] = (
                            .35 * overlay[reference & ~visible] + .65 * np.array([255, 30, 30])
                        ).astype(np.uint8)
                        overlay[visible] = (
                            .35 * overlay[visible] + .65 * np.array([20, 240, 80])
                        ).astype(np.uint8)
                Image.fromarray(rendered).save(args.output / f"{mode}-{index:03d}.png")
                panel.paste(Image.fromarray(rendered), (column * WIDTH, 30))
                values = evidence_by_id
                draw.text((column * WIDTH + 10, 8), mode, fill="black")
                draw.text((column * WIDTH + 10, HEIGHT + 42),
                          f"frame {index}   patch {values['patch']['visibility']:.3f}   fixed sign {values['fixed-sign']['visibility']:.3f}", fill="black")
                record["modes"][mode] = values
            draw.text((10, HEIGHT + 78), "CARLA road scene | depth visibility | prescribed motion | fixed sign has constant world position", fill="black")
            draw.text((10, HEIGHT + 100), "Visibility = visible opaque pixels / complete projected opaque pixels (outside image counts as invisible)", fill="black")
            rows.append(record)
            Image.fromarray(overlay).save(args.output / f"visibility-{index:03d}.png")
            panel.save(args.output / f"comparison-{index:03d}.jpg")
            animation.append(panel.resize((1280, 420)))
        animation[0].save(args.output / "demo.gif", save_all=True, append_images=animation[1:], duration=150, loop=0)
        contact = Image.new("RGB", (1280, 420 * 3), "white")
        for row, index in enumerate((0, len(animation) // 2, len(animation) - 1)):
            contact.paste(animation[index], (0, row * 420))
        contact.save(args.output / "contact-sheet.jpg")
        visibility_contact = Image.new("RGB", (WIDTH * 3, (HEIGHT + 32) * 3), "white")
        draw = ImageDraw.Draw(visibility_contact)
        for row, index in enumerate((0, len(animation) // 2, len(animation) - 1)):
            for column, (prefix, title) in enumerate((
                ("clean", "Rendered scene"), ("follow_host", "Model input"),
                ("visibility", "Green: visible / red: occluded"),
            )):
                y = row * (HEIGHT + 32)
                draw.text((column * WIDTH + 10, y + 8), f"Frame {index}: {title}", fill="black")
                with Image.open(args.output / f"{prefix}-{index:03d}.png") as image:
                    visibility_contact.paste(image, (column * WIDTH, y + 32))
        visibility_contact.save(args.output / "visibility-contact.jpg")
        (args.output / "measurements.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(f"Saved {args.frames} frames to {args.output}", flush=True)
    finally:
        demo.close()
        world.apply_settings(previous)


if __name__ == "__main__":
    main()
