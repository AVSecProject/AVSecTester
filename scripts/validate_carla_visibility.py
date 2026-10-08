"""Validate visibility on a road using a dedicated CARLA server.

Vehicles settle under physics before their poses are frozen for reference measurements.
The reference camera has a larger canvas but the SAME focal length, pose and pixel scale.
The actual camera provides scene depth. This changes only the dedicated validation world.
"""

import argparse
import json
import math
from pathlib import Path
from queue import Queue

import numpy as np

from avsectester.scenarios.visibility import visibility_from_depth


WIDTH, HEIGHT = 640, 480
SCALE = 3
FOCAL = WIDTH / (2 * math.tan(math.radians(60) / 2))
BOUNDS = (WIDTH, HEIGHT, 2 * WIDTH, 2 * HEIGHT)
TOLERANCE = 0.0002  # A few units of CARLA's 24-bit, 1000 m depth encoding.


def depth_metres(bgra):
    depth = bgra.astype(np.float64)
    return (depth[..., 2] + 256 * depth[..., 1] + 65536 * depth[..., 0]) * (1000 / 16777215)


class RoadValidation:
    def __init__(self, world, output):
        self.world = world
        self.output = output
        self.actors = []
        self.sensors = []
        self.rows = []
        self.patch_tag = None

    def spawn(self, blueprint, transform):
        actor = self.world.spawn_actor(
            self.world.get_blueprint_library().find(blueprint), transform
        )
        self.actors.append(actor)
        return actor

    def settle(self, actor):
        import carla

        actor.apply_control(carla.VehicleControl(hand_brake=True))
        for _ in range(60):
            self.world.tick()
        if actor.get_velocity().length() > 0.02:
            raise RuntimeError("Vehicle did not settle before the static visibility measurement")
        actor.set_simulate_physics(False)
        return actor.get_transform()

    def cameras(self, transform):
        for name, kind, scale in [
            ("rgb", "rgb", 1),
            ("depth", "depth", 1),
            ("semantic", "semantic_segmentation", 1),
            ("reference", "depth", SCALE),
        ]:
            bp = self.world.get_blueprint_library().find("sensor.camera." + kind)
            values = {
                "image_size_x": str(WIDTH * scale),
                "image_size_y": str(HEIGHT * scale),
                "fov": str(math.degrees(2 * math.atan(WIDTH * scale / (2 * FOCAL)))),
                "sensor_tick": "0",
                "lens_k": "0",
                "lens_kcube": "0",
            }
            for key, value in values.items():
                bp.set_attribute(key, value)
            actor = self.world.spawn_actor(bp, transform)
            self.actors.append(actor)
            queue = Queue()
            actor.listen(queue.put)
            self.sensors.append((name, actor, queue))

    def capture(self, transform):
        for _, actor, _ in self.sensors:
            actor.set_transform(transform)
        # Sensor rendering is asynchronous. This is local to the validation script.
        for _ in range(3):
            frame = self.world.tick()
            data = {}
            for name, _, queue in self.sensors:
                sample = queue.get(timeout=30)
                while sample.frame < frame:
                    sample = queue.get(timeout=30)
                if sample.frame != frame:
                    raise RuntimeError("Camera frames do not agree")
                bgra = (
                    np.frombuffer(sample.raw_data, np.uint8)
                    .reshape(sample.height, sample.width, 4)
                    .copy()
                )
                if name == "rgb":
                    data[name] = bgra[..., :3][..., ::-1]
                elif name == "semantic":
                    data[name] = bgra[..., 2]
                else:
                    data[name] = depth_metres(bgra)
        return frame, data

    def reference(self, background, present, name, actor, camera_pose):
        from PIL import Image

        # Background and target-present renders differ only by this target's geometry.
        # Use the actor box only to locate its depth samples. Pixel membership still comes
        # from the rendered surface, never from filling that box as a silhouette.
        vertices = np.array(
            [
                [v.x, v.y, v.z, 1]
                for v in actor.bounding_box.get_world_vertices(actor.get_transform())
            ]
        )
        local = vertices @ np.asarray(camera_pose.get_inverse_matrix()).T
        if (local[:, 0] <= 0).any():
            raise RuntimeError("Reference target crosses the camera plane")
        u = FOCAL * local[:, 1] / local[:, 0] + WIDTH * SCALE / 2
        v = -FOCAL * local[:, 2] / local[:, 0] + HEIGHT * SCALE / 2
        left, right = int(np.floor(u.min())) - 4, int(np.ceil(u.max())) + 4
        top, bottom = int(np.floor(v.min())) - 4, int(np.ceil(v.max())) + 4
        if left < 0 or top < 0 or right >= present.shape[1] or bottom >= present.shape[0]:
            raise RuntimeError(f"{name}: enlarge the reference canvas")
        region = np.zeros(present.shape, bool)
        region[top:bottom, left:right] = True
        mask = (background - present > TOLERANCE) & region
        if (
            not mask.any()
            or mask[0].any()
            or mask[-1].any()
            or mask[:, 0].any()
            or mask[:, -1].any()
        ):
            raise RuntimeError(f"{name}: missing or truncated reference, enlarge the canvas")
        Image.fromarray(mask.astype(np.uint8) * 255).save(self.output / f"{name}-reference.png")
        return mask, present.copy()

    def measure(self, name, reference, frame, data):
        from PIL import Image

        mask, target_depth = reference
        left, top, right, bottom = BOUNDS
        scene_depth = np.full(mask.shape, np.nan)
        scene_depth[top:bottom, left:right] = data["depth"]
        result = visibility_from_depth(
            mask,
            target_depth,
            scene_depth,
            camera="front",
            image_bounds=BOUNDS,
            tolerance_m=TOLERANCE,
        )
        visible = mask & (target_depth <= scene_depth + TOLERANCE)
        row = {
            "case": name,
            "frame": frame,
            "fraction": result.fraction,
            "reference_pixels": int(mask.sum()),
            "in_image_pixels": int(mask[top:bottom, left:right].sum()),
            "visible_pixels": int(visible.sum()),
        }
        if name == "patch-front-isolated":
            tags, counts = np.unique(
                data["semantic"][mask[top:bottom, left:right]], return_counts=True
            )
            self.patch_tag = int(tags[np.argmax(counts)])
        if name.startswith("patch") and self.patch_tag is not None:
            # Independent check for these scenes: the patch and vehicle occluders have
            # different semantic classes. This does not supply the reference denominator.
            semantic_visible = (data["semantic"] == self.patch_tag) & mask[top:bottom, left:right]
            disagreements = np.count_nonzero(semantic_visible != visible[top:bottom, left:right])
            row["semantic_visible_pixels"] = int(semantic_visible.sum())
            row["semantic_disagreement_pixels"] = int(disagreements)
            assert disagreements <= max(1, int(mask.sum() * 0.001)), (
                f"{name}: depth and independent semantic masks disagree beyond 0.1% of the reference"
            )
        self.rows.append(row)
        Image.fromarray(data["rgb"]).save(self.output / f"{name}-rgb.png")
        Image.fromarray(visible[top:bottom, left:right].astype(np.uint8) * 255).save(
            self.output / f"{name}-visible.png"
        )
        print(json.dumps(row), flush=True)
        return result.fraction

    def close(self):
        for _, actor, _ in self.sensors:
            actor.stop()
        for actor in reversed(self.actors):
            if actor.is_alive:
                actor.destroy()


def main():
    import carla
    from avsectester.attacks.patch.physical_patch import PhysicalPatch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=2300)
    parser.add_argument("--output", type=Path, default=Path("tmp/visibility-road"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(60)
    world = client.get_world()
    previous = world.get_settings()
    validation = RoadValidation(world, args.output)
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        spawn = world.get_map().get_spawn_points()[0]
        ego = validation.spawn("vehicle.tesla.model3", spawn)
        ego_tf = validation.settle(ego)

        def relative(x, y=0, z=0, yaw=0):
            position = ego_tf.transform(carla.Location(x, y, z))
            return carla.Transform(position, carla.Rotation(yaw=ego_tf.rotation.yaw + yaw))

        # Road-level camera on the ego, at normal roof height.
        poses = {
            "front": relative(1.5, z=1.6),
            "clipped": relative(1.5, z=1.6, yaw=30),
            "outside": relative(1.5, z=1.6, yaw=45),
        }
        validation.cameras(poses["front"])
        backgrounds = {key: validation.capture(pose)[1]["reference"] for key, pose in poses.items()}
        target = validation.spawn("vehicle.tesla.model3", relative(16, z=0.6))
        target_tf = validation.settle(target)
        road = world.get_map().get_waypoint(target_tf.location)
        if abs(target_tf.location.z - road.transform.location.z) > 0.3:
            raise RuntimeError("Target is not settled on the road")
        references = {}
        for key, pose in poses.items():
            frame, data = validation.capture(pose)
            references[key] = validation.reference(
                backgrounds[key], data["reference"], f"vehicle-{key}", target, pose
            )
            fraction = validation.measure(f"vehicle-{key}-clear", references[key], frame, data)
            if key == "front":
                assert fraction > 0.98
            elif key == "clipped":
                assert 0.05 < fraction < 0.95
            else:
                assert fraction == 0

        # Measure the unpatched vehicle so the patch itself is not an extra occluder.
        blocker = validation.spawn("vehicle.carlamotors.carlacola", relative(8, 3.5, 0.6))
        blocker_tf = validation.settle(blocker)

        def occlusion_cases(kind, refs):
            for name, lateral in [("partial", 1.2), ("blocked", 0)]:
                pose = relative(8, lateral)
                pose.location.z = blocker_tf.location.z
                blocker.set_transform(pose)
                for view in ("front", "clipped"):
                    frame, data = validation.capture(poses[view])
                    fraction = validation.measure(f"{kind}-{view}-{name}", refs[view], frame, data)
                    if name == "partial" and view == "front":
                        assert 0.05 < fraction < 0.95
                    elif name == "blocked":
                        assert fraction < 0.02
            blocker.set_transform(blocker_tf)

        occlusion_cases("vehicle", references)

        # Exercise the repository's actual patch mesh and default placement. The host car
        # can occlude this surface too, so it must be absent from the reference render.
        patch = PhysicalPatch()
        patch_actors = patch.apply(world, target)
        validation.actors.extend(patch_actors)
        patch_tf = patch_actors[0].get_transform()
        target.set_transform(relative(-30))  # Park host and attached panel behind the camera.
        isolated_patch = validation.spawn(patch.prop, patch_tf)
        patch_refs = {}
        for key, pose in poses.items():
            frame, data = validation.capture(pose)
            patch_refs[key] = validation.reference(
                backgrounds[key], data["reference"], f"patch-{key}", isolated_patch, pose
            )
            fraction = validation.measure(f"patch-{key}-isolated", patch_refs[key], frame, data)
            if key == "front":
                assert fraction > 0.98
            elif key == "clipped":
                assert 0.05 < fraction < 0.95
            else:
                assert fraction == 0
        isolated_patch.destroy()
        target.set_transform(target_tf)
        for key, pose in poses.items():
            frame, data = validation.capture(pose)
            fraction = validation.measure(f"patch-{key}-mounted", patch_refs[key], frame, data)
            # Restoring the host cannot reveal pixels absent from the isolated reference.
            assert (
                fraction
                <= next(
                    r["fraction"] for r in validation.rows if r["case"] == f"patch-{key}-isolated"
                )
                + 0.01
            )
        occlusion_cases("patch", patch_refs)

        report = {
            "server": client.get_server_version(),
            "client": client.get_client_version(),
            "map": world.get_map().name,
            "camera_size": [WIDTH, HEIGHT],
            "reference_size": [WIDTH * SCALE, HEIGHT * SCALE],
            "image_bounds": BOUNDS,
            "focal_length_pixels": FOCAL,
            "depth_tolerance_m": TOLERANCE,
            "target_height_above_road_m": target_tf.location.z - road.transform.location.z,
            "method": "Unoccluded full silhouette reference, scene depth inside the actual camera only",
            "scope": "Static road scenes. References captured before adding occluders. No full-run monitoring.",
            "results": validation.rows,
        }
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        validation.close()
        world.apply_settings(previous)


if __name__ == "__main__":
    main()
