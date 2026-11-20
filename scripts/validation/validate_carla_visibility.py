"""Measure full-silhouette visibility using simulator-owned calibrated RGB/depth captures.

Use a dedicated CARLA server. The recipe supplies an enlarged camera with the original focal
length and pixel scale. The central crop represents the normal field of view. Background and
target-present depth provide an independent silhouette reference within the target's projected box
and optical-axis depth range. CarlaSimulatorRuntime.sensor_view supplies all RGB and depth captures.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path

import cv2
import numpy as np

from avsectester.attack.insertion.visibility import visibility_from_depth
from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.closed_loop.simulators.carla.conversions import rotation_to_carla
from avsectester.data_types import ActorState, Control
from avsectester.datasets.carla import CarlaDataset, PrescribedMotion
from avsectester.experiment import load_configuration
from avsectester.geometry.poses import mirror_y_points, rotation_matrix, transform
from avsectester.rendering.textures import CheckerboardTexture
from scripts import REPO_ROOT

WIDTH, HEIGHT, SCALE = 640, 480, 3
FOCAL = WIDTH / (2 * math.tan(math.radians(60) / 2))
BOUNDS = (WIDTH, HEIGHT, 2 * WIDTH, 2 * HEIGHT)
TOLERANCE = 0.0002
BOX_MARGIN_PIXELS = 2
BOX_DEPTH_MARGIN_M = 0.02


def validation_configuration(path: Path, port: int, traffic_manager_port: int | None = None):
    """Use a wide, simulator-owned camera and keep RPC/TM connections on the dedicated server."""
    configuration = load_configuration(path)
    scenario_id = next(configuration.dataset.scenarios()).id
    recipe = configuration.dataset.native(scenario_id)
    scenario = deepcopy(recipe.scenario)
    scenario.pop("lead", None)
    scenario["npcs"] = []
    sensor = {
        "type": "CarlaRgbCamera", "name": "camera-0", "source_ID": 0,
        "image_size_x": WIDTH * SCALE, "image_size_y": HEIGHT * SCALE,
        "fov": math.degrees(2 * math.atan(WIDTH * SCALE / (2 * FOCAL))),
        "sensor_tick": 0.0,
    }
    scenario["ego"]["sensors"] = [sensor]
    dataset = CarlaDataset({scenario_id: replace(
        recipe, scenario=scenario, decision_steps=2048,
        motion={"ego": PrescribedMotion(constant_control=Control(brake=1.0))})})
    client = dict(configuration.simulator.client_config)
    client.update(connect_port=port,
                  traffic_manager_port=port + 6000 if traffic_manager_port is None
                  else traffic_manager_port)
    simulator = CarlaSimulator(client, depth_camera="camera-0-0")
    return dataset, simulator, scenario_id


class RoadValidation:
    """Native scene edits with capture/calibration delegated to the simulator runtime."""

    def __init__(self, simulator_runtime, output):
        self.simulator_runtime = simulator_runtime
        self.world = simulator_runtime.native["world"]
        self.ego = simulator_runtime.native["actors"]["ego"]
        self.queue_command = simulator_runtime.native["queue_command"]
        self.output = output
        self.actors = []
        self.rows = []

    def spawn(self, blueprint, transform, *, attributes=None):
        blueprint = self.world.get_blueprint_library().find(blueprint)
        for name, value in (attributes or {}).items():
            blueprint.set_attribute(name, value)
        actor = self.world.spawn_actor(blueprint, transform)
        self.actors.append(actor)
        return actor

    def settle(self, actor):
        import carla

        self.queue_command(carla.command.ApplyVehicleControl(actor.id, carla.VehicleControl(hand_brake=True)))
        for _ in range(60):
            self.simulator_runtime.step(Control(brake=1.0))
        if actor.get_velocity().length() > 0.02:
            raise RuntimeError("Vehicle did not settle before visibility measurement")
        self.queue_command(carla.command.SetSimulatePhysics(actor.id, False))
        return actor.get_transform()

    def place(self, actor, transform):
        import carla

        self.queue_command(carla.command.ApplyTransform(actor.id, transform))

    def capture(self, ego_transform):
        self.place(self.ego, ego_transform)
        # Let rendering settle through the runtime's aligned capture path.
        for _ in range(3):
            self.simulator_runtime.step(Control(brake=1.0))
        view = self.simulator_runtime.sensor_view("camera-0-0")
        if view.depth is None:
            raise RuntimeError("Validation requires aligned simulator depth")
        return view

    def reference(self, background, view, actor, name, *, world_aligned_box=False):
        from PIL import Image

        box = actor.bounding_box
        pose = actor.get_transform()
        if world_aligned_box:
            import carla

            # CARLA combines a static mesh's bounds in world axes and localizes its center.
            box = carla.BoundingBox(pose.transform(box.location), box.extent)
            pose = carla.Transform()
        vertices = box.get_world_vertices(pose)
        world_points = mirror_y_points([(point.x, point.y, point.z) for point in vertices])
        camera_points = transform(view.cam_from_world, world_points)
        near, far = camera_points[:, 2].min(), camera_points[:, 2].max()
        if near <= 0:
            raise RuntimeError(f"{name}: reference box crosses or lies behind the camera plane")
        pixels = view.camera.project(camera_points)
        hull = cv2.convexHull(np.rint(pixels).astype(np.int32))
        projected = np.zeros(view.depth.shape, dtype=np.uint8)
        cv2.fillConvexPoly(projected, hull, 1)
        margin_size = 2 * BOX_MARGIN_PIXELS + 1
        projected = cv2.dilate(projected, np.ones((margin_size, margin_size), dtype=np.uint8))
        within_depth = ((view.depth >= near - BOX_DEPTH_MARGIN_M)
                        & (view.depth <= far + BOX_DEPTH_MARGIN_M))
        mask = (background - view.depth > TOLERANCE) & projected.astype(bool) & within_depth
        if (not mask.any() or mask[0].any() or mask[-1].any()
                or mask[:, 0].any() or mask[:, -1].any()):
            raise RuntimeError(f"{name}: missing/truncated silhouette; enlarge reference camera")
        Image.fromarray(mask.astype(np.uint8) * 255).save(self.output / f"{name}-reference.png")
        return mask, view.depth.copy()

    def measure(self, name, reference, view):
        from PIL import Image

        mask, target_depth = reference
        left, top, right, bottom = BOUNDS
        scene_depth = np.full(mask.shape, np.nan)
        scene_depth[top:bottom, left:right] = view.depth[top:bottom, left:right]
        result = visibility_from_depth(mask, target_depth, scene_depth, camera=view.sensor,
                                       image_bounds=BOUNDS, tolerance_m=TOLERANCE)
        visible = mask & (target_depth <= scene_depth + TOLERANCE)
        row = {"case": name, "frame": view.frame, "fraction": result.fraction,
               "reference_pixels": int(mask.sum()),
               "in_image_pixels": int(mask[top:bottom, left:right].sum()),
               "visible_pixels": int(visible.sum())}
        self.rows.append(row)
        Image.fromarray(view.data[top:bottom, left:right]).save(self.output / f"{name}-rgb.png")
        Image.fromarray(visible[top:bottom, left:right].astype(np.uint8) * 255).save(
            self.output / f"{name}-visible.png")
        print(json.dumps(row), flush=True)
        return result.fraction

    def close(self):
        for actor in reversed(self.actors):
            if actor.is_alive:
                actor.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=2300)
    parser.add_argument("--traffic-manager-port", type=int,
                        help="Dedicated TM port; default is RPC port plus 6000")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/carla_patch.yaml")
    parser.add_argument("--output", type=Path, default=Path("tmp/visibility-road"))
    args = parser.parse_args()
    import carla

    from avsectester.implemented_attacks.patch_attack.online import (
        CarlaPatchPanel, DEFAULT_PROP_MESH_PATH,
    )

    dataset, simulator, scenario_id = validation_configuration(
        args.config, args.port, args.traffic_manager_port)
    args.output.mkdir(parents=True, exist_ok=True)
    with simulator.create_runtime() as simulator_runtime:
        simulator_runtime.reset(dataset.scenario_start(scenario_id, dataset.frames(scenario_id)[0]))
        validation = RoadValidation(simulator_runtime, args.output)
        try:
            ego_transform = validation.settle(validation.ego)

            def relative(x, y=0, z=0, yaw=0):
                location = ego_transform.transform(carla.Location(x, y, z))
                return carla.Transform(location, carla.Rotation(yaw=ego_transform.rotation.yaw + yaw))

            viewpoints = {"front": ego_transform, "clipped": relative(0, yaw=30),
                          "outside": relative(0, yaw=45)}
            backgrounds = {key: validation.capture(pose).depth.copy()
                           for key, pose in viewpoints.items()}
            target = validation.spawn("vehicle.tesla.model3", relative(16, z=.6))
            target_transform = validation.settle(target)
            road = validation.world.get_map().get_waypoint(target_transform.location)
            if abs(target_transform.location.z - road.transform.location.z) > .3:
                raise RuntimeError("Target is not settled on the road")
            references = {}
            for key, pose in viewpoints.items():
                view = validation.capture(pose)
                references[key] = validation.reference(
                    backgrounds[key], view, target, f"vehicle-{key}")
                fraction = validation.measure(f"vehicle-{key}-clear", references[key], view)
                if key == "front":
                    assert fraction > .98
                elif key == "clipped":
                    assert .05 < fraction < .95
                else:
                    assert fraction == 0
            blocker = validation.spawn("vehicle.carlacola.actors", relative(8, 3.5, .6))
            blocker_transform = validation.settle(blocker)
            blocker_box = blocker.bounding_box
            # Put the side of the truck on the target centerline for partial occlusion.
            partial_offset = blocker_box.extent.y - blocker_box.location.y

            def occlusion_cases(kind, refs):
                for name, lateral in [("partial", partial_offset), ("blocked", 0)]:
                    pose = relative(8, lateral)
                    pose.location.z = blocker_transform.location.z
                    validation.place(blocker, pose)
                    for key in ("front", "clipped"):
                        view = validation.capture(viewpoints[key])
                        fraction = validation.measure(f"{kind}-{key}-{name}", refs[key], view)
                        if name == "partial" and key == "front":
                            assert .05 < fraction < .95
                        elif name == "blocked":
                            assert fraction < .02
                validation.place(blocker, blocker_transform)

            occlusion_cases("vehicle", references)
            box = target.bounding_box
            rotation = box.rotation
            box_transform = ActorState(
                np.eye(4), box_center=(box.location.x, -box.location.y, box.location.z),
                box_rotation=(rotation.roll, -rotation.pitch, -rotation.yaw),
            ).box_transform()
            box_rotation = box_transform[:3, :3]
            local = (box_transform @ np.array([-box.extent.x - .01, 0., 0., 1.]))[:3]
            flip = np.diag([1., -1., 1.])
            mounting = np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]])
            native_rotation = flip @ box_rotation @ rotation_matrix((0., 0., 180.)) @ flip @ mounting
            panel = CarlaPatchPanel(tuple(flip @ local), rotation_to_carla(native_rotation),
                                    CheckerboardTexture().rgba(), mesh_path=DEFAULT_PROP_MESH_PATH)
            panels = panel.apply(validation.world, target,
                                 tick=lambda: simulator_runtime.step(Control(brake=1.0)),
                                 queue_command=validation.queue_command)
            validation.actors.extend(panels)
            panel_transform = panels[0].get_transform()
            validation.place(target, relative(-30))
            validation.place(blocker, relative(-40, 4))
            isolated = validation.spawn(panel.prop, panel_transform,
                                        attributes={"mesh_path": panel.mesh_path})
            patch_references = {}
            for key, pose in viewpoints.items():
                view = validation.capture(pose)
                patch_references[key] = validation.reference(
                    backgrounds[key], view, isolated, f"patch-{key}", world_aligned_box=True)
                validation.measure(f"patch-{key}-isolated", patch_references[key], view)
            isolated.destroy()
            validation.place(target, target_transform)
            validation.place(blocker, blocker_transform)
            for key, pose in viewpoints.items():
                validation.measure(f"patch-{key}-mounted", patch_references[key],
                                   validation.capture(pose))
            occlusion_cases("patch", patch_references)
            report = {"simulator": dict(simulator.identity), "camera_size": [WIDTH, HEIGHT],
                      "reference_size": [WIDTH * SCALE, HEIGHT * SCALE], "image_bounds": BOUNDS,
                      "focal_length_pixels": FOCAL, "depth_tolerance_m": TOLERANCE,
                      "method": "Simulator SensorView depth difference within target box projection and depth",
                      "results": validation.rows}
            (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        finally:
            validation.close()


if __name__ == "__main__":
    main()
