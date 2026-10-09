"""CARLA -> :class:`SceneGT`: the ground truth a requirement is validated against, two ways.

* :func:`predict_scene_gt` — **analytic**: given a scenario config + a lead placement (gap, lateral),
  compute an approximate scene without running CARLA. This is an offline preview only.
* :func:`carla_scene_gt` — **live**: from a running ``CarlaBackend`` (ego + real vehicle actors +
  sensor calibration), for initial-scene selection and insertion geometry.

SceneGT ego frame is (x forward, y left, z up); CARLA/UE is (x forward, y right, z up), so ``y`` flips.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from avsectester.scenarios.scene import CameraCalib, EgoState, ObjectGT, SceneGT

_TESLA_EXTENT = (4.7, 2.0, 1.45)  # nominal vehicle L,W,H (m) for analytic prediction

_FLIP_Y = np.diag([1.0, -1.0, 1.0, 1.0])


def carla_rgb_sensor(backend, camera="front"):
    """Resolve a named RGB sensor, accepting a logical alias only when unambiguous."""
    sensors = {
        name: sensor
        for name, sensor in backend.ego.sensors.items()
        if sensor.object.type_id == "sensor.camera.rgb"
    }
    sensor = sensors.get(camera)
    if sensor is None:
        if len(sensors) != 1:
            raise ValueError(f"Select a specific RGB camera from {list(sensors)}")
        sensor = next(iter(sensors.values()))
    return sensor


def carla_actor_pose(actor):
    """Return a canonical world pose at the actor's bounding-box centre.

    CARLA uses left-handed axes. Both world and object axes are reflected so the
    resulting rotation remains right-handed, including roll, pitch and box rotation.
    """
    from avsectester.insertion import ActorPose

    transform = actor.get_transform()
    world_from_actor = np.linalg.inv(np.asarray(transform.get_inverse_matrix()))
    box = actor.bounding_box
    local = np.eye(4)
    if hasattr(box, "rotation"):
        import carla

        local = np.asarray(carla.Transform(box.location, box.rotation).get_matrix())
    else:
        local[:3, 3] = [box.location.x, box.location.y, box.location.z]
    return ActorPose(
        _FLIP_Y @ world_from_actor @ local @ _FLIP_Y,
        (2 * box.extent.x, 2 * box.extent.y, 2 * box.extent.z),
    )


def _pinhole_k(width: int, height: int, fov_deg: float) -> tuple[float, float, float]:
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return f, width / 2.0, height / 2.0


def _project_ego(pts_ego: np.ndarray, cam_pos_ego: np.ndarray, k) -> tuple[np.ndarray, np.ndarray]:
    """Project ego-frame points to a forward-looking pinhole camera at ``cam_pos_ego`` (ego frame).

    Ego (x fwd, y left, z up) -> camera (x right, y down, z fwd): right=-y, down=-z, fwd=x."""
    f, cx, cy = k
    rel = np.asarray(pts_ego, dtype=np.float64) - cam_pos_ego
    x_cam, y_cam, z_cam = -rel[:, 1], -rel[:, 2], rel[:, 0]
    u = cx + f * x_cam / z_cam
    v = cy + f * y_cam / z_cam
    return np.stack([u, v], axis=1), z_cam


def _box2d_from_corners(px: np.ndarray, z: np.ndarray, width: int, height: int):
    """Clamp the projected 3-D-box corners to a 2-D image box; None if the box is entirely behind."""
    front = z > 0.1
    if not front.any():
        return None
    p = px[front]
    x1, y1 = float(np.clip(p[:, 0].min(), 0, width)), float(np.clip(p[:, 1].min(), 0, height))
    x2, y2 = float(np.clip(p[:, 0].max(), 0, width)), float(np.clip(p[:, 1].max(), 0, height))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _corners_ego(center: np.ndarray, extent: tuple, yaw: float) -> np.ndarray:
    """8 corners of an axis-box at ``center`` (ego frame), size ``extent`` (L,W,H), heading ``yaw``."""
    lx, wy, hz = extent[0] / 2, extent[1] / 2, extent[2] / 2
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for sx in (-lx, lx):
        for sy in (-wy, wy):
            for sz in (-hz, hz):
                out.append(
                    [center[0] + sx * c - sy * s, center[1] + sx * s + sy * c, center[2] + sz]
                )
    return np.array(out)


def _camera_config(scenario: dict) -> tuple[str, int, int, float, np.ndarray]:
    """Front RGB camera (name, width, height, fov, position in ego frame) from a scenario config."""
    for s in scenario.get("ego", {}).get("sensors", []):
        if s.get("type") == "CarlaRgbCamera":
            loc = (s.get("reference", {}) or {}).get("location", [1.0, 0.0, 1.6])
            pos_ego = np.array([loc[0], -loc[1], loc[2]])  # UE y (right) -> ego y (left)
            return (
                "front",
                int(s.get("image_size_x", 800)),
                int(s.get("image_size_y", 600)),
                float(s.get("fov", 90)),
                pos_ego,
            )
    return ("front", 800, 600, 90.0, np.array([1.0, 0.0, 1.6]))


def predict_scene_gt(
    scenario: dict, gap: float, lateral: float, speed: float = 0.0, extent: tuple = _TESLA_EXTENT
) -> SceneGT:
    """The SceneGT for a lead vehicle ``gap`` m ahead and ``lateral`` m to the ego's left, as the
    scenario's front camera would see it — computed analytically (no CARLA)."""
    name, w, h, fov, cam_pos = _camera_config(scenario)
    k = _pinhole_k(w, h, fov)
    # `lateral` matches the scenario's `lead.lateral` (CARLA right +); ego frame y is left, so y = -lateral
    y_left = -lateral
    center = np.array([gap, y_left, extent[2] / 2])  # lead centre in the ego frame (on the road)
    px, z = _project_ego(_corners_ego(center, extent, yaw=0.0), cam_pos, k)
    box2d = _box2d_from_corners(px, z, w, h)
    lead = ObjectGT(
        track_id="lead",
        category="vehicle",
        center=(gap, y_left, extent[2] / 2),
        extent=extent,
        yaw=0.0,
        box2d={name: box2d} if box2d else {},
    )
    cam_to_ego = np.eye(4)
    cam_to_ego[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    cam_to_ego[:3, 3] = cam_pos
    calib = CameraCalib(name=name, width=w, height=h, model=k, cam_to_ego=cam_to_ego)
    return SceneGT(
        frame=0,
        t=0.0,
        ego=EgoState(speed=speed),
        cameras={name: calib},
        objects=[lead],
        source={"backend": "carla", "params": {"gap": gap, "lateral": lateral}},
    )


def carla_scene_gt(backend: Any, camera: str = "front") -> SceneGT:
    """Snapshot live actors and calibration for initial-scene selection.

    Stable logical names (``lead`` and ``npc:<index>``) survive paired resets. Other
    actors retain their native IDs. The complete native world is available separately
    through the selection context.
    """
    from avsectester.rendering.cameras import carla_cam_coords, project_to_pixels

    world = backend.client.world
    ego = backend.ego.actor
    ego_tf = ego.get_transform()
    ego_inv = np.array(ego_tf.get_inverse_matrix())
    sensor = carla_rgb_sensor(backend, camera)
    k_full = np.asarray(sensor.P)[:, :3]
    cam_inv = np.array(sensor.object.get_transform().get_inverse_matrix())
    h_img, w_img = sensor.imsize

    def to_ego(xyz):  # world point -> ego frame (x fwd, y left, z up)
        p = ego_inv @ np.array([xyz[0], xyz[1], xyz[2], 1.0])
        return (float(p[0]), float(-p[1]), float(p[2]))

    aliases = {}
    if getattr(backend, "lead", None) is not None:
        aliases[backend.lead.id] = "lead"
    for index, npc in enumerate(getattr(backend, "npcs", ())):
        aliases[npc.actor.id] = f"npc:{index}"
    objects = []
    for actor in world.get_actors().filter("*vehicle*"):
        if actor.id == ego.id:
            continue
        tf = actor.get_transform()
        bb = actor.bounding_box
        # NB: carla.Transform.transform() mutates its argument in place, so derive the box centre from
        # the world vertices (never call tf.transform(bb.location), which corrupts bb for the next call).
        corners = np.array([[v.x, v.y, v.z] for v in bb.get_world_vertices(tf)])
        cx, cy, cz = to_ego(corners.mean(axis=0))
        cc = carla_cam_coords(corners, cam_inv)
        box2d = _box2d_from_corners(project_to_pixels(cc, k_full), cc[:, 2], w_img, h_img)
        # Include full rotation for host attachments. Reading vertices avoids mutating
        # CARLA's bounding-box location while computing its centre.
        object_pose = _FLIP_Y @ ego_inv @ _FLIP_Y @ carla_actor_pose(actor).transform
        yaw = math.atan2(object_pose[1, 0], object_pose[0, 0])
        objects.append(
            ObjectGT(
                track_id=aliases.get(actor.id, str(actor.id)),
                category="vehicle",
                center=(cx, cy, cz),
                extent=(bb.extent.x * 2, bb.extent.y * 2, bb.extent.z * 2),
                yaw=math.atan2(math.sin(yaw), math.cos(yaw)),
                box2d={camera: box2d} if box2d else {},
                pose=object_pose,
            )
        )
    state = backend.ego.get_object_state()
    # Convert Unreal ego axes to our x-forward/y-left/z-up convention.
    camera_to_ego = np.diag([1, -1, 1, 1]) @ ego_inv @ np.linalg.inv(cam_inv)
    camera_axes = np.eye(4)
    camera_axes[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, -1, 0]]
    camera_to_ego = camera_to_ego @ camera_axes
    snapshot = world.get_snapshot()
    ego_box = ego.bounding_box
    return SceneGT(
        frame=snapshot.frame,
        t=snapshot.timestamp.elapsed_seconds,
        ego=EgoState(
            speed=float(state.velocity.norm()),
            pose=_FLIP_Y @ np.linalg.inv(ego_inv) @ _FLIP_Y,
            center=(ego_box.location.x, -ego_box.location.y, ego_box.location.z),
            extent=(2 * ego_box.extent.x, 2 * ego_box.extent.y, 2 * ego_box.extent.z),
        ),
        cameras={
            camera: CameraCalib(
                name=camera, width=w_img, height=h_img, model=k_full, cam_to_ego=camera_to_ego
            )
        },
        objects=objects,
        source={"backend": "carla", "live": True},
    )
