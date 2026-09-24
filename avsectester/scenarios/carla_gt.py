"""CARLA -> :class:`SceneGT`: the ground truth a requirement is validated against, two ways.

* :func:`predict_scene_gt` — **analytic**: given a scenario config + a lead placement (gap, lateral),
  compute the SceneGT a front camera would see, without running CARLA. Lets
  :class:`~avsectester.scenarios.source.CarlaScenarioBuilder` enumerate qualifying placements cheaply
  and build the real backend only for the ones it runs.
* :func:`carla_scene_gt` — **live**: from a running ``CarlaBackend`` (ego + real vehicle actors +
  sensor calibration), for run-time validation that a built scene actually satisfies the requirement.

SceneGT ego frame is (x forward, y left, z up); CARLA/UE is (x forward, y right, z up), so ``y`` flips.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from avsectester.scenarios.scene import CameraCalib, EgoState, ObjectGT, SceneGT

_TESLA_EXTENT = (4.7, 2.0, 1.45)  # nominal vehicle L,W,H (m) for analytic prediction


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
                out.append([center[0] + sx * c - sy * s, center[1] + sx * s + sy * c, center[2] + sz])
    return np.array(out)


def _camera_config(scenario: dict) -> tuple[str, int, int, float, np.ndarray]:
    """Front RGB camera (name, width, height, fov, position in ego frame) from a scenario config."""
    for s in scenario.get("ego", {}).get("sensors", []):
        if s.get("type") == "CarlaRgbCamera":
            loc = (s.get("reference", {}) or {}).get("location", [1.0, 0.0, 1.6])
            pos_ego = np.array([loc[0], -loc[1], loc[2]])  # UE y (right) -> ego y (left)
            return ("front", int(s.get("image_size_x", 800)), int(s.get("image_size_y", 600)),
                    float(s.get("fov", 90)), pos_ego)
    return ("front", 800, 600, 90.0, np.array([1.0, 0.0, 1.6]))


def predict_scene_gt(scenario: dict, gap: float, lateral: float, speed: float = 0.0,
                     extent: tuple = _TESLA_EXTENT) -> SceneGT:
    """The SceneGT for a lead vehicle ``gap`` m ahead and ``lateral`` m to the ego's left, as the
    scenario's front camera would see it — computed analytically (no CARLA)."""
    name, w, h, fov, cam_pos = _camera_config(scenario)
    k = _pinhole_k(w, h, fov)
    # `lateral` matches the scenario's `lead.lateral` (CARLA right +); ego frame y is left, so y = -lateral
    y_left = -lateral
    center = np.array([gap, y_left, extent[2] / 2])  # lead centre in the ego frame (on the road)
    px, z = _project_ego(_corners_ego(center, extent, yaw=0.0), cam_pos, k)
    box2d = _box2d_from_corners(px, z, w, h)
    lead = ObjectGT(track_id="lead", category="vehicle", center=(gap, y_left, extent[2] / 2),
                    extent=extent, yaw=0.0, box2d={name: box2d} if box2d else {}, visibility=1.0)
    calib = CameraCalib(name=name, width=w, height=h)
    return SceneGT(frame=0, t=0.0, ego=EgoState(speed=speed), cameras={name: calib}, objects=[lead],
                   source={"backend": "carla", "params": {"gap": gap, "lateral": lateral}})


def carla_scene_gt(backend: Any, camera: str = "front") -> SceneGT:
    """Live ground truth from a running ``CarlaBackend``: real vehicle actors -> ObjectGT (ego frame),
    3-D boxes projected to 2-D via the ego's RGB sensor. Run-time validation of a built scene."""
    from avsectester.simulators.patch_insertion import carla_cam_coords, project_to_pixels

    world = backend.client.world
    ego = backend.ego.actor
    ego_tf = ego.get_transform()
    ego_inv = np.array(ego_tf.get_inverse_matrix())
    sensor = next(iter(backend.ego.sensors.values()))
    k_full = np.asarray(sensor.P)[:, :3]
    cam_inv = np.array(sensor.object.get_transform().get_inverse_matrix())
    h_img, w_img = sensor.imsize

    def to_ego(xyz):  # world point -> ego frame (x fwd, y left, z up)
        p = ego_inv @ np.array([xyz[0], xyz[1], xyz[2], 1.0])
        return (float(p[0]), float(-p[1]), float(p[2]))

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
        yaw = math.radians(tf.rotation.yaw - ego_tf.rotation.yaw)
        objects.append(ObjectGT(
            track_id=str(actor.id), category="vehicle", center=(cx, cy, cz),
            extent=(bb.extent.x * 2, bb.extent.y * 2, bb.extent.z * 2),
            yaw=math.atan2(math.sin(yaw), math.cos(yaw)),
            box2d={camera: box2d} if box2d else {}))
    state = backend.ego.get_object_state()
    return SceneGT(frame=0, t=0.0, ego=EgoState(speed=float(state.velocity.norm())),
                   cameras={camera: CameraCalib(name=camera, width=w_img, height=h_img)},
                   objects=objects, source={"backend": "carla", "live": True})
