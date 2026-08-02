"""Closed-loop CARLA backend (wraps lib-avstack-carla / avcarla + avstack modules).

Productionizes the demo loop (scripts/demo_avstack_carla.py) behind the ``Backend``
interface: connect → spawn ego + NPCs → synchronous stepping of an avstack pipeline
(perception → tracking → route-follow → PID control) → teardown.

Two things make it useful for security testing:

- **Perception-input hook seam** (``add_perception_hook``): every registered hook is
  applied to the perception INPUT each tick, in avstack pre-hook shape
  ``hook(data, ego_state=...) -> data``. Attacks/defenses/monitors attach here. With the
  ground-truth ``Passthrough3DObjectDetector`` this is the object-level seam; the identical
  hook moves to the raw-LiDAR boundary once a neural detector replaces it.
- **Forward-collision brake reflex** so perception/track errors have a *driving consequence*:
  a confirmed track inside the ego's forward corridor forces throttle→0, brake→1. This is
  what lets an injected phantom escalate into an unsafe stop.

Heavy imports (carla / avstack) are lazy so the package stays importable core-only.

Coordinate-frame note: CARLA is left-handed; avstack/mmdet3d are right-handed. All
CARLA↔avstack conversion goes through ``avcarla.geometry`` (the tested ``[x,-y,z]`` flip).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..config import BACKENDS
from ..core.interfaces import Backend


def carla_actor_to_object_state(actor: Any, t: float) -> Any:
    """Raw CARLA vehicle actor -> avstack ObjectState (global frame)."""
    from avcarla.geometry import wrap_mobile_actor_to_object_state

    return wrap_mobile_actor_to_object_state(SimpleNamespace(ID=actor.id, actor=actor), t)


def build_lane_route(start_wp: Any, n_points: int, step: float = 2.0) -> list:
    """Follow the lane from ``start_wp``, taking the most-turning branch at junctions."""
    route = [start_wp]
    wp = start_wp
    for _ in range(n_points):
        nxts = wp.next(step)
        if not nxts:
            break
        if len(nxts) == 1:
            wp = nxts[0]
        else:
            cur = wp.transform.rotation.yaw
            wp = max(nxts, key=lambda w: abs(((w.transform.rotation.yaw - cur + 180) % 360) - 180))
        route.append(wp)
    return route


@BACKENDS.register_module()
class CarlaBackend(Backend):
    """Closed-loop CARLA execution backend."""

    def __init__(
        self,
        connect_ip: str = "127.0.0.1",
        connect_port: int = 2000,
        town: str | None = None,
        fixed_delta_seconds: float = 0.05,
        n_npcs: int = 15,
        frames: int = 300,
        target_speed: float = 6.0,
        ego_vehicle: str = "vehicle.tesla.model3",
        brake_distance: float = 8.0,
        brake_corridor: float = 2.5,
    ) -> None:
        self.connect_ip = connect_ip
        self.connect_port = connect_port
        self.town = town
        self.fixed_delta_seconds = fixed_delta_seconds
        self.n_npcs = n_npcs
        self.frames = frames
        self.target_speed = target_speed
        self.ego_vehicle = ego_vehicle
        self.brake_distance = brake_distance
        self.brake_corridor = brake_corridor

        self._perception_hooks: list[Callable] = []
        self._client = None
        self._world = None
        self._orig_settings = None
        self._tm = None
        self._ego = None
        self._npcs: list = []
        self._perception = None
        self._tracker = None
        self._controller = None
        self._plan = None
        self._route_poses: list = []
        self._idx = 0
        self._frame = 0
        self._t0: float | None = None

    # -- attack/defense/monitor attachment -------------------------------------
    def add_perception_hook(self, hook: Callable) -> None:
        """Register ``hook(perception_input, ego_state=...) -> perception_input``."""
        self._perception_hooks.append(hook)

    # -- lifecycle -------------------------------------------------------------
    def build(self, spec: Any = None) -> None:
        import carla
        from avcarla.geometry import carla_transform_to_pose
        from avstack.modules.control.vehicle import VehiclePIDController
        from avstack.modules.perception.object3d import Passthrough3DObjectDetector
        from avstack.modules.planning.types import WaypointPlan
        from avstack.modules.tracking.tracker3d import BasicBoxTracker3D

        # allow the experiment spec to override scenario knobs
        if spec is not None:
            ic = getattr(getattr(spec, "scenario", None), "initial_conditions", {}) or {}
            self.n_npcs = ic.get("n_npcs", self.n_npcs)
            self.frames = ic.get("frames", self.frames)
            self.target_speed = ic.get("target_speed", self.target_speed)

        client = carla.Client(self.connect_ip, self.connect_port)
        client.set_timeout(30.0)
        if self.town:
            client.load_world(self.town)
        world = client.get_world()
        world_map = world.get_map()
        self._client, self._world = client, world
        self._orig_settings = world.get_settings()
        self._tm = client.get_trafficmanager()

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        world.apply_settings(settings)
        self._tm.set_synchronous_mode(True)

        bp = world.get_blueprint_library()
        spawn_points = world_map.get_spawn_points()

        self._ego = world.spawn_actor(bp.filter(self.ego_vehicle)[0], spawn_points[0])

        vehicle_bps = [
            b for b in bp.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4
        ]
        for sp in spawn_points[1 : 1 + self.n_npcs]:
            npc = world.try_spawn_actor(vehicle_bps[len(self._npcs) % len(vehicle_bps)], sp)
            if npc is not None:
                npc.set_autopilot(True, self._tm.get_port())
                self._npcs.append(npc)

        # settle one tick: actor.get_location() is invalid until the first tick after spawn
        world.tick()
        start_wp = world_map.get_waypoint(self._ego.get_location(), project_to_road=True)
        route = build_lane_route(start_wp, n_points=self.frames)
        self._route_poses = [carla_transform_to_pose(w.transform) for w in route]

        self._perception = Passthrough3DObjectDetector()
        self._tracker = BasicBoxTracker3D()
        self._controller = VehiclePIDController(
            args_lateral={"K_P": 1.2, "K_D": 0.1, "K_I": 0.05},
            args_longitudinal={"K_P": 0.4, "K_D": 0.0, "K_I": 0.05},
        )
        self._plan = WaypointPlan(max_dist=50, dist_arrived=1.5)
        self._idx = 0
        self._frame = 0
        self._t0 = None

    def _forward_hazard(self, ego_state: Any, tracks: Any) -> float | None:
        """Return forward distance (m) to the nearest confirmed track in the ego corridor."""
        from avstack.geometry import transformations as tforms

        R = tforms.get_rot_yaw_matrix(ego_state.attitude.yaw, "+z")
        fwd, left = R[:, 0], R[:, 1]
        ego_pos = ego_state.position.x
        nearest = None
        for tk in tracks:
            d = np.asarray(tk.position.x) - ego_pos
            dx, dy = float(np.dot(d, fwd)), float(np.dot(d, left))
            if 0.0 < dx < self.brake_distance and abs(dy) < self.brake_corridor:
                nearest = dx if nearest is None else min(nearest, dx)
        return nearest

    def step(self) -> dict[str, Any]:
        import carla
        from avstack.datastructs import DataContainer
        from avstack.modules.planning.types import Waypoint

        world, ego = self._world, self._ego
        world.tick()
        self._frame += 1
        frame = self._frame - 1
        t_abs = world.get_snapshot().timestamp.elapsed_seconds
        self._t0 = t_abs if self._t0 is None else self._t0
        t = t_abs - self._t0
        ego_state = carla_actor_to_object_state(ego, t)

        # ---- perception input: ground-truth NPC boxes ----
        gt_objects = []
        for npc in self._npcs:
            if npc.is_alive:
                try:
                    gt_objects.append(carla_actor_to_object_state(npc, t))
                except NotImplementedError:
                    continue
        data = DataContainer(frame, t, gt_objects, "carla_gt")

        # ---- attack/defense/monitor hooks on the perception INPUT ----
        for hook in self._perception_hooks:
            data = hook(data, ego_state=ego_state)
        n_input = len(data)

        # ---- perception + tracking ----
        detections = self._perception(data, frame=frame)
        self._tracker(detections, platform=self._global_origin())
        confirmed = self._tracker.tracks_confirmed

        # ---- route follow (pure pursuit) ----
        best, best_d = self._idx, 1e9
        for j in range(self._idx, min(self._idx + 40, len(self._route_poses))):
            d = ego_state.position.distance(self._route_poses[j].position)
            if d < best_d:
                best_d, best = d, j
        self._idx = best
        target = self._route_poses[min(self._idx + 3, len(self._route_poses) - 1)]

        # ---- avstack control + forward-collision brake reflex ----
        self._plan.clear()
        self._plan.push(ego_state.position.distance(target.position), Waypoint(target, self.target_speed))
        ctrl = self._controller(ego_state, self._plan)
        hazard = self._forward_hazard(ego_state, confirmed)
        throttle, brake = float(ctrl.throttle), float(ctrl.brake)
        if hazard is not None:
            throttle, brake = 0.0, 1.0
        ego.apply_control(carla.VehicleControl(throttle=throttle, steer=float(ctrl.steer), brake=brake))

        return {
            "frame": frame,
            "t": t,
            "n_input": n_input,
            "n_detections": len(detections),
            "n_tracks": len(confirmed),
            "ego_speed": float(ego_state.velocity.norm()),
            "throttle": throttle,
            "brake": brake,
            "steer": float(ctrl.steer),
            "hazard_dist": hazard,
            "braking": hazard is not None,
        }

    @staticmethod
    def _global_origin() -> Any:
        from avstack.geometry import GlobalOrigin3D

        return GlobalOrigin3D

    def run(self) -> Iterator[dict[str, Any]]:
        for _ in range(self.frames):
            yield self.step()

    def close(self) -> None:
        import carla

        try:
            if self._tm is not None:
                self._tm.set_synchronous_mode(False)
            if self._world is not None and self._orig_settings is not None:
                self._world.apply_settings(self._orig_settings)
            if self._client is not None:
                actors = ([self._ego] if self._ego else []) + self._npcs
                self._client.apply_batch([carla.command.DestroyActor(a) for a in actors])
        except Exception:  # noqa: BLE001, S110 - best-effort teardown
            pass
        self._client = self._world = self._ego = None
        self._npcs = []
