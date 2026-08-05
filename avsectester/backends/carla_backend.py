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


class _Sink:
    """Minimal stand-in for avstack's SensorDataManager: keep only the latest frame."""

    def __init__(self) -> None:
        self.latest = None

    def push(self, data: Any) -> None:
        self.latest = data


@BACKENDS.register_module()
class CarlaBackend(Backend):
    """Closed-loop CARLA execution backend.

    ``perception="groundtruth"`` feeds ground-truth NPC boxes through a passthrough detector
    (perfect, simulator-cheap baseline). ``perception="neural"`` attaches a real ``CarlaLidar``
    and runs an mmdet3d detector (default the CARLA-trained ``carla-vehicle`` PointPillars) on
    the point cloud each tick -- genuine neural perception in the loop.
    """

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
        perception: str = "groundtruth",
        nn_model: str = "pointpillars",
        nn_dataset: str = "carla-vehicle",
        nn_threshold: float = 0.3,
        gpu: int = 0,
        record_camera: bool = False,
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
        self.perception = perception
        self.nn_model = nn_model
        self.nn_dataset = nn_dataset
        self.nn_threshold = nn_threshold
        self.gpu = gpu
        self.record_camera = record_camera

        self._perception_hooks: list[Callable] = []
        self._ctx = None
        self._lidar = None
        self._sink = None
        self._camera = None
        self._camera_sink = None
        self._recorder = None
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

    def set_recorder(self, recorder: Any) -> None:
        """Attach a viz.RunRecorder; the backend feeds it points/detections/rgb per tick."""
        self._recorder = recorder

    # -- attack/defense/monitor attachment -------------------------------------
    def profile(self):
        """Advertise seams/capabilities for the configured perception mode.

        Neural mode runs a real LiDAR + learned detector (only the detection-level seam is
        wired today); ground-truth mode feeds object-level boxes through a passthrough
        detector, exposing the object-level input seam as well.
        """
        from ..core.capability import Capability, StackProfile

        if self.perception == "neural":
            return StackProfile.of(
                seams=["perception_out"],
                capabilities=[Capability.NEURAL_PERCEPTION, Capability.RAW_LIDAR, Capability.TRACKER],
            )
        return StackProfile.of(
            seams=["perception_input", "perception_out"],
            capabilities=[Capability.GT_PERCEPTION, Capability.TRACKER],
        )

    def attach(self, plugin: Callable, seam: str = "perception_out") -> None:
        """Attach at ``perception_input`` (object-level pre-loop) or ``perception_out``
        (detector post-hook via the avstack hook adapter)."""
        if seam == "perception_input":
            self._perception_hooks.append(plugin)
        else:
            from ..hooks import attach as attach_hook

            attach_hook(self._perception, plugin, seam, self._ctx)

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

        from ..hooks import RunContext

        if self.perception == "neural":
            self._setup_neural(world)
        else:
            self._perception = Passthrough3DObjectDetector()
        if self.record_camera:
            self._setup_camera(world)
        self._tracker = BasicBoxTracker3D()
        self._ctx = RunContext(run_id="carla")
        self._controller = VehiclePIDController(
            args_lateral={"K_P": 1.2, "K_D": 0.1, "K_I": 0.05},
            args_longitudinal={"K_P": 0.4, "K_D": 0.0, "K_I": 0.05},
        )
        self._plan = WaypointPlan(max_dist=50, dist_arrived=1.5)
        self._idx = 0
        self._frame = 0
        self._t0 = None

    def _setup_neural(self, world: Any) -> None:
        """Attach a real CarlaLidar to the ego and build the neural detector.

        Uses a minimal parent stub (world/actor/reference/ID/sensor_data_manager sink) so we
        reuse avcarla's coordinate-correct ``LidarData`` without its full actor framework.
        """
        from types import SimpleNamespace

        from avcarla.geometry import CarlaReferenceFrame
        from avcarla.sensors import CarlaLidar
        from avstack.geometry import GlobalOrigin3D
        from avstack.modules.perception.object3d import MMDetObjectDetector3D

        egT = self._ego.get_transform()
        ego_ref = CarlaReferenceFrame(
            reference=GlobalOrigin3D,
            location=(egT.location.x, egT.location.y, egT.location.z),
            rotation=(egT.rotation.roll, egT.rotation.pitch, egT.rotation.yaw),
        )
        self._sink = _Sink()
        host = SimpleNamespace(
            world=world, actor=self._ego, reference=ego_ref, ID=self._ego.id,
            sensor_data_manager=self._sink,
        )
        client_stub = SimpleNamespace(map=world.get_map())
        # default avstack CarlaLidar (32-beam) -- the sensor the carla-vehicle model trained on
        self._lidar = CarlaLidar(
            parent=host, client=client_stub,
            rotation_frequency=1.0 / self.fixed_delta_seconds, range=70.0,
        )
        snap = world.get_snapshot()
        self._lidar.initialize(snap.timestamp.elapsed_seconds, snap.frame)
        self._perception = MMDetObjectDetector3D(
            model=self.nn_model, dataset=self.nn_dataset, gpu=self.gpu, threshold=self.nn_threshold,
        )

    def _setup_camera(self, world: Any) -> None:
        """Attach a forward RGB camera (for recording) via the same minimal-parent stub."""
        from types import SimpleNamespace

        from avcarla.geometry import CarlaReferenceFrame
        from avcarla.sensors import CarlaRgbCamera
        from avstack.geometry import GlobalOrigin3D

        egT = self._ego.get_transform()
        ego_ref = CarlaReferenceFrame(
            reference=GlobalOrigin3D,
            location=(egT.location.x, egT.location.y, egT.location.z),
            rotation=(egT.rotation.roll, egT.rotation.pitch, egT.rotation.yaw),
        )
        self._camera_sink = _Sink()
        host = SimpleNamespace(
            world=world, actor=self._ego, reference=ego_ref, ID=self._ego.id,
            sensor_data_manager=self._camera_sink,
        )
        client_stub = SimpleNamespace(map=world.get_map())
        self._camera = CarlaRgbCamera(
            parent=host, client=client_stub, image_size_x=640, image_size_y=360,
        )
        snap = world.get_snapshot()
        self._camera.initialize(snap.timestamp.elapsed_seconds, snap.frame)

    def _forward_hazard(self, ego_state: Any, tracks: Any) -> float | None:
        from .common import forward_hazard

        return forward_hazard(ego_state, tracks, self.brake_distance, self.brake_corridor)

    def _neural_hazard(self, detections: Any) -> float | None:
        """Nearest forward detection within the brake corridor, in the ego/sensor frame."""
        best = None
        for d in detections:
            p = d.position.x  # [forward, left, up] relative to the ego-mounted lidar
            in_corridor = 0.0 < p[0] < self.brake_distance and abs(p[1]) < self.brake_corridor
            if in_corridor and (best is None or p[0] < best):
                best = float(p[0])
        return best

    def _perceive(self, frame: int, t: float, ego_state: Any):
        """Return (n_input, detections, confirmed_tracks, hazard) for the current tick."""
        from avstack.datastructs import DataContainer

        if self.perception == "neural":
            data = self._sink.latest
            self._ctx.tick(frame, t, ego_state=ego_state, ground_truth=None)
            if data is None:
                return 0, [], [], None
            n_input = len(data.data.raw_data) // 16  # 4 float32 (x,y,z,intensity) per point
            detections = self._perception(data, frame=frame)  # perception_out hooks fire here
            try:
                self._tracker(detections, platform=data.calibration.reference)
                confirmed = self._tracker.tracks_confirmed
            except Exception:  # noqa: BLE001 - tracking is best-effort in the sensor frame
                confirmed = detections
            return n_input, detections, confirmed, self._neural_hazard(detections)

        # ground-truth perception
        gt_objects = []
        for npc in self._npcs:
            if npc.is_alive:
                try:
                    gt_objects.append(carla_actor_to_object_state(npc, t))
                except NotImplementedError:
                    continue
        data = DataContainer(frame, t, gt_objects, "carla_gt")
        self._ctx.tick(frame, t, ego_state=ego_state, ground_truth=gt_objects)
        for hook in self._perception_hooks:  # perception_input seam (object level)
            data = hook(data, ego_state=ego_state)
        n_input = len(data)
        detections = self._perception(data, frame=frame)  # perception_out hooks fire here
        self._tracker(detections, platform=self._global_origin())
        confirmed = self._tracker.tracks_confirmed
        return n_input, detections, confirmed, self._forward_hazard(ego_state, confirmed)

    def step(self) -> dict[str, Any]:
        import carla
        from avstack.modules.planning.types import Waypoint

        world, ego = self._world, self._ego
        world.tick()
        self._frame += 1
        frame = self._frame - 1
        t_abs = world.get_snapshot().timestamp.elapsed_seconds
        self._t0 = t_abs if self._t0 is None else self._t0
        t = t_abs - self._t0
        ego_state = carla_actor_to_object_state(ego, t)

        # ---- perception (groundtruth or neural) + attack/defense hooks ----
        n_input, detections, confirmed, hazard = self._perceive(frame, t, ego_state)

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
        throttle, brake = float(ctrl.throttle), float(ctrl.brake)
        if hazard is not None:
            throttle, brake = 0.0, 1.0
        ego.apply_control(carla.VehicleControl(throttle=throttle, steer=float(ctrl.steer), brake=brake))

        record = {
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
        if self._recorder is not None:
            self._recorder.capture(
                record, points=self._latest_points(), detections=detections, rgb=self._latest_rgb()
            )
        return record

    def _latest_points(self) -> Any:
        if self.perception != "neural" or self._sink is None or self._sink.latest is None:
            return None
        import numpy as np

        return np.frombuffer(bytes(self._sink.latest.data.raw_data), dtype=np.float32).reshape(-1, 4)

    def _latest_rgb(self) -> Any:
        if self._camera_sink is None or self._camera_sink.latest is None:
            return None
        return self._camera_sink.latest.data[:, :, ::-1]  # BGR -> RGB

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
            if self._lidar is not None:
                self._lidar.destroy()  # stop the sensor callback before destroying actors
            if self._camera is not None:
                self._camera.destroy()
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
