"""Closed-loop CARLA environment + system (via lib-avstack-carla / avcarla).

``CarlaEnv`` owns the world: connect, spawn ego + NPCs, tick, read sensors, apply control.
``CarlaSystem`` owns the AV stack: perception (ground-truth passthrough or a real neural
detector on a CarlaLidar cloud) -> tracking -> route-follow PID control, with a
forward-collision brake reflex, and attacks/defenses fired at its perception seams.

NOTE: ported to the minimal Environment/System interface from the previous CarlaBackend;
the offline suite (mock) is green, but this closed-loop path has NOT been re-verified against
a live CARLA server yet. Coordinate-frame conversion goes through ``avcarla.geometry``.
Heavy imports (carla / avstack / avcarla) are lazy so the package stays importable core-only.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ..config import ENVIRONMENTS, SYSTEMS
from ..core.environment import Environment
from ..core.frame import Frame
from ..core.seam import Seam
from ..core.system import Outcome, System
from .common import forward_hazard


def carla_actor_to_object_state(actor: Any, t: float) -> Any:
    from avcarla.geometry import wrap_mobile_actor_to_object_state

    return wrap_mobile_actor_to_object_state(SimpleNamespace(ID=actor.id, actor=actor), t)


def build_lane_route(start_wp: Any, n_points: int, step: float = 2.0) -> list:
    route, wp = [start_wp], start_wp
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


@ENVIRONMENTS.register_module()
class CarlaEnv(Environment):
    def __init__(
        self,
        connect_ip: str = "127.0.0.1",
        connect_port: int = 2000,
        town: str | None = None,
        fixed_delta_seconds: float = 0.05,
        n_npcs: int = 15,
        frames: int = 300,
        ego_vehicle: str = "vehicle.tesla.model3",
        perception: str = "groundtruth",   # "groundtruth" | "neural" (selects which sensors to read)
        record_camera: bool = False,
    ) -> None:
        self.connect_ip = connect_ip
        self.connect_port = connect_port
        self.town = town
        self.fixed_delta_seconds = fixed_delta_seconds
        self.n_npcs = n_npcs
        self.frames = frames
        self.ego_vehicle = ego_vehicle
        self.perception = perception
        self.record_camera = record_camera
        self._client = self._world = self._ego = None
        self._orig_settings = self._tm = None
        self._npcs: list = []
        self._lidar = self._sink = self._camera = self._camera_sink = None
        self._route_poses: list = []
        self._i = 0
        self._t0: float | None = None

    # -- lifecycle ------------------------------------------------------------
    def reset(self) -> Frame:
        import carla
        from avcarla.geometry import carla_transform_to_pose

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
        vehicle_bps = [b for b in bp.filter("vehicle.*")
                       if int(b.get_attribute("number_of_wheels")) == 4]
        for sp in spawn_points[1:1 + self.n_npcs]:
            npc = world.try_spawn_actor(vehicle_bps[len(self._npcs) % len(vehicle_bps)], sp)
            if npc is not None:
                npc.set_autopilot(True, self._tm.get_port())
                self._npcs.append(npc)

        world.tick()  # settle: get_location() is invalid until the first tick after spawn
        start_wp = world_map.get_waypoint(self._ego.get_location(), project_to_road=True)
        self._route_poses = [carla_transform_to_pose(w.transform)
                             for w in build_lane_route(start_wp, n_points=self.frames)]
        if self.perception == "neural":
            self._setup_lidar(world)
        if self.record_camera:
            self._setup_camera(world)
        self._i = 0
        self._t0 = None
        world.tick()  # produce the first sensor frame
        return self._frame()

    def step(self, control: Any = None) -> tuple[Frame, bool]:
        import carla

        throttle, steer, brake = control or (0.0, 0.0, 0.0)
        self._ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))
        self._world.tick()
        self._i += 1
        return self._frame(), self._i >= self.frames

    def _frame(self) -> Frame:
        t_abs = self._world.get_snapshot().timestamp.elapsed_seconds
        self._t0 = t_abs if self._t0 is None else self._t0
        t = t_abs - self._t0
        ego = carla_actor_to_object_state(self._ego, t)
        gt = []
        for npc in self._npcs:
            if npc.is_alive:
                try:
                    gt.append(carla_actor_to_object_state(npc, t))
                except NotImplementedError:
                    continue
        sensors: dict[str, Any] = {}
        if self._sink is not None and self._sink.latest is not None:
            sensors["lidar"] = self._sink.latest
        if self._camera_sink is not None and self._camera_sink.latest is not None:
            sensors["camera"] = self._camera_sink.latest
        return Frame(index=self._i, timestamp=t, sensors=sensors, ego=ego, ground_truth=gt,
                     meta={"route": self._route_poses})

    def _ego_ref(self, world):
        from avcarla.geometry import CarlaReferenceFrame
        from avstack.geometry import GlobalOrigin3D

        egT = self._ego.get_transform()
        return CarlaReferenceFrame(
            reference=GlobalOrigin3D,
            location=(egT.location.x, egT.location.y, egT.location.z),
            rotation=(egT.rotation.roll, egT.rotation.pitch, egT.rotation.yaw),
        )

    def _setup_lidar(self, world) -> None:
        from avcarla.sensors import CarlaLidar

        self._sink = _Sink()
        host = SimpleNamespace(world=world, actor=self._ego, reference=self._ego_ref(world),
                               ID=self._ego.id, sensor_data_manager=self._sink)
        client_stub = SimpleNamespace(map=world.get_map())
        self._lidar = CarlaLidar(parent=host, client=client_stub,
                                 rotation_frequency=1.0 / self.fixed_delta_seconds, range=70.0)
        snap = world.get_snapshot()
        self._lidar.initialize(snap.timestamp.elapsed_seconds, snap.frame)

    def _setup_camera(self, world) -> None:
        from avcarla.sensors import CarlaRgbCamera

        self._camera_sink = _Sink()
        host = SimpleNamespace(world=world, actor=self._ego, reference=self._ego_ref(world),
                               ID=self._ego.id, sensor_data_manager=self._camera_sink)
        client_stub = SimpleNamespace(map=world.get_map())
        self._camera = CarlaRgbCamera(parent=host, client=client_stub,
                                      image_size_x=640, image_size_y=360)
        snap = world.get_snapshot()
        self._camera.initialize(snap.timestamp.elapsed_seconds, snap.frame)

    def close(self) -> None:
        try:
            import carla

            if self._lidar is not None:
                self._lidar.destroy()
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


@SYSTEMS.register_module()
class CarlaSystem(System):
    def __init__(
        self,
        perception: str = "groundtruth",
        nn_model: str = "pointpillars",
        nn_dataset: str = "carla-vehicle",
        nn_threshold: float = 0.3,
        gpu: int = 0,
        target_speed: float = 6.0,
        brake_distance: float = 8.0,
        brake_corridor: float = 2.5,
    ) -> None:
        super().__init__()
        self.perception = perception
        self.target_speed = target_speed
        self.brake_distance = brake_distance
        self.brake_corridor = brake_corridor
        # neural mode: only the detection-level seam is wired; GT mode also has the object input
        self.seams = ((Seam.PERCEPTION_OUT,) if perception == "neural"
                      else (Seam.PERCEPTION_INPUT, Seam.PERCEPTION_OUT))

        from avstack.modules.control.vehicle import VehiclePIDController
        from avstack.modules.planning.types import WaypointPlan
        from avstack.modules.tracking.tracker3d import BasicBoxTracker3D

        if perception == "neural":
            from avstack.modules.perception.object3d import MMDetObjectDetector3D
            self._perception = MMDetObjectDetector3D(
                model=nn_model, dataset=nn_dataset, gpu=gpu, threshold=nn_threshold)
        else:
            from avstack.modules.perception.object3d import Passthrough3DObjectDetector
            self._perception = Passthrough3DObjectDetector()
        self._tracker = BasicBoxTracker3D()
        self._controller = VehiclePIDController(
            args_lateral={"K_P": 1.2, "K_D": 0.1, "K_I": 0.05},
            args_longitudinal={"K_P": 0.4, "K_D": 0.0, "K_I": 0.05})
        self._plan = WaypointPlan(max_dist=50, dist_arrived=1.5)
        self._idx = 0

    def _perceive(self, frame: Frame):
        from avstack.datastructs import DataContainer
        from avstack.geometry import GlobalOrigin3D

        if self.perception == "neural":
            cloud = frame.sensors.get("lidar")
            if cloud is None:
                return 0, [], []
            n_input = len(cloud.data.raw_data) // 16
            detections = self._perception(cloud, frame=frame.index)
            detections = self.fire(Seam.PERCEPTION_OUT, detections, frame)
            try:
                self._tracker(detections, platform=cloud.calibration.reference)
                confirmed = self._tracker.tracks_confirmed
            except Exception:  # noqa: BLE001 - tracking best-effort in the sensor frame
                confirmed = detections
            return n_input, detections, confirmed

        data = DataContainer(frame.index, frame.timestamp, list(frame.ground_truth or []), "carla_gt")
        data = self.fire(Seam.PERCEPTION_INPUT, data, frame)
        n_input = len(data)
        detections = self._perception(data, frame=frame.index)
        detections = self.fire(Seam.PERCEPTION_OUT, detections, frame)
        self._tracker(detections, platform=GlobalOrigin3D)
        return n_input, detections, self._tracker.tracks_confirmed

    def _hazard(self, frame: Frame, detections, confirmed):
        if self.perception == "neural":
            best = None
            for d in detections:
                p = d.position.x  # [forward, left, up] in the sensor frame
                if 0.0 < p[0] < self.brake_distance and abs(p[1]) < self.brake_corridor:
                    best = p[0] if best is None else min(best, p[0])
            return best
        return forward_hazard(frame.ego, confirmed, self.brake_distance, self.brake_corridor)

    def process(self, frame: Frame) -> Outcome:
        from avstack.modules.planning.types import Waypoint

        n_input, detections, confirmed = self._perceive(frame)
        hazard = self._hazard(frame, detections, confirmed)

        # route follow (pure pursuit) over frame.meta["route"]
        route = frame.meta.get("route", [])
        ego = frame.ego
        best, best_d = self._idx, 1e9
        for j in range(self._idx, min(self._idx + 40, len(route))):
            d = ego.position.distance(route[j].position)
            if d < best_d:
                best_d, best = d, j
        self._idx = best
        throttle, steer, brake = 0.0, 0.0, 0.0
        if route:
            target = route[min(self._idx + 3, len(route) - 1)]
            self._plan.clear()
            self._plan.push(ego.position.distance(target.position), Waypoint(target, self.target_speed))
            ctrl = self._controller(ego, self._plan)
            throttle, steer, brake = float(ctrl.throttle), float(ctrl.steer), float(ctrl.brake)
        if hazard is not None:   # forward-collision brake reflex
            throttle, brake = 0.0, 1.0

        record = {
            "frame": frame.index, "t": frame.timestamp,
            "n_input": n_input, "n_detections": len(detections), "n_tracks": len(confirmed),
            "ego_speed": float(ego.velocity.norm()), "throttle": throttle, "brake": brake,
            "steer": steer, "hazard_dist": hazard, "braking": hazard is not None,
        }
        return Outcome(control=(throttle, steer, brake), record=record)
