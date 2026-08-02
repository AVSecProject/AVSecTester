"""Working closed-loop demo: avstack's decision stack drives a CARLA ego.

Pipeline (all avstack modules), stepped in CARLA synchronous mode:

    CARLA ground-truth actors
        -> avstack Passthrough3DObjectDetector   (perception, checkpoint-free)
        -> avstack BasicBoxTracker3D             (multi-object tracking)
        -> forward-waypoint planner              (avstack WaypointPlan/Waypoint/Pose;
                                                  inlined — upstream GoStraightPlanner has a bug)
        -> avstack VehiclePIDController          (throttle/steer/brake)
        -> carla.VehicleControl applied to the ego

The ego is driven entirely by avstack (no CARLA autopilot); NPC traffic uses
autopilot. Perception runs in avstack's ground-truth mode (the sim knows every
actor's 3D box) so the demo needs no downloaded model weights — this is also the
exact seam where an attacked/real detector would later be swapped in.

Run (inside the `avsec` conda env, with a CARLA 0.9.15 server on :2000):
    python scripts/demo_avstack_carla.py --frames 200 --npcs 30
"""
from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import carla
from avcarla.geometry import wrap_mobile_actor_to_object_state
from avstack.datastructs import DataContainer
from avstack.geometry import GlobalOrigin3D, Pose
from avstack.geometry import transformations as tforms
from avstack.modules.control.vehicle import VehiclePIDController
from avstack.modules.perception.object3d import Passthrough3DObjectDetector
from avstack.modules.planning.types import Waypoint, WaypointPlan
from avstack.modules.tracking.tracker3d import BasicBoxTracker3D


def push_forward_waypoint(plan: WaypointPlan, ego_state, d_forward: float, target_speed: float):
    """Keep one waypoint `d_forward` metres ahead of the ego.

    This is what avstack's GoStraightPlanner is meant to do, inlined because the
    upstream planner builds its target Pose with position/attitude swapped
    (planning/vehicle.py:50 `Pose(ego_state.attitude, target_loc)`) and raises.
    Uses avstack geometry/plan types so the avstack PID controller consumes it.
    """
    plan.update(ego_state)
    if plan.needs_waypoint():
        forward_vec = tforms.get_rot_yaw_matrix(ego_state.attitude.yaw, "+z")[:, 0]
        target_loc = ego_state.position + d_forward * forward_vec
        target_point = Pose(target_loc, ego_state.attitude)  # (position, attitude)
        plan.push(ego_state.position.distance(target_loc), Waypoint(target_point, target_speed))
    return plan


def to_object_state(actor: carla.Actor, t: float):
    """CARLA vehicle actor -> avstack ObjectState in the global frame.

    Reuses avcarla's tested left->right-handed conversion; it expects a wrapper
    exposing `.ID` and `.actor`, so we pass a tiny shim over the raw actor.
    """
    return wrap_mobile_actor_to_object_state(SimpleNamespace(ID=actor.id, actor=actor), t)


def build_pipeline():
    perception = Passthrough3DObjectDetector()
    tracker = BasicBoxTracker3D()
    plan = WaypointPlan()
    controller = VehiclePIDController(
        args_lateral={"K_P": 0.9, "K_D": 0.05, "K_I": 0.05},
        args_longitudinal={"K_P": 0.4, "K_D": 0.0, "K_I": 0.05},
    )
    return perception, tracker, plan, controller


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--npcs", type=int, default=30)
    ap.add_argument("--rate", type=float, default=20.0, help="sim ticks per second")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    print(f"client {client.get_client_version()} | server {client.get_server_version()}")
    world = client.get_world()
    original_settings = world.get_settings()
    tm = client.get_trafficmanager()
    bp = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    ego = None
    npcs: list = []
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.rate
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)

        # -- ego (driven by avstack, NOT autopilot)
        ego_bp = bp.filter("vehicle.tesla.model3")[0]
        ego = world.spawn_actor(ego_bp, spawn_points[0])
        print(f"spawned ego {ego.type_id} @ spawn[0]")

        # -- NPC traffic (autopilot). Restrict to 4-wheeled cars: avcarla's
        # get_obj_type_from_actor only maps 2/4-wheel actors (trucks raise).
        vehicle_bps = [
            b
            for b in bp.filter("vehicle.*")
            if int(b.get_attribute("number_of_wheels")) == 4
        ]
        for sp in spawn_points[1 : 1 + args.npcs]:
            npc = world.try_spawn_actor(vehicle_bps[len(npcs) % len(vehicle_bps)], sp)
            if npc is not None:
                npc.set_autopilot(True, tm.get_port())
                npcs.append(npc)
        print(f"spawned {len(npcs)} NPC vehicles (autopilot)")

        perception, tracker, plan, controller = build_pipeline()

        t0 = None
        max_confirmed = 0
        for frame in range(args.frames):
            world.tick()
            snap = world.get_snapshot()
            t_abs = snap.timestamp.elapsed_seconds
            t0 = t_abs if t0 is None else t0
            t = t_abs - t0

            # ---- perception: ground-truth NPC boxes -> detections ----
            gt_objects = []
            for npc in npcs:
                if npc.is_alive:
                    try:
                        gt_objects.append(to_object_state(npc, t))
                    except NotImplementedError:
                        continue  # skip actor types avcarla can't classify
            data = DataContainer(frame, t, gt_objects, "carla_gt")
            detections = perception(data, frame=frame)

            # ---- tracking (global frame) ----
            tracker(detections, platform=GlobalOrigin3D)
            confirmed = tracker.tracks_confirmed
            max_confirmed = max(max_confirmed, len(confirmed))

            # ---- planning + control on the ego ----
            ego_state = to_object_state(ego, t)
            plan = push_forward_waypoint(plan, ego_state, d_forward=6.0, target_speed=6.0)
            ctrl = controller(ego_state, plan)
            ego.apply_control(
                carla.VehicleControl(
                    throttle=float(ctrl.throttle),
                    steer=float(ctrl.steer),
                    brake=float(ctrl.brake),
                )
            )

            if frame % 20 == 0 or frame == args.frames - 1:
                speed = ego_state.velocity.norm()
                print(
                    f"  f{frame:3d} t={t:5.2f}s | dets={len(detections):2d} "
                    f"confirmed_tracks={len(confirmed):2d} | ego_speed={speed:4.1f} m/s "
                    f"throttle={ctrl.throttle:.2f} steer={ctrl.steer:+.2f}"
                )

        final_speed = to_object_state(ego, t).velocity.norm()
        ok = max_confirmed > 0 and final_speed > 0.5
        print(
            f"\nSUMMARY: peak confirmed tracks={max_confirmed}, "
            f"final ego speed={final_speed:.1f} m/s"
        )
        print("AVSTACK+CARLA CLOSED-LOOP DEMO:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        try:
            tm.set_synchronous_mode(False)
            world.apply_settings(original_settings)
            client.apply_batch(
                [carla.command.DestroyActor(a) for a in ([ego] if ego else []) + npcs]
            )
        except Exception as e:  # noqa: BLE001 - best-effort cleanup
            print("cleanup note:", type(e).__name__, e)


if __name__ == "__main__":
    sys.exit(main())
