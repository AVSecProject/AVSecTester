"""Working closed-loop demo: avstack's decision stack drives a CARLA ego.

Pipeline (all avstack modules), stepped in CARLA synchronous mode:

    CARLA ground-truth actors
        -> avstack Passthrough3DObjectDetector   (perception, checkpoint-free)
        -> avstack BasicBoxTracker3D             (multi-object tracking)
        -> route follower over CARLA lane waypoints (avstack WaypointPlan/Waypoint/Pose)
        -> avstack VehiclePIDController          (throttle / steer / brake)
        -> carla.VehicleControl applied to the ego

The ego is driven ENTIRELY by avstack's own controller — no CARLA autopilot. To
prove the custom control has real authority (throttle AND steering), the ego
follows the actual road: each step targets a CARLA lane waypoint ahead, so the
avstack lateral PID must steer the car through curves and junction turns. The run
reports heading change + peak steer as evidence the vehicle really turned.

Perception runs in ground-truth mode (the sim knows every actor's 3D box) so the
demo needs no model weights — that detector is the seam where an attacked/real
detector will later be swapped in. NPC traffic uses autopilot.

Run (inside the `avsec` conda env, with a CARLA 0.9.15 server on :2000):
    python scripts/demo_avstack_carla.py --frames 300 --npcs 15
"""
from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import carla
from avcarla.geometry import carla_transform_to_pose, wrap_mobile_actor_to_object_state
from avstack.datastructs import DataContainer
from avstack.geometry import GlobalOrigin3D
from avstack.modules.control.vehicle import VehiclePIDController
from avstack.modules.perception.object3d import Passthrough3DObjectDetector
from avstack.modules.planning.types import Waypoint, WaypointPlan
from avstack.modules.tracking.tracker3d import BasicBoxTracker3D


def to_object_state(actor: carla.Actor, t: float):
    """CARLA vehicle actor -> avstack ObjectState in the global frame.

    Reuses avcarla's tested left->right-handed conversion; it expects a wrapper
    exposing `.ID` and `.actor`, so we pass a tiny shim over the raw actor.
    """
    return wrap_mobile_actor_to_object_state(SimpleNamespace(ID=actor.id, actor=actor), t)


def build_route(start_wp: carla.Waypoint, n_points: int, step: float = 2.0):
    """Follow the lane from start_wp, choosing the most-turning branch at junctions.

    Returns a list of carla.Waypoint. Picking the turning branch guarantees the
    route contains real turns, so the controller's steering authority is exercised.
    """
    route = [start_wp]
    wp = start_wp
    for _ in range(n_points):
        nxts = wp.next(step)
        if not nxts:
            break
        if len(nxts) == 1:
            wp = nxts[0]
        else:  # junction: take the branch whose heading differs most from current
            cur = wp.transform.rotation.yaw
            wp = max(nxts, key=lambda w: abs(((w.transform.rotation.yaw - cur + 180) % 360) - 180))
        route.append(wp)
    return route


def build_pipeline():
    perception = Passthrough3DObjectDetector()
    tracker = BasicBoxTracker3D()
    controller = VehiclePIDController(
        args_lateral={"K_P": 1.2, "K_D": 0.1, "K_I": 0.05},
        args_longitudinal={"K_P": 0.4, "K_D": 0.0, "K_I": 0.05},
        max_steering=0.8,
    )
    return perception, tracker, controller


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--npcs", type=int, default=15)
    ap.add_argument("--rate", type=float, default=20.0, help="sim ticks per second")
    ap.add_argument("--target-speed", type=float, default=6.0, help="m/s cruise target")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    print(f"client {client.get_client_version()} | server {client.get_server_version()}")
    world = client.get_world()
    world_map = world.get_map()
    original_settings = world.get_settings()
    tm = client.get_trafficmanager()
    bp = world.get_blueprint_library()
    spawn_points = world_map.get_spawn_points()

    ego = None
    npcs: list = []
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.rate
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)

        # -- ego (driven by avstack's controller, NOT autopilot)
        ego_bp = bp.filter("vehicle.tesla.model3")[0]
        ego = world.spawn_actor(ego_bp, spawn_points[0])
        print(f"spawned ego {ego.type_id} @ spawn[0]")

        # -- NPC traffic (autopilot). Restrict to 4-wheeled cars: avcarla's
        # get_obj_type_from_actor only maps 2/4-wheel actors (trucks raise).
        vehicle_bps = [
            b for b in bp.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4
        ]
        for sp in spawn_points[1 : 1 + args.npcs]:
            npc = world.try_spawn_actor(vehicle_bps[len(npcs) % len(vehicle_bps)], sp)
            if npc is not None:
                npc.set_autopilot(True, tm.get_port())
                npcs.append(npc)
        print(f"spawned {len(npcs)} NPC vehicles (autopilot)")

        # Settle one tick: in synchronous mode actor.get_location() is invalid
        # (returns the origin) until the first world.tick() after spawn.
        world.tick()

        # -- route to follow: real road with turns, anchored at the ego's true pose
        start_wp = world_map.get_waypoint(ego.get_location(), project_to_road=True)
        route = build_route(start_wp, n_points=args.frames)
        route_poses = [carla_transform_to_pose(w.transform) for w in route]
        print(f"built {len(route)}-waypoint route following the lane")

        perception, tracker, controller = build_pipeline()
        plan = WaypointPlan(max_dist=50, dist_arrived=1.5)

        t0 = None
        idx = 0                     # current route target index
        max_confirmed = 0
        peak_steer = 0.0
        total_heading = 0.0         # cumulative |yaw change| (deg) — path curvature
        max_lat_err = 0.0           # worst cross-track error to the route (m)
        prev_yaw = None
        start_loc = ego.get_location()
        for frame in range(args.frames):
            world.tick()
            snap = world.get_snapshot()
            t_abs = snap.timestamp.elapsed_seconds
            t0 = t_abs if t0 is None else t0
            t = t_abs - t0
            ego_state = to_object_state(ego, t)

            # ---- perception: ground-truth NPC boxes -> detections -> tracking ----
            gt_objects = []
            for npc in npcs:
                if npc.is_alive:
                    try:
                        gt_objects.append(to_object_state(npc, t))
                    except NotImplementedError:
                        continue
            detections = perception(DataContainer(frame, t, gt_objects, "carla_gt"), frame=frame)
            tracker(detections, platform=GlobalOrigin3D)
            confirmed = tracker.tracks_confirmed
            max_confirmed = max(max_confirmed, len(confirmed))

            # ---- route follow (pure-pursuit): nearest route point + lookahead ----
            # Search forward only (monotonic progress) for the closest route pose,
            # then aim a few waypoints beyond it so the lateral PID has a lead target.
            best, best_d = idx, 1e9
            for j in range(idx, min(idx + 40, len(route_poses))):
                d = ego_state.position.distance(route_poses[j].position)
                if d < best_d:
                    best_d, best = d, j
            idx = best
            max_lat_err = max(max_lat_err, best_d)  # cross-track = distance to nearest route pt
            target = route_poses[min(idx + 3, len(route_poses) - 1)]  # ~6 m lookahead
            lead = ego_state.position.distance(target.position)

            # ---- avstack control -> carla.VehicleControl ----
            plan.clear()
            plan.push(lead, Waypoint(target, args.target_speed))
            ctrl = controller(ego_state, plan)
            ego.apply_control(
                carla.VehicleControl(
                    throttle=float(ctrl.throttle),
                    steer=float(ctrl.steer),
                    brake=float(ctrl.brake),
                )
            )
            peak_steer = max(peak_steer, abs(ctrl.steer))

            # ---- heading-change accounting (evidence of real steering) ----
            yaw = ego.get_transform().rotation.yaw
            if prev_yaw is not None:
                total_heading += abs((yaw - prev_yaw + 180) % 360 - 180)
            prev_yaw = yaw

            if frame % 30 == 0 or frame == args.frames - 1:
                print(
                    f"  f{frame:3d} t={t:5.2f}s | tracks={len(confirmed):2d} "
                    f"| speed={ego_state.velocity.norm():4.1f} m/s "
                    f"thr={ctrl.throttle:.2f} steer={ctrl.steer:+.2f} "
                    f"| wp {idx}/{len(route)} xtrack={best_d:4.1f}m heading_sum={total_heading:5.1f}deg"
                )

        dist_travelled = start_loc.distance(ego.get_location())
        # PASS demands the custom control actually STEERED the car through the route,
        # not merely rolled forward: real distance, a real turn, real steering effort.
        ok = (
            max_confirmed > 0
            and dist_travelled > 5.0
            and total_heading > 20.0
            and peak_steer > 0.05
        )
        print(
            f"\nSUMMARY: peak tracks={max_confirmed} | distance={dist_travelled:.1f} m "
            f"| cumulative heading change={total_heading:.1f} deg | peak |steer|={peak_steer:.2f} "
            f"| max cross-track={max_lat_err:.1f} m"
        )
        print("AVSTACK-CONTROLLED CARLA DRIVE:", "PASS" if ok else "FAIL")
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
