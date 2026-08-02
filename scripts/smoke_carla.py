"""Closed-loop CARLA smoke test: connect, sync-mode ticks, ego + LiDAR sensor, collect a frame."""
import queue
import sys

import carla
import numpy as np


def main():
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30.0)
    cv = client.get_client_version()
    sv = client.get_server_version()
    print(f"client {cv} | server {sv}")
    if cv != sv:
        print(f"WARNING: client/server version mismatch ({cv} != {sv})")

    world = client.get_world()
    original_settings = world.get_settings()
    tm = client.get_trafficmanager()

    try:
        # synchronous mode for reproducible perception
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        tm.set_synchronous_mode(True)

        bp = world.get_blueprint_library()
        vehicle_bp = bp.filter("vehicle.*")[0]
        spawn = world.get_map().get_spawn_points()[0]
        ego = world.spawn_actor(vehicle_bp, spawn)
        ego.set_autopilot(True, tm.get_port())
        print(f"spawned ego: {ego.type_id} @ {spawn.location}")

        # attach a LiDAR
        lidar_bp = bp.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", "50")
        lidar_bp.set_attribute("points_per_second", "100000")
        lidar_bp.set_attribute("rotation_frequency", "20")
        lidar_bp.set_attribute("channels", "32")
        lidar = world.spawn_actor(
            lidar_bp, carla.Transform(carla.Location(z=2.4)), attach_to=ego
        )
        q: queue.Queue = queue.Queue()
        lidar.listen(q.put)

        n_frames = 20
        got = 0
        pts_last = None
        for i in range(n_frames):
            world.tick()
            try:
                data = q.get(timeout=2.0)
                pts = np.frombuffer(data.raw_data, dtype=np.float32).reshape(-1, 4)
                pts_last = pts
                got += 1
            except queue.Empty:
                pass
        loc = ego.get_location()
        print(f"ticked {n_frames} frames, received {got} lidar frames")
        if pts_last is not None:
            print(f"  last lidar frame: {pts_last.shape[0]} points, dims {pts_last.shape[1]}")
        print(f"  ego moved to ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f}) under autopilot")

        ok = got >= n_frames // 2 and pts_last is not None and pts_last.shape[0] > 0
        print("CARLA CLOSED-LOOP SMOKE:", "PASS" if ok else "FAIL", flush=True)

        # ---- ordered teardown to avoid callbacks on destroyed actors ----
        lidar.stop()          # detach callback first
        world.tick()          # let the stop propagate
        tm.set_synchronous_mode(False)
        world.apply_settings(original_settings)  # back to async before destroying
        client.apply_batch_sync(
            [carla.command.DestroyActor(lidar), carla.command.DestroyActor(ego)], True
        )
        return 0 if ok else 1
    except Exception as e:  # noqa: BLE001
        print("CARLA CLOSED-LOOP SMOKE: FAIL ->", type(e).__name__, e, flush=True)
        try:
            tm.set_synchronous_mode(False)
            world.apply_settings(original_settings)
        except Exception:  # noqa: BLE001, S110 - best-effort cleanup on the error path
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
