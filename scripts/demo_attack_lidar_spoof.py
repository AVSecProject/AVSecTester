"""End-to-end security demo: a LiDAR-spoof phantom escalates to an unsafe stop.

Runs the CarlaBackend twice on the same route and compares:

    clean     : no attack        -> ego cruises the route
    attacked  : LidarSpoofAttack  -> phantom obstacle injected at the perception
                input -> confirmed track -> forward-collision brake -> ego stops

This is the attack-escalation path end to end:
    attack signal (injected object) -> component error (phantom track)
                                     -> driving consequence (unsafe hard stop)

The attack attaches purely as a perception-input hook — the backend/AV code is
unchanged between the two runs. Perception is ground-truth mode (no checkpoints).

Run (inside the `avsec` conda env, with a CARLA 0.9.15 server on :2000):
    python scripts/demo_attack_lidar_spoof.py
"""
from __future__ import annotations

import argparse
import sys

from avsectester.attacks.sensor.lidar_spoof import LidarSpoofAttack
from avsectester.backends.carla_backend import CarlaBackend

WARMUP = 15  # frames the ego spends accelerating from rest; ignore for min-speed


def summarize(records: list[dict]) -> dict:
    after = records[WARMUP:] or records
    return {
        "peak_tracks": max(r["n_tracks"] for r in records),
        "max_speed": max(r["ego_speed"] for r in records),
        "min_speed_cruise": min(r["ego_speed"] for r in after),
        "brake_frames": sum(1 for r in records if r["braking"]),
        "final_speed": records[-1]["ego_speed"],
    }


def run(host: str, port: int, frames: int, attack: LidarSpoofAttack | None) -> list[dict]:
    backend = CarlaBackend(
        connect_ip=host, connect_port=port, n_npcs=0, frames=frames, target_speed=6.0
    )
    backend.build()
    if attack is not None:
        backend.add_perception_hook(attack)
    try:
        records = list(backend.run())
    finally:
        backend.close()
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--frames", type=int, default=140)
    args = ap.parse_args()

    print("=== CLEAN run (no attack) ===")
    clean = run(args.host, args.port, args.frames, attack=None)
    cs = summarize(clean)
    print(f"  peak_tracks={cs['peak_tracks']} max_speed={cs['max_speed']:.1f} "
          f"min_cruise_speed={cs['min_speed_cruise']:.1f} brake_frames={cs['brake_frames']} "
          f"final_speed={cs['final_speed']:.1f}")

    print("\n=== ATTACKED run (LidarSpoofAttack: phantom 12 m ahead) ===")
    attack = LidarSpoofAttack(target_xyz=[12.0, 0.0, 0.0])
    attacked = run(args.host, args.port, args.frames, attack=attack)
    as_ = summarize(attacked)
    print(f"  peak_tracks={as_['peak_tracks']} max_speed={as_['max_speed']:.1f} "
          f"min_cruise_speed={as_['min_speed_cruise']:.1f} brake_frames={as_['brake_frames']} "
          f"final_speed={as_['final_speed']:.1f}")
    # show the moment of escalation
    for r in attacked:
        if r["braking"]:
            print(f"  -> first brake at frame {r['frame']} (t={r['t']:.2f}s): "
                  f"phantom track {r['hazard_dist']:.1f} m ahead, speed {r['ego_speed']:.1f} m/s")
            break

    print("\n=== ESCALATION ===")
    escalated = (
        cs["peak_tracks"] == 0
        and cs["brake_frames"] == 0               # clean ego never brakes
        and cs["final_speed"] > 3.0               # ... and is still driving at the end
        and as_["peak_tracks"] >= 1               # phantom became a track
        and as_["brake_frames"] > 0               # brake reflex fired
        and as_["final_speed"] < 0.5              # ego forced to a stop
    )
    print(f"  clean: no phantom, 0 brakes, ego still driving at {cs['final_speed']:.1f} m/s")
    print(f"  attacked: 1 injected object -> {as_['peak_tracks']} phantom track(s) -> "
          f"{as_['brake_frames']} brake frames -> ego stopped ({as_['final_speed']:.1f} m/s)")
    print("LIDAR-SPOOF ESCALATION DEMO:", "PASS" if escalated else "FAIL")
    return 0 if escalated else 1


if __name__ == "__main__":
    sys.exit(main())
