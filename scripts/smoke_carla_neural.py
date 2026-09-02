"""Smoke: neural CARLA closed loop — clean vs phantom-attacked.

Runs the CarlaEnv + CarlaSystem (real CarlaLidar -> CARLA-trained PointPillars) twice, once
clean and once with a detection-level PhantomDetectionAttack at the perception_out seam, and
reports whether the phantom forces an unsafe stop. Needs a live CARLA 0.9.15 server on :2000
and the carla-vehicle weights (./scripts/fetch_models.sh).

    python scripts/smoke_carla_neural.py [frames]
"""
from __future__ import annotations

import sys


def _run(attack, frames: int) -> list[dict]:
    from avsectester.core import run
    from avsectester.envs import CarlaEnv, CarlaSystem

    env = CarlaEnv(perception="neural", n_npcs=8, frames=frames, target_speed=6.0)
    system = CarlaSystem(perception="neural", nn_dataset="carla-vehicle", nn_threshold=0.3,
                         target_speed=6.0, brake_distance=8.0)
    if attack is not None:
        attack.reset()
        system.attach(attack)
    return run(env, system).records


def _summary(tag: str, recs: list[dict]) -> tuple[float, int]:
    speeds = [r["ego_speed"] for r in recs]
    brakes = sum(1 for r in recs if r["braking"])
    dets = [r["n_detections"] for r in recs]
    print(f"[{tag}] frames={len(recs)} max_speed={max(speeds):.1f} final_speed={speeds[-1]:.1f} "
          f"mean_detections={sum(dets) / max(1, len(dets)):.1f} brake_frames={brakes}")
    return speeds[-1], brakes


def main() -> int:
    from avsectester.attacks import PhantomDetectionAttack

    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    print("=== CLEAN neural run ===")
    c_final, c_brakes = _summary("clean", _run(None, frames))
    print("=== ATTACKED neural run (phantom 6 m ahead) ===")
    a_final, a_brakes = _summary(
        "attacked", _run(PhantomDetectionAttack(target_xyz=[6.0, 0.0, -1.5], score=0.9), frames))

    impacted = a_brakes > c_brakes and a_final < 1.0 and c_final > 3.0
    print("SMOKE:", "PASS" if impacted else "FAIL", "(phantom forced an unsafe stop)")
    return 0 if impacted else 1


if __name__ == "__main__":
    sys.exit(main())
