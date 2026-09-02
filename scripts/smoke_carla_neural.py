"""Closed-loop CARLA smoke with REAL neural perception: clean vs phantom-attacked.

Runs the CarlaBackend with perception="neural" (CarlaLidar -> CARLA-trained PointPillars),
first a clean pass then one with a detection-level PhantomDetectionAttack, and reports the
escalation. Manual, hardware-dependent: needs a CARLA 0.9.15 server on :2000 and the
carla-vehicle weights (./scripts/fetch_models.sh). Run in the avsec env.

Validated result: clean cruises with ~5-6 real detections/frame and 0 brakes; attacked
brakes to a stop for the phantom -> escalation.
"""
from __future__ import annotations

import sys


def _run(attack, frames: int):
    from avsectester.backends.carla_backend import CarlaBackend

    be = CarlaBackend(
        perception="neural", nn_dataset="carla-vehicle", nn_threshold=0.3,
        n_npcs=8, frames=frames, target_speed=6.0, brake_distance=8.0,
    )
    be.build(None)
    if attack is not None:
        attack.reset()
        attack.check(be.supported_seams())
        for seam in attack.seams:
            be.attach(attack, seam)
    recs = []
    try:
        recs = list(be.run())
    finally:
        be.close()
    return recs


def _summary(tag: str, recs: list) -> tuple[float, int]:
    speeds = [r["ego_speed"] for r in recs]
    dets = [r["n_detections"] for r in recs]
    brakes = sum(1 for r in recs if r["braking"])
    print(f"[{tag}] frames={len(recs)} max_speed={max(speeds):.1f} final_speed={speeds[-1]:.1f} "
          f"mean_detections={sum(dets) / max(1, len(dets)):.1f} brake_frames={brakes}")
    return speeds[-1], brakes


def main() -> int:
    from avsectester.attacks.detection_manipulation import PhantomDetectionAttack

    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    print("=== CLEAN neural run ===")
    c_final, c_brakes = _summary("clean", _run(None, frames))
    print("=== ATTACKED neural run (phantom 6 m ahead) ===")
    a_final, a_brakes = _summary("attacked", _run(PhantomDetectionAttack(target_xyz=[6.0, 0.0, -1.5], score=0.9), frames))

    escalated = a_brakes > c_brakes and a_final < 1.0 and c_final > 3.0
    print(f"CARLA NEURAL SMOKE: {'PASS (escalated)' if escalated else 'FAIL'} "
          f"-> clean(final={c_final:.1f}, brakes={c_brakes}) vs attacked(final={a_final:.1f}, brakes={a_brakes})")
    return 0 if escalated else 1


if __name__ == "__main__":
    sys.exit(main())
