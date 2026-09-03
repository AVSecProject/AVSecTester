"""Record CARLA screenshots + BEV detections for a clean vs phantom-attacked neural run.

Runs the neural CarlaEnv/CarlaSystem twice (clean, then PhantomDetectionAttack at perception_out)
with the forward RGB camera on, saving per-frame CARLA screenshots + a bird's-eye detection view
via ``viz.Recorder``. Needs a live CARLA 0.9.15 server + the carla-vehicle weights.

    python scripts/record_carla.py [frames]

Output: results/carla_rec_<clean|attacked>/frames/{rgb,bev}_XXXX.png + records.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path


def _record(outdir: Path, attack, frames: int) -> None:
    from avsectester.core import run
    from avsectester.envs import CarlaEnv, CarlaSystem
    from avsectester.viz import Recorder

    env = CarlaEnv(perception="neural", n_npcs=8, frames=frames, record_camera=True)
    system = CarlaSystem(perception="neural", nn_dataset="carla-vehicle", nn_threshold=0.3,
                         target_speed=6.0, brake_distance=8.0)
    if attack is not None:
        attack.reset()
        system.attach(attack)
    rec = Recorder(outdir, brake_distance=8.0)
    try:
        run(env, system, observer=rec)
    finally:
        rec.close()
    print(f"  saved {len(rec.records)} frames to {outdir}/frames")


def main() -> int:
    from avsectester.attacks import PhantomDetectionAttack

    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    root = Path("results")
    print(f"== CLEAN == ({frames} frames)")
    _record(root / "carla_rec_clean", None, frames)
    print(f"== ATTACKED == ({frames} frames)")
    _record(root / "carla_rec_attacked",
            PhantomDetectionAttack(target_xyz=[6.0, 0.0, -1.5], score=0.9), frames)
    print(f"\nDONE. screenshots under {root}/carla_rec_clean and {root}/carla_rec_attacked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
