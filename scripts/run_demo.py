#!/usr/bin/env python
"""End-to-end demo / smoke: the phantom attack against real neural perception in real CARLA.

Builds the avcarla closed loop from configs/carla_scenario.yaml, runs it clean then phantom-attacked,
and asserts the attack forced an unsafe stop the clean run never had. Requires a running CARLA
server (docs/DOCKER.md) and the carla-vehicle weights (scripts/fetch_models.sh).

    python scripts/run_demo.py [frames] [--gpu N]

--gpu overrides the perception CUDA device: the config targets GPU 0 (the container's dedicated ego
GPU), but on a single host CARLA already renders on GPU 0, so pass --gpu 1 to avoid contention.
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from avsectester.metric import impact
from avsectester.scenario import run_scenario, set_perception_gpu

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "carla_scenario.yaml"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("frames", nargs="?", type=int, default=40, help="frames per run (default 40)")
    ap.add_argument("--gpu", type=int, default=None, help="override perception CUDA device")
    args = ap.parse_args()

    scenario = set_perception_gpu(yaml.safe_load(CONFIG.read_text()), args.gpu)

    print(f"[clean]    {args.frames} frames ...")
    clean = run_scenario(scenario, attacks=None, frames=args.frames)
    print(f"[clean]    mean_detections={clean.mean_detections:.1f} "
          f"final_speed={clean.final_speed:.2f} brake_frames={clean.braking_frames}")

    print(f"[attacked] {args.frames} frames (phantom) ...")
    attacked = run_scenario(scenario, attacks=scenario["attacks"], frames=args.frames)
    print(f"[attacked] mean_detections={attacked.mean_detections:.1f} "
          f"final_speed={attacked.final_speed:.2f} brake_frames={attacked.braking_frames}")

    result = impact(clean, attacked)
    print(result)
    ok = result.attack_succeeded
    print("SMOKE: PASS (phantom forced an unsafe stop)" if ok else "SMOKE: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
