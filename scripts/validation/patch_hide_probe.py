#!/usr/bin/env python
"""Measure how much a composited patch drops the lead-car detection confidence, per harmonizer/size.

One CARLA reset, roll the ego a few frames toward the lead, then at one frame composite the patch onto
the host-bound rear surface under each variant and report the detector's top car-box confidence. Decides
which deployment actually hides the car (< 0.3 threshold) before a full driving run.
"""
import argparse
import logging
import sys

import yaml
from avsectester.attacks.patch.physical_patch import image_rgba
from avsectester.plane import Control
from avsectester.rendering.harmonizers import ClassicHarmonizer, PCTNetHarmonizer
from avsectester.simulators import carla as carla_sim
from avsectester.scenarios.carla_provider import CarlaSelectionBackend
from avsectester.simulators.patch_insertion import PatchCompositor
from scripts.common.demo_common import build_detector, carla_rear_perturbation
from PIL import Image

from scripts import REPO_ROOT as REPO

OUT = REPO / "tmp" / "patch_probe"


def top_car_conf(detect, rgb):
    best = 0.0
    for _, sc, _ in detect(rgb):
        best = max(best, sc)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=float, default=8.0)
    ap.add_argument("--approach", type=int, default=16)
    ap.add_argument("--texture", default="tmp/patch_optim/phys_texture.png")
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)

    scen = yaml.safe_load((REPO/"configs"/"carla_patch_scenario.yaml").read_text())
    scen.setdefault("lead", {})["gap"] = args.gap
    scen.pop("patches", None)
    detect = build_detector(args.gpu)
    patch = image_rgba(args.texture, 256)

    backend = CarlaSelectionBackend(scen)
    try:
        obs = backend.reset()
        for _ in range(args.approach):
            obs = backend.step(Control(throttle=0.55))
        rgb = carla_sim.camera_view(obs)
        Image.fromarray(rgb).save(OUT/"probe_clean.png")
        base = top_car_conf(detect, rgb)
        print(f"\n=== lead-car top confidence (threshold 0.3), gap {args.gap} m, frame {args.approach} ===")
        print(f"no patch                 : {base:.3f}")
        variants = [
            ("patch + none", 0.85, 0.6, None),
            ("patch + classic", 0.85, 0.6, ClassicHarmonizer()),
            ("patch + libcom PCTNet", 0.85, 0.6, PCTNetHarmonizer(device=args.gpu, strict=True)),
            ("BIG patch + none", 1.0, 0.95, None),
            ("BIG patch + libcom", 1.0, 0.95, PCTNetHarmonizer(device=args.gpu, strict=True)),
        ]
        for name, width_frac, height_frac, harm in variants:
            compositor = PatchCompositor(harm, harmonize=harm is not None)
            perturb = carla_rear_perturbation(
                backend, patch, compositor, width_frac=width_frac, height_frac=height_frac,
            )
            comp = carla_sim.camera_view(perturb(obs))
            conf = top_car_conf(detect, comp)
            tag = "HIDDEN" if conf < 0.3 else ""
            print(f"{name:25s}: {conf:.3f}  {tag}")
            size = "big" if "BIG" in name else "std"
            method = "raw" if harm is None else type(harm).__name__
            Image.fromarray(comp).save(OUT / f"probe_{name.split()[0]}_{size}_{method}.png")
    finally:
        backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
