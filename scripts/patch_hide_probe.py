#!/usr/bin/env python
"""Measure how much a composited patch drops the lead-car detection confidence, per harmonizer/size.

One CARLA reset, roll the ego a few frames toward the lead, then at one frame composite the patch onto
the projected rear quad under each variant and report the detector's top car-box confidence. Decides
which deployment actually hides the car (< 0.3 threshold) before a full driving run.
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml
from avsectester.attacks.patch_composite import ClassicHarmonizer, PatchCompositor, PCTNetHarmonizer
from avsectester.attacks.physical_patch import image_rgba
from avsectester.plane import Control
from avsectester.simulators import carla as carla_sim
from avsectester.simulators.carla import CarlaBackend, lead_rear_quad
from demo_common import build_detector
from PIL import Image

REPO = Path(__file__).resolve().parents[1]; OUT = REPO/"tmp"/"patch_probe"

def top_car_conf(detect, rgb):
    best = 0.0
    for b, sc, _ in detect(rgb):
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
    scen.setdefault("lead", {})["gap"] = args.gap; scen.pop("patches", None)
    detect = build_detector(args.gpu)
    patch = image_rgba(args.texture, 256)

    backend = CarlaBackend(scen)
    try:
        obs = backend.reset()
        for _ in range(args.approach):
            obs = backend.step(Control(throttle=0.55))
        rgb = carla_sim.camera_view(obs)
        quad = lead_rear_quad(backend, width_frac=0.85, height_frac=0.6)(obs)
        quad_big = lead_rear_quad(backend, width_frac=1.0, height_frac=0.95)(obs)
        Image.fromarray(rgb).save(OUT/"probe_clean.png")
        base = top_car_conf(detect, rgb)
        print(f"\n=== lead-car top confidence (threshold 0.3), gap {args.gap} m, frame {args.approach} ===")
        print(f"no patch                 : {base:.3f}")
        if quad is None:
            print("quad None (lead not in view) — increase --gap/--approach"); return 1
        variants = [
            ("patch + none (raw warp)", quad, None),
            ("patch + classic",         quad, ClassicHarmonizer()),
            ("patch + libcom PCTNet",   quad, PCTNetHarmonizer()),
            ("BIG patch + none",        quad_big, None),
            ("BIG patch + libcom",      quad_big, PCTNetHarmonizer()),
        ]
        for name, q, harm in variants:
            if q is None:
                print(f"{name:25s}: quad None"); continue
            if harm is None:
                from avsectester.attacks.patch_composite import warp_patch
                comp, _ = warp_patch(rgb, q, patch)
            else:
                comp = PatchCompositor(harm).apply(rgb, q, patch)
            conf = top_car_conf(detect, comp)
            tag = "HIDDEN" if conf < 0.3 else ""
            print(f"{name:25s}: {conf:.3f}  {tag}")
            Image.fromarray(comp).save(OUT/f"probe_{name.split()[0]}_{'big' if 'BIG' in name else 'std'}_{harm.__class__.__name__ if harm else 'raw'}.png")
    finally:
        backend.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
