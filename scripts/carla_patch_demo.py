#!/usr/bin/env python
"""Physical-patch CARLA scenario as a SEQUENCE — driven and visualized through the standard pipeline.

Drives the ego (a cruise stack) in ``configs/carla_patch_scenario.yaml`` with a physical patch on the
lead car, recording a sequence through :func:`avsectester.simulators.viz.record_run` (the same
pipeline every simulator uses). Each frame is overlaid with the detector's output (``detections_view``)
so the attack's effect is visible over the sequence, then assembled into a filmstrip + GIF.

    conda run -n avsec python scripts/carla_patch_demo.py --frames 16 --gap 6 \
        --texture tmp/patch_optim/phys_texture.png       # the PGD-optimized adversarial patch

Omit --texture for the benign checkerboard patch. Needs a CARLA 0.9.15 server on :2000 (GPU 2);
launch it with ``-quality-level=Epic`` for the most realistic rendering (the patch is a matte,
scene-lit surface by default).
"""

import argparse
import sys
from pathlib import Path

import yaml
from avsectester.simulators import carla as carla_view  # CarlaBackend + CARLA view adapters
from avsectester.simulators.carla import CarlaBackend
from avsectester.simulators.viz import detections_view, record_run, save_sequence
from demo_common import CruiseStack, build_detector  # shared demo glue

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tmp" / "carla_patch"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO / "configs" / "carla_patch_scenario.yaml"))
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--gap", type=float, default=None, help="override lead distance (m)")
    ap.add_argument("--texture", default=None, help="adversarial patch image (else benign checkerboard)")
    ap.add_argument("--tex", type=int, default=192)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--no-detect", action="store_true", help="skip the detection overlay")
    args = ap.parse_args()

    scenario = yaml.safe_load(Path(args.config).read_text())
    if args.gap is not None:
        scenario.setdefault("lead", {})["gap"] = args.gap
    patches = scenario.get("patches")
    kind = "benign checkerboard"
    if args.texture and patches:
        for p in patches:  # deploy the optimized adversarial texture as an emissive patch
            p["texture"] = {"image": str(Path(args.texture).resolve()), "size": args.tex}
            p["emissive"] = True
        kind = "adversarial (PGD-optimized)"

    visualize = (
        carla_view.camera_view if args.no_detect
        else detections_view(build_detector(args.gpu), base=carla_view.camera_view)
    )

    print(f"[demo] {kind} physical patch, {args.frames}-frame sequence via record_run ...")
    backend = CarlaBackend(scenario, patches=patches)
    try:
        trace = record_run(backend, CruiseStack(), args.frames, out_dir=OUT / "seq",
                           visualize=visualize, collect=True)
    finally:
        backend.close()

    out = save_sequence(trace.frames, OUT, name="patch")
    if out:
        print(f"[output] {len(trace.frames)} frames -> {OUT}/seq/ ; filmstrip+gif -> {out[0]} , {out[1]}")
    else:
        print("[warn] no camera frames captured (is a CarlaRgbCamera in the ego sensors?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
