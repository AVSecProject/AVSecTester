#!/usr/bin/env python
"""Warp + harmonize patch insertion, end-to-end, as a SEQUENCE through the standard viz pipeline.

Unlike ``carla_patch_demo.py`` (which paints a physical prop in the world and re-renders), this inserts
the patch as a **2D post-process on the clean render**: project the lead car's rear face -> perspective-
warp the patch onto it -> harmonize it to the scene. The same insert works for any backend; the only
CARLA-specific piece is projecting the target quad (``simulators.carla.lead_rear_quad``). It layers on
the existing pipeline as a view wrapper — ``detections_view(detect, base=composite_view(...))`` — so the
detector runs on the *composited* frame and the attack's effect is visible over the sequence.

    conda run -n avsec python scripts/patch_composite_demo.py --frames 16 \
        --texture tmp/patch_optim/phys_texture.png --harmonizer libcom

Omit --texture for the benign checkerboard. --harmonizer classic (default, zero-dep) | libcom (learned,
isolated env — see scripts/setup_libcom_env.sh). Needs a CARLA server on :2000 (GPU 2).
"""

import argparse
import sys
from pathlib import Path

import yaml
from avsectester.attacks.patch_composite import (
    ClassicHarmonizer,
    LibcomHarmonizer,
    PatchCompositor,
)
from avsectester.attacks.physical_patch import checkerboard_rgba, image_rgba
from avsectester.backend import AVStack
from avsectester.plane import Control
from avsectester.simulators import carla as carla_sim
from avsectester.simulators.carla import CarlaBackend, lead_rear_quad
from avsectester.simulators.viz import (
    composite_view,
    detections_view,
    filmstrip,
    record_run,
    save_gif,
    save_image,
)

# reuse the detector builder from the physical-patch demo (CARLA-trained 2D detector)
from carla_patch_demo import CruiseStack, build_detector  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tmp" / "patch_composite"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO / "configs" / "carla_patch_scenario.yaml"))
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--gap", type=float, default=None, help="override lead distance (m)")
    ap.add_argument("--texture", default=None, help="adversarial patch image (else benign checkerboard)")
    ap.add_argument("--tex", type=int, default=256)
    ap.add_argument("--harmonizer", choices=["classic", "libcom"], default="classic")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--no-detect", action="store_true", help="skip the detection overlay")
    args = ap.parse_args()

    scenario = yaml.safe_load(Path(args.config).read_text())
    if args.gap is not None:
        scenario.setdefault("lead", {})["gap"] = args.gap
    # The composite is a 2D insert on the CLEAN render, so drop any world-level physical patch.
    scenario.pop("patches", None)

    patch = (image_rgba(args.texture, args.tex) if args.texture
             else checkerboard_rgba(args.tex, squares=8))
    harmonizer = LibcomHarmonizer() if args.harmonizer == "libcom" else ClassicHarmonizer()
    compositor = PatchCompositor(harmonizer)
    kind = f"{'adversarial' if args.texture else 'benign checkerboard'} / {args.harmonizer} harmonizer"

    backend = CarlaBackend(scenario)
    base = composite_view(compositor, patch, lead_rear_quad(backend), base=carla_sim.camera_view)
    visualize = base if args.no_detect else detections_view(build_detector(args.gpu), base=base)

    print(f"[demo] warp+harmonize insert ({kind}), {args.frames}-frame sequence via record_run ...")
    try:
        trace = record_run(backend, CruiseStack(), args.frames, out_dir=OUT / "seq",
                           visualize=visualize, collect=True)
    finally:
        backend.close()

    if trace.frames:
        save_image(filmstrip(trace.frames, cols=4), OUT / "composite_filmstrip.png")
        save_gif(trace.frames, OUT / "composite_sequence.gif", fps=4)
        print(f"[output] {len(trace.frames)} frames -> {OUT}/seq/")
        print(f"[output] filmstrip -> {OUT}/composite_filmstrip.png")
        print(f"[output] gif       -> {OUT}/composite_sequence.gif")
    else:
        print("[warn] no camera frames captured (is a CarlaRgbCamera in the ego sensors?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
