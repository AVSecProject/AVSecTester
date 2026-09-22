#!/usr/bin/env python
"""Realistic patch insertion into a NuRec (AlpaSim) reconstructed trace — the AlpaSim-side end-to-end.

The same backend-agnostic warp+harmonize compositor used for CARLA, now bridged to NuRec: render a
clean frame from the neural reconstruction, project a virtual panel planted ahead of the ego through
the real f-theta camera model, warp the patch onto it, and harmonize it into the reconstructed
imagery. The patch is NEVER baked into the reconstruction (the render API cannot insert objects) — it
is a 2-D composite on the render, exactly as on CARLA. Only the projection is simulator-specific
(:func:`avsectester.simulators.nurec.nurec_panel_quad`); the warp + harmonizer are shared.

Run in the AlpaSim driver env (Python 3.12), with an nre-ga server on :50051 (GPU 2):

    cd /workspace/nvme/qzzhang/alpasim
    PYTHONPATH=/workspace/nvme/qzzhang/AVSecProject/AVSecTester \
        uv run python scripts/nurec_patch_demo.py --frames 16 \
        --texture tmp/patch_optim/phys_texture.png --harmonizer classic

--harmonizer classic (in-process) | libcom (learned PCTNet, resident worker in the libcom env).
"""

import argparse
import logging
import sys
from pathlib import Path

from avsectester.attacks.patch_composite import (
    ClassicHarmonizer,
    LibcomHarmonizer,
    PatchCompositor,
)
from avsectester.attacks.physical_patch import checkerboard_rgba, image_rgba
from avsectester.backend import AVStack
from avsectester.plane import Control
from avsectester.simulators.nurec import NuRecBackend, NuRecRenderer, nurec_panel_quad
from avsectester.simulators.viz import (
    camera_view,
    composite_view,
    filmstrip,
    record_run,
    save_gif,
    save_image,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tmp" / "nurec_patch"
CAM = "camera_front_wide_120fov"


class CruiseStack(AVStack):
    """Hold a gentle throttle so the ego rolls toward the planted panel over the sequence."""

    def __init__(self, throttle: float = 0.4) -> None:
        self.throttle = throttle

    def reset(self, observation) -> None:
        pass

    def __call__(self, observation) -> Control:
        return Control(throttle=self.throttle)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--scene", default="01d503d4", help="NuRec scene id substring")
    ap.add_argument("--endpoint", default="127.0.0.1:50051")
    ap.add_argument("--texture", default=None, help="adversarial patch image (else checkerboard)")
    ap.add_argument("--tex", type=int, default=256)
    ap.add_argument("--harmonizer", choices=["classic", "libcom"], default="classic")
    ap.add_argument("--ahead", type=float, default=12.0, help="panel distance ahead of the ego (m)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    patch = (image_rgba(args.texture, args.tex) if args.texture
             else checkerboard_rgba(args.tex, squares=8))
    harmonizer = LibcomHarmonizer() if args.harmonizer == "libcom" else ClassicHarmonizer()
    compositor = PatchCompositor(harmonizer)

    renderer = NuRecRenderer(endpoint=args.endpoint, scene_id=args.scene, cameras=[CAM])
    backend = NuRecBackend({"dt": 0.1, "ego0": {"speed": 4.0}}, renderer=renderer)
    visualize = composite_view(compositor, patch, nurec_panel_quad(renderer, ahead=args.ahead),
                               base=camera_view)

    kind = f"{'adversarial' if args.texture else 'checkerboard'} / {args.harmonizer}"
    print(f"[demo] NuRec patch insert ({kind}), {args.frames}-frame sequence via record_run ...")
    try:
        trace = record_run(backend, CruiseStack(), args.frames, out_dir=OUT / "seq",
                           visualize=visualize, collect=True)
    finally:
        backend.close()
        if isinstance(harmonizer, LibcomHarmonizer):
            harmonizer.close()

    if trace.frames:
        save_image(filmstrip(trace.frames, cols=4), OUT / "nurec_filmstrip.png")
        save_gif(trace.frames, OUT / "nurec_sequence.gif", fps=4)
        print(f"[output] {len(trace.frames)} frames -> {OUT}/seq/")
        print(f"[output] filmstrip -> {OUT}/nurec_filmstrip.png")
        print(f"[output] gif       -> {OUT}/nurec_sequence.gif")
    else:
        print("[warn] no frames captured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
