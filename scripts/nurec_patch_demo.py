#!/usr/bin/env python
"""Realistic patch insertion into a NuRec (AlpaSim) reconstructed trace — the AlpaSim-side end-to-end.

The same backend-agnostic warp+harmonize compositor used for CARLA, now bridged to NuRec by **planar
warping**: render a clean frame from the neural reconstruction, get the lead vehicle's rear-face quad
from a COCO detection box (:func:`avsectester.simulators.viz.detector_quad` — image-space, no depth,
since NuRec exposes neither actor boxes nor depth), homography-warp the patch onto it, and harmonize
it in. The patch is NEVER baked into the reconstruction — it is a 2-D composite on the render, exactly
as on CARLA (where the quad instead comes from the ground-truth 3-D box). The warp + harmonizer are
shared; only how the quad is obtained differs per backend.

Run in the AlpaSim driver env (Python 3.12), with an nre-ga server on :50051 (GPU 2):

    cd /workspace/nvme/qzzhang/alpasim
    PYTHONPATH=/workspace/nvme/qzzhang/AVSecProject/AVSecTester \
        uv run python scripts/nurec_patch_demo.py --frames 16 \
        --texture tmp/patch_optim/phys_texture.png --harmonizer classic

--harmonizer classic | libcom (learned PCTNet, in-process). Both run in this process, no separate env.
"""

import argparse
import logging
import sys
from pathlib import Path

from avsectester.attacks.patch_composite import (
    ClassicHarmonizer,
    PatchCompositor,
    PCTNetHarmonizer,
)
from avsectester.attacks.physical_patch import checkerboard_rgba, image_rgba
from avsectester.simulators.nurec import NuRecBackend, NuRecRenderer
from avsectester.simulators.viz import (
    camera_view,
    composite_view,
    detector_quad,
    record_run,
    save_sequence,
)
from demo_common import CruiseStack, build_coco_detector  # shared demo glue

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tmp" / "nurec_patch"
CAM = "camera_front_wide_120fov"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--scene", default="01d503d4", help="NuRec scene id substring")
    ap.add_argument("--endpoint", default="127.0.0.1:50051")
    ap.add_argument("--texture", default=None, help="adversarial patch image (else checkerboard)")
    ap.add_argument("--tex", type=int, default=256)
    ap.add_argument("--harmonizer", choices=["classic", "libcom"], default="classic")
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device for the COCO detector")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    patch = (image_rgba(args.texture, args.tex) if args.texture
             else checkerboard_rgba(args.tex, squares=8))
    harmonizer = PCTNetHarmonizer() if args.harmonizer == "libcom" else ClassicHarmonizer()
    compositor = PatchCompositor(harmonizer)

    renderer = NuRecRenderer(endpoint=args.endpoint, scene_id=args.scene, cameras=[CAM])
    backend = NuRecBackend({"dt": 0.1, "ego0": {"speed": 4.0}}, renderer=renderer)
    # planar warp on the lead vehicle's rear: quad approximated from a COCO detection (no depth)
    quad_of = detector_quad(build_coco_detector(args.gpu), base=camera_view)
    visualize = composite_view(compositor, patch, quad_of, base=camera_view)

    kind = f"{'adversarial' if args.texture else 'checkerboard'} / {args.harmonizer}"
    print(f"[demo] NuRec patch insert ({kind}), {args.frames}-frame sequence via record_run ...")
    try:
        trace = record_run(backend, CruiseStack(), args.frames, out_dir=OUT / "seq",
                           visualize=visualize, collect=True)
    finally:
        backend.close()

    out = save_sequence(trace.frames, OUT, name="nurec")
    if out:
        print(f"[output] {len(trace.frames)} frames -> {OUT}/seq/ ; filmstrip+gif -> {out[0]} , {out[1]}")
    else:
        print("[warn] no frames captured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
