#!/usr/bin/env python
"""Complete closed-loop patch attack in CARLA: clean vs patched, driving impact + visualization.

The full end-to-end (not a view-only overlay): the patch is composited into the ego *camera
observation* (warp + harmonize), so a camera forward-collision stack acts on the patched frame. The
attack HIDES the parked lead car from the detector, so the ego that safely brakes in the clean run
fails to brake in the patched run — an object-hiding attack with a driving consequence.

Runs the identical scene twice (clean, then patched), diffs the two driving Traces with
:mod:`avsectester.metric` (the ``suppressed_stop`` verdict), and writes:
  * ``impact.png``            — ego speed + brake, clean vs patched (metric view)
  * ``clean_filmstrip.png`` / ``patched_filmstrip.png`` — detector overlay per frame (scene view)
  * ``clean.gif`` / ``patched.gif``

    conda run -n avsec python scripts/patch_driving_demo.py --frames 40 \
        --texture tmp/patch_optim/phys_texture.png --harmonizer libcom

--harmonizer classic (fast, in-process) | libcom (learned PCTNet, resident worker). Needs a CARLA
server on :2000 (GPU 2); detector on --gpu (default 1).
"""

import argparse
import logging
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
from avsectester.metric import impact, plot_impact
from avsectester.plane import Control
from avsectester.simulators import carla as carla_sim
from avsectester.simulators.carla import CarlaBackend, camera_patch_perturbation
from avsectester.simulators.viz import detections_view, record_run, save_sequence
from demo_common import build_detector, plausible_detector  # shared demo glue

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tmp" / "patch_driving"


class CameraForwardCollisionStack(AVStack):
    """Brake when the camera detector sees a close car ahead; otherwise hold throttle.

    Proximity proxy = the largest car box's area fraction of the frame (the parked lead grows as the
    ego approaches). Hiding the lead removes the box, so the ego never brakes — the attack's payload.
    """

    def __init__(self, detect, area_brake: float = 0.05, throttle: float = 0.55) -> None:
        self.detect = detect  # already plausibility-gated (see plausible_detector)
        self.area_brake = area_brake  # a plausible car this big (fraction of frame) is close -> brake
        self.throttle = throttle

    def reset(self, observation) -> None:
        pass

    def __call__(self, observation) -> Control:
        rgb = carla_sim.camera_view(observation)
        close = False
        if rgb is not None:
            h, w = rgb.shape[:2]
            for box, _score, _label in self.detect(rgb):
                if (box[2] - box[0]) * (box[3] - box[1]) >= self.area_brake * w * h:
                    close = True
        return Control(throttle=0.0, brake=1.0) if close else Control(throttle=self.throttle)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO / "configs" / "carla_patch_scenario.yaml"))
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--gap", type=float, default=8.0, help="lead distance (m) — room to approach")
    ap.add_argument("--texture", default=None, help="adversarial patch image (else benign checkerboard)")
    ap.add_argument("--tex", type=int, default=256)
    ap.add_argument("--harmonizer", choices=["classic", "libcom"], default="classic")
    ap.add_argument("--area-brake", type=float, default=0.03)
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    scenario = yaml.safe_load(Path(args.config).read_text())
    scenario.setdefault("lead", {})["gap"] = args.gap
    scenario.pop("patches", None)  # composite is a 2D insert on the clean render, not a world prop

    patch = (image_rgba(args.texture, args.tex) if args.texture
             else checkerboard_rgba(args.tex, squares=8))
    harmonizer = LibcomHarmonizer() if args.harmonizer == "libcom" else ClassicHarmonizer()
    compositor = PatchCompositor(harmonizer)
    detect = plausible_detector(build_detector(args.gpu))  # gate phantom boxes; feeds stack + overlay
    overlay = detections_view(detect, base=carla_sim.camera_view)

    def drive(patched: bool):
        backend = CarlaBackend(scenario)
        stack = CameraForwardCollisionStack(detect, area_brake=args.area_brake)
        perturb = camera_patch_perturbation(backend, compositor, patch) if patched else None
        try:
            return record_run(backend, stack, args.frames, out_dir=OUT / ("patched" if patched else "clean"),
                              visualize=overlay, perturb=perturb, collect=True)
        finally:
            backend.close()

    print(f"[demo] CLEAN run ({args.frames} frames, lead {args.gap} m) ...")
    clean = drive(patched=False)
    print(f"[demo] PATCHED run ({'adversarial' if args.texture else 'checkerboard'} / "
          f"{args.harmonizer} harmonizer) ...")
    attacked = drive(patched=True)
    if isinstance(harmonizer, LibcomHarmonizer):
        harmonizer.close()

    result = impact(clean, attacked)
    print("\n" + str(result) + "\n")
    plot_impact(clean, attacked, str(OUT / "impact.png"),
                title="Physical patch (object-hiding) — driving impact", result=result)
    for name, trace in (("clean", clean), ("patched", attacked)):
        save_sequence(trace.frames, OUT, name=name, cols=5, fps=6)
    print(f"[output] impact plot -> {OUT}/impact.png")
    print(f"[output] filmstrips  -> {OUT}/clean_filmstrip.png , {OUT}/patched_filmstrip.png")
    print(f"[output] gifs        -> {OUT}/clean.gif , {OUT}/patched.gif")
    return 0


if __name__ == "__main__":
    sys.exit(main())
