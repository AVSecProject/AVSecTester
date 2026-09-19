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
from avsectester.backend import AVStack
from avsectester.plane import Control
from avsectester.simulators import carla as carla_view  # CarlaBackend + CARLA view adapters
from avsectester.simulators.carla import CarlaBackend
from avsectester.simulators.viz import (
    detections_view,
    filmstrip,
    record_run,
    save_gif,
    save_image,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tmp" / "carla_patch"


class CruiseStack(AVStack):
    """Hold a gentle constant throttle so the ego rolls toward the lead over the sequence."""

    def __init__(self, throttle: float = 0.25) -> None:
        self.throttle = throttle

    def __call__(self, obs) -> Control:
        return Control(throttle=self.throttle)


def build_detector(gpu: int):
    """Return ``detect(rgb) -> [(xyxy, score, 'car')]`` using the CARLA-trained 2D detector."""
    import avstack.modules.perception.object2dfv  # noqa: F401
    from avstack.config import MODELS
    from mmdet.apis import inference_detector

    det = MODELS.build({"type": "MMDetObjectDetector2D", "model": "fasterrcnn",
                        "dataset": "carla-vehicle", "gpu": gpu, "threshold": 0.3})

    def detect(rgb):
        h, w = rgb.shape[:2]
        inst = inference_detector(det.model, rgb[:, :, ::-1]).pred_instances
        boxes = inst.bboxes.detach().cpu().numpy()
        scores = inst.scores.detach().cpu().numpy()
        best = None  # the single best plausible lead-car box (drop near-full-frame false positives)
        for b, sc in zip(boxes, scores):
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area < 0.6 * w * h and (best is None or sc > best[1]):
                best = (b, float(sc), "car")
        return [best] if best else []

    return detect


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

    if trace.frames:
        save_image(filmstrip(trace.frames, cols=4), OUT / "patch_filmstrip.png")
        save_gif(trace.frames, OUT / "patch_sequence.gif", fps=4)
        print(f"[output] {len(trace.frames)} frames -> {OUT}/seq/")
        print(f"[output] filmstrip -> {OUT}/patch_filmstrip.png")
        print(f"[output] gif       -> {OUT}/patch_sequence.gif")
    else:
        print("[warn] no camera frames captured (is a CarlaRgbCamera in the ego sensors?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
