#!/usr/bin/env python
"""Optimize a physical-patch OBJECT-REMOVAL attack with PGD (white-box, digital surrogate).

Captures a lead-car frame in CARLA, loads the CARLA-trained 2D detector (FasterRCNN carla-vehicle),
and runs PGD (HideObject objective + PatchPerturbation threat model + white-box RPN-objectness scorer)
to optimize a patch on the lead car's rear that suppresses its detection. Validates against the
detector's actual final confidence (clean vs patched) and saves the texture for CARLA transfer via
PhysicalPatch(texture={"image": ...}).

Run in the `avsec` conda env with a CARLA 0.9.15 server on :2000 (GPU 2):
    conda run -n avsec python scripts/optimize_patch.py [--steps 300] [--gpu 1]
"""

import argparse
import queue
import sys
from pathlib import Path

import numpy as np
import torch
from avsectester.attacks.optim.attacks import PGD
from avsectester.attacks.optim.data import FrameSource
from avsectester.attacks.optim.interface import AdvSample, HideObject, TargetSpec
from avsectester.attacks.optim.perturbations import PatchPerturbation, rear_panel_homography
from avsectester.attacks.optim.scorers import MMDetRPNObjectnessScorer

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tmp" / "patch_optim"


def capture_lead_frame(gap=6.0):
    """Spawn ego + a lead car `gap` m ahead, return the ego RGB camera frame (H, W, 3) uint8."""
    import carla

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(20.0)
    world = client.get_world()
    bl = world.get_blueprint_library()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 0.05
    world.apply_settings(s)
    actors = []
    try:
        sps = world.get_map().get_spawn_points()
        lead_tf = sps[0]
        fwd = lead_tf.get_forward_vector()
        ego_tf = carla.Transform(
            carla.Location(lead_tf.location.x - fwd.x * gap, lead_tf.location.y - fwd.y * gap,
                           lead_tf.location.z), lead_tf.rotation)
        lead = world.spawn_actor(bl.filter("vehicle.tesla.model3")[0], lead_tf)
        actors.append(lead)
        ego = world.spawn_actor(bl.filter("vehicle.audi.tt")[0], ego_tf)
        actors.append(ego)
        cam_bp = bl.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "800")
        cam_bp.set_attribute("image_size_y", "600")
        cam = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=1.0, z=1.6)), attach_to=ego)
        actors.append(cam)
        q = queue.Queue()
        cam.listen(q.put)
        for _ in range(8):
            world.tick()
        img = None
        while not q.empty():
            img = q.get()
        return np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3][:, :, ::-1].copy()
    finally:
        for a in reversed(actors):
            try:
                a.destroy()
            except Exception:  # noqa: BLE001, S110 - best-effort actor teardown
                pass
        s.synchronous_mode = False
        s.fixed_delta_seconds = 0.0
        world.apply_settings(s)


def detect_car_in_box(model, img_rgb, box):
    """Max detection confidence whose box center lies inside `box` (the lead-car region)."""
    from mmdet.apis import inference_detector

    inst = inference_detector(model, img_rgb[:, :, ::-1]).pred_instances
    scores = inst.scores.detach().cpu().numpy()
    boxes = inst.bboxes.detach().cpu().numpy()
    best = 0.0
    for b, sc in zip(boxes, scores):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
            best = max(best, float(sc))
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--step-size", type=float, default=6 / 255)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patch", type=int, default=128, help="patch size (square, px)")
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}"

    from PIL import Image

    print("[1] capturing a lead-car frame from CARLA ...")
    frame = capture_lead_frame()
    Image.fromarray(frame).save(OUT / "clean_frame.png")

    print("[2] loading the CARLA-trained 2D detector (FasterRCNN carla-vehicle) ...")
    import avstack.modules.perception.object2dfv  # noqa: F401 (registers the detector)
    from avstack.config import MODELS

    det = MODELS.build({"type": "MMDetObjectDetector2D", "model": "fasterrcnn",
                        "dataset": "carla-vehicle", "gpu": args.gpu, "threshold": 0.3})
    model = det.model

    from mmdet.apis import inference_detector

    inst = inference_detector(model, frame[:, :, ::-1]).pred_instances
    order = inst.scores.argsort(descending=True)
    box = tuple(float(v) for v in inst.bboxes[order[0]].detach().cpu().numpy())
    clean_score = float(inst.scores[order[0]])
    print(f"    lead car detected @ {clean_score:.3f}  box={tuple(round(v) for v in box)}")

    print(f"[3] PGD object-removal: patch {args.patch}px, {args.steps} steps, batch {args.batch} ...")
    x01 = torch.from_numpy(frame).permute(2, 0, 1).float().div(255).to(dev)
    placement = torch.from_numpy(rear_panel_homography(box, (args.patch, args.patch))).float()
    sample = AdvSample(x=x01, target=TargetSpec(box=box, label="car"),
                       context={"placement": placement})
    data = FrameSource([sample])
    pert = PatchPerturbation(size=(args.patch, args.patch), device=dev, init="random")
    scorer = MMDetRPNObjectnessScorer(model)
    pgd = PGD(steps=args.steps, step_size=args.step_size, batch=args.batch, tv_weight=0.02,
              verbose=True)
    result = pgd.run(data, pert, scorer, HideObject())

    print("[4] validating against the detector's final confidence ...")
    # paste the optimized patch (no jitter) and measure the real detection score
    with torch.no_grad():
        patched = pert.apply(AdvSample(x=x01, target=sample.target, context={"placement": placement}),
                             result.delta.to(dev))
    patched_np = (patched.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(patched_np).save(OUT / "patched_frame.png")
    Image.fromarray(result.export).save(OUT / "patch_texture.png")
    patched_score = detect_car_in_box(model, patched_np, box)

    print("\n=== OBJECT-REMOVAL RESULT (digital surrogate) ===")
    print(f"  clean lead-car detection:   {clean_score:.3f}")
    print(f"  patched lead-car detection: {patched_score:.3f}")
    print(f"  RPN objectness in box:      {float(result.history[0]):.3f} -> {result.final_score:.3f}")
    drop = clean_score - patched_score
    print(f"  => {'REMOVED' if patched_score < 0.3 else 'reduced' if drop > 0.05 else 'NO effect'} "
          f"(Δ={drop:+.3f}; detector threshold 0.3)")
    print(f"[output] {OUT}/  (clean_frame, patched_frame, patch_texture)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
