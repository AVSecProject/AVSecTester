#!/usr/bin/env python
"""PHYSICALLY-RENDERED object-removal patch on the lead car — optimized, then validated in CARLA.

Pipeline (the acceptance test is the CARLA-RENDERED patch, not a digital paste):
  1. spawn the physical patch on the lead car and MEASURE its real image footprint (paint it solid,
     render, segment) — so the optimization is aligned to exactly where the patch renders;
  2. optimize the texture with white-box PGD (HideObject + RPN-objectness) aligned to that footprint;
  3. DEPLOY the optimized texture physically as an EMISSIVE patch (self-lit, so scene lighting does
     not dim it — this closes the digital->physical appearance gap);
  4. render and re-run the detector: report the CARLA-rendered detection confidence.

Run in the `avsec` conda env with a CARLA 0.9.15 server on :2000 (GPU 2):
    conda run -n avsec python scripts/optimize_patch_physical.py [--steps 250] [--gpu 1]
"""

import argparse
import queue
import sys
from pathlib import Path

import numpy as np
from avsectester.attacks.optim.attacks import PGD
from avsectester.attacks.optim.data import FrameSource
from avsectester.attacks.optim.geometry import homography_4pt
from avsectester.attacks.optim.interface import AdvSample, HideObject, TargetSpec
from avsectester.attacks.optim.perturbations import PatchPerturbation
from avsectester.attacks.optim.scorers import MMDetRPNObjectnessScorer
from avsectester.attacks.physical_patch import PhysicalPatch, to_carla_texture

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tmp" / "patch_optim"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--patch", type=int, default=192, help="optimized texture resolution")
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = f"cuda:{args.gpu}"

    import carla
    import torch
    from PIL import Image

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
            carla.Location(lead_tf.location.x - fwd.x * 6, lead_tf.location.y - fwd.y * 6,
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

        def render():
            for _ in range(3):
                world.tick()
            img = None
            while not q.empty():
                img = q.get()
            return np.frombuffer(img.raw_data, np.uint8).reshape(
                img.height, img.width, 4)[:, :, :3][:, :, ::-1].copy()

        clean = render()
        Image.fromarray(clean).save(OUT / "phys_clean.png")

        print("[1] detector + clean detection ...")
        import avstack.modules.perception.object2dfv  # noqa: F401
        from avstack.config import MODELS
        from mmdet.apis import inference_detector

        det = MODELS.build({"type": "MMDetObjectDetector2D", "model": "fasterrcnn",
                            "dataset": "carla-vehicle", "gpu": args.gpu, "threshold": 0.3})
        model = det.model

        def detect_in_box(img, box):
            inst = inference_detector(model, img[:, :, ::-1]).pred_instances
            sc = inst.scores.detach().cpu().numpy()
            bx = inst.bboxes.detach().cpu().numpy()
            best = 0.0
            for b, c in zip(bx, sc):
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
                    best = max(best, float(c))
            return best

        inst = inference_detector(model, clean[:, :, ::-1]).pred_instances
        o = inst.scores.argsort(descending=True)
        box = tuple(float(v) for v in inst.bboxes[o[0]].detach().cpu().numpy())
        clean_score = float(inst.scores[o[0]])
        print(f"    clean lead-car detection: {clean_score:.3f} box={tuple(round(v) for v in box)}")

        # spawn the physical patch (emissive), then measure its rendered footprint via a solid paint
        patch = PhysicalPatch(texture={"pattern": "checkerboard", "size": 64}, emissive=True)
        actors += patch.apply(world, lead)
        name = patch.painted_objects[0]

        def paint(rgba):
            tex = to_carla_texture(rgba)
            world.apply_color_texture_to_object(name, carla.MaterialParameter.Diffuse, tex)
            world.apply_color_texture_to_object(name, carla.MaterialParameter.Emissive, tex)

        green = np.zeros((64, 64, 4), np.uint8)
        green[..., 1] = 255
        green[..., 3] = 255
        paint(green)
        seg = render()
        m = (seg[:, :, 1] > 120) & (seg[:, :, 0] < 100) & (seg[:, :, 2] < 100)
        ys, xs = np.where(m)
        footprint = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        print(f"[2] physical patch footprint {footprint} px ({int(m.sum())} px)")

        print(f"[3] white-box PGD aligned to the footprint: {args.patch}px, {args.steps} steps ...")
        x01 = torch.from_numpy(clean).permute(2, 0, 1).float().div(255).to(dev)
        dst = np.array([[footprint[0], footprint[1]], [footprint[2], footprint[1]],
                        [footprint[2], footprint[3]], [footprint[0], footprint[3]]], np.float64)
        src = np.array([[0, 0], [args.patch, 0], [args.patch, args.patch], [0, args.patch]], np.float64)
        placement = torch.from_numpy(homography_4pt(src, dst)).float()
        sample = AdvSample(x=x01, target=TargetSpec(box=box, label="car"),
                           context={"placement": placement})
        pert = PatchPerturbation(size=(args.patch, args.patch), device=dev, init="random")
        pgd = PGD(steps=args.steps, step_size=6 / 255, batch=4, tv_weight=0.02, verbose=True,
                  log_every=50)
        result = pgd.run(FrameSource([sample]), pert, MMDetRPNObjectnessScorer(model), HideObject())

        print("[4] DEPLOY physically (emissive) + render + detect ...")
        paint(result.export)
        phys = render()
        Image.fromarray(phys).save(OUT / "phys_patched.png")
        Image.fromarray(result.export).save(OUT / "phys_texture.png")
        phys_score = detect_in_box(phys, box)

        print("\n=== PHYSICALLY-RENDERED OBJECT-REMOVAL RESULT ===")
        print(f"  clean detection:                  {clean_score:.3f}")
        print(f"  digital surrogate (RPN objness):  {float(result.history[0]):.3f} -> {result.final_score:.3f}")
        print(f"  CARLA-RENDERED patched detection: {phys_score:.3f}")
        print(f"  => {'REMOVED' if phys_score < 0.3 else 'reduced'} "
              f"(Δ={clean_score - phys_score:+.3f}; detector threshold 0.3)")
        print(f"[output] {OUT}/ (phys_clean, phys_patched, phys_texture)")
        return 0
    finally:
        for a in reversed(actors):
            try:
                a.destroy()
            except Exception:  # noqa: BLE001, S110 - best-effort teardown
                pass
        s.synchronous_mode = False
        s.fixed_delta_seconds = 0.0
        world.apply_settings(s)


if __name__ == "__main__":
    sys.exit(main())
