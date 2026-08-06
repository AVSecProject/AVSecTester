"""Evaluate avstack CARLA 2D camera detectors on REAL nuCarla images.

nuCarla (third_party/nuCarla; data at ./data/nuCarla) is a nuScenes-format CARLA dataset of
real 1600x900 camera frames containing vehicles -- the "good CARLA traces" needed to check
whether the fetched camera models produce *reasonable outputs*, not just that they load.

For each model this samples CAM_FRONT images across towns, runs the detector, and reports the
detection rate + score stats; with --save-montage it writes an annotated montage of the
top-detection frames for visual confirmation.

    conda activate avsec
    python scripts/eval_camera_nucarla.py                    # all camera models, 30 frames
    python scripts/eval_camera_nucarla.py --n 60 --save-montage results/nucarla_eval

Note: nuCarla is an ego 6-camera rig. The carla-vehicle/joint models match that viewpoint;
the carla-infrastructure models were trained on an elevated/roadside view, so running them on
ego frames is off-distribution (they still detect vehicles, with a few more false positives).
nuCarla's LiDAR files are dummy placeholders -- LiDAR models cannot be tested here.
"""
from __future__ import annotations

import argparse
import glob
import random
import sys
from pathlib import Path

DATA = "data/nuCarla"
# (model, dataset) pairs for the avstack MMDetObjectDetector2D CARLA checkpoints
CAM_MODELS = [
    ("fasterrcnn", "carla-vehicle"),
    ("cascadercnn", "carla-vehicle"),
    ("fasterrcnn", "carla-joint"),
    ("fasterrcnn", "carla-infrastructure"),
    ("cascadercnn", "carla-infrastructure"),
]


def _front_images(n: int, seed: int) -> list[str]:
    imgs: list[str] = []
    for d in sorted(glob.glob(f"{DATA}/Town*/samples/CAM_FRONT")):
        imgs += glob.glob(f"{d}/*.jpg")
    if not imgs:
        raise FileNotFoundError(
            f"no CAM_FRONT images under {DATA}/Town*/samples/ -- is ./data/nuCarla linked?"
        )
    random.Random(seed).shuffle(imgs)
    return imgs[:n]


def _bgr(path: str):
    import numpy as np
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1].copy()


def _montage(frames: list, path: Path, cols: int = 3, tile=(533, 300)) -> None:
    from PIL import Image

    if not frames:
        return
    w, h = tile
    rows = (len(frames) + cols - 1) // cols
    m = Image.new("RGB", (w * cols, h * rows), (20, 20, 20))
    for i, im in enumerate(frames):
        m.paste(im.resize(tile), ((i % cols) * w, (i // cols) * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    m.save(path)


def _eval_model(model: str, dataset: str, images: list[str], montage_dir: Path | None):
    import numpy as np
    from avstack.modules.perception.object2dfv import MMDetObjectDetector2D
    from PIL import Image, ImageDraw

    det = MMDetObjectDetector2D(model=model, dataset=dataset, gpu=0)
    thr = det.threshold
    rows = []
    for p in images:
        pi = det.inference_detector(det.model, _bgr(p)).pred_instances
        sc = pi.scores.cpu().numpy()
        keep = sc >= thr
        rows.append((int(keep.sum()), p, pi.bboxes.cpu().numpy()[keep], sc[keep]))
    fired = sum(1 for r in rows if r[0] > 0)
    total = sum(r[0] for r in rows)
    scores = np.concatenate([r[3] for r in rows if len(r[3])]) if total else np.array([0.0])
    stat = {
        "model": model, "dataset": dataset, "thr": thr, "frames": len(images),
        "frames_with_dets": fired, "total": total,
        "mean_score": round(float(scores.mean()), 3), "max_score": round(float(scores.max()), 3),
    }
    if montage_dir is not None:
        rows.sort(key=lambda r: -r[0])
        tiles = []
        for _, p, boxes, scs in rows[:6]:
            im = Image.open(p).convert("RGB")
            d = ImageDraw.Draw(im)
            for b, s in zip(boxes, scs):
                d.rectangle(list(b), outline=(0, 255, 0), width=4)
                d.text((b[0], max(0, b[1] - 16)), f"{s:.2f}", fill=(0, 255, 0))
            tiles.append(im)
        _montage(tiles, montage_dir / f"{model}_{dataset}.png")
    return stat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of CAM_FRONT frames to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-montage", metavar="DIR", default=None,
                    help="write annotated top-detection montages to DIR")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("EVAL: FAIL -> CUDA not available")
        return 1

    images = _front_images(args.n, args.seed)
    montage_dir = Path(args.save_montage) if args.save_montage else None
    print(f"evaluating {len(CAM_MODELS)} camera models on {len(images)} real nuCarla CAM_FRONT frames\n")

    stats = []
    for model, dataset in CAM_MODELS:
        s = _eval_model(model, dataset, images, montage_dir)
        stats.append(s)
        print(f"[{model:12s} {dataset:22s}] thr={s['thr']}  "
              f"frames_with_dets={s['frames_with_dets']}/{s['frames']}  "
              f"total={s['total']}  mean_score={s['mean_score']}  max_score={s['max_score']}")

    if montage_dir:
        print(f"\nannotated montages -> {montage_dir}/")
    print("\nNote: carla-infrastructure models run off their trained (elevated) viewpoint here; "
          "carla-vehicle/joint match nuCarla's ego rig. LiDAR models are NOT testable on "
          "nuCarla (dummy placeholder clouds).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
