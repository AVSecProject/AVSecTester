#!/usr/bin/env python
"""Compare clean and inserted-object NuRec drives at a constant ego speed.

Roadside objects have explicit world coordinates. Vehicle objects attach to a stable
``--host`` from the matching ``--usdz`` metadata, using current 3D poses every frame.
Both modes perturb the camera observation supplied to the stack. Visibility uses
known cuboids and does not measure exact neural-rendered occlusion.

Use ``--mode roadside`` or ``--mode vehicle --host ID``. ``--eval`` enables auxiliary
COCO detection scores. Outputs include clean/attacked images and side-by-side GIFs.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from avsectester.simulators.nurec import NuRecBackend, NuRecRenderer
from avsectester.simulators.patch_insertion import PatchCompositor, frame_perturbation
from avsectester.simulators.viz import filmstrip, record_run, save_gif, save_image
from scripts.common.demo_common import (
    OBJECTS, CruiseStack, build_coco_detector, build_object, make_harmonizer,
    nurec_rear_insertion, nurec_source,
)

from scripts import REPO_ROOT as REPO

CAM = "camera_front_wide_120fov"


def side_by_side(a: np.ndarray, b: np.ndarray, scale: float = 0.5) -> np.ndarray:
    import cv2

    small = [cv2.resize(x, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for x in (a, b)]
    for img, label in zip(small, ("clean", "attacked")):
        cv2.putText(img, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return np.hstack([small[0], np.full((small[0].shape[0], 6, 3), 255, np.uint8), small[1]])


def diff_region(clean: np.ndarray, attacked: np.ndarray):
    """Pixels where the attacked frame differs from the clean one (the inserted object), or None."""
    diff = np.abs(clean.astype(np.int16) - attacked.astype(np.int16)).max(axis=2) > 12
    return np.where(diff) if diff.sum() >= 20 else None


def zoom(clean: np.ndarray, attacked: np.ndarray, pad: int = 60, out_h: int = 360):
    """Crop both frames around where they differ (the inserted object); None if identical."""
    import cv2

    region = diff_region(clean, attacked)
    if region is None:
        return None
    ys, xs = region
    h, w = clean.shape[:2]
    cx, cy = int(np.median(xs)), int(np.median(ys))
    half = max(xs.max() - xs.min(), ys.max() - ys.min(), 40) // 2 + pad
    x0, x1, y0, y1 = max(cx - half, 0), min(cx + half, w), max(cy - half, 0), min(cy + half, h)
    crops = [cv2.resize(im[y0:y1, x0:x1], None, fx=out_h / (y1 - y0), fy=out_h / (y1 - y0),
                        interpolation=cv2.INTER_CUBIC) for im in (clean, attacked)]
    return np.hstack([crops[0], np.full((out_h, 6, 3), 255, np.uint8), crops[1]])


def perception_eval(clean: list, attacked: list, detect) -> list[dict]:
    """Per frame, the detector's best score for a box on the inserted object (>= half its area inside
    the object's footprint), on the clean and the attacked frame — did perception pick the object up?"""

    def best(frame, box):
        scores = []
        for b, score, _ in detect(frame):
            ix = max(0.0, min(b[2], box[2]) - max(b[0], box[0]))
            iy = max(0.0, min(b[3], box[3]) - max(b[1], box[1]))
            if ix * iy >= 0.5 * (b[2] - b[0]) * (b[3] - b[1]):
                scores.append(score)
        return max(scores, default=0.0)

    rows = []
    for i, (c, a) in enumerate(zip(clean, attacked)):
        region = diff_region(c, a)
        if region is None:
            rows.append({"frame": i, "visible": False, "clean": 0.0, "attacked": 0.0})
            continue
        ys, xs = region
        box = (xs.min() - 4, ys.min() - 4, xs.max() + 4, ys.max() + 4)
        rows.append({"frame": i, "visible": True, "clean": round(best(c, box), 3),
                     "attacked": round(best(a, box), 3)})
    return rows


def plot_eval(results: dict, path: Path, label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3.2))
    for tag, rows in results.items():
        ax.plot([r["frame"] for r in rows], [r["attacked"] for r in rows], label=tag)
    first = next(iter(results.values()))
    ax.plot([r["frame"] for r in first], [r["clean"] for r in first], "k--", label="clean")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("frame (0.1 s)")
    ax.set_ylabel(f"'{label}' score")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def drive(endpoint: str, scene, frames: int, speed: float, out_dir: Path, perturb=None):
    renderer = NuRecRenderer(endpoint=endpoint, scene_id=scene.source["scene_id"], cameras=[CAM],
                             start_timestamp_us=scene.source["timestamp_us"])
    backend = NuRecBackend({"dt": 0.1, "ego0": {"speed": speed}}, renderer=renderer)
    try:
        # the perturbation needs the loaded scene's camera (model + rig extrinsic): load it up front
        renderer.load_scene(scene.source["scene_id"])
        p = perturb(backend) if perturb is not None else None
        return record_run(backend, CruiseStack(throttle=0.0), frames, out_dir=out_dir, perturb=p,
                          collect=True).frames
    finally:
        backend.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--speed", type=float, default=3.7, help="constant ego speed (m/s)")
    ap.add_argument("--scene", default=None, help="optional scene ID substring matching --usdz")
    ap.add_argument("--usdz", required=True, help="matching NuRec scene archive with actor metadata")
    ap.add_argument("--host", help="stable vehicle track ID, required for vehicle mode")
    ap.add_argument("--endpoint", default="127.0.0.1:50051")
    ap.add_argument("--object", choices=sorted(OBJECTS), default="stop")
    ap.add_argument("--asset", default=None, help="person cut-out PNG (standee / billboard)")
    ap.add_argument("--mode", nargs="+", choices=["roadside", "vehicle"], default=["roadside", "vehicle"])
    ap.add_argument("--harmonizer", nargs="+", choices=["none", "classic", "chroma", "libcom"],
                    default=["classic", "chroma", "libcom"])
    ap.add_argument("--x", type=float, default=None, help="roadside: absolute scene X coordinate (m)")
    ap.add_argument("--y", type=float, default=None, help="roadside: absolute scene Y coordinate (m)")
    ap.add_argument("--yaw", type=float, default=0.15, help="roadside: face turned toward the road (rad)")
    ap.add_argument("--size", type=float, default=None,
                    help="roadside size (m): sign/board width, standee height (default 0.9 / 1.4 / 1.75)")
    ap.add_argument("--mount", type=float, default=None, help="roadside bottom edge above ground (m)")
    ap.add_argument("--ground-z", type=float, default=0.0, help="scene ground height at the sign (m)")
    ap.add_argument("--soften", type=float, default=0.6, help="blur (px) to match the soft neural render")
    ap.add_argument("--eval", action="store_true",
                    help="score whether a COCO detector sees the inserted object (clean vs attacked)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None, help="output dir (default tmp/nurec_<object>)")
    args = ap.parse_args()
    if args.frames < 1:
        ap.error("--frames must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    out = Path(args.out or REPO / "tmp" / f"nurec_{args.object}")
    payload = build_object(args)
    dataset, scene = nurec_source(args.usdz, args.scene)
    insertions = {"roadside": payload.roadside}
    if "vehicle" in args.mode:
        if args.host is None:
            ap.error("--mode vehicle requires --host <recorded vehicle track ID>")
        insertions["vehicle"] = nurec_rear_insertion(payload, scene, args.host)
    label_id, label = OBJECTS[args.object]["label"]

    print(f"[demo] clean drive: {args.frames} frames at {args.speed} m/s")
    clean = drive(args.endpoint, scene, args.frames, args.speed, out / "clean")
    obj_detect = build_coco_detector(args.gpu, threshold=0.05, labels={label_id: label}) if args.eval else None
    evals: dict = {}

    for mode in args.mode:
        for hname in args.harmonizer:
            compositor = PatchCompositor(
                make_harmonizer(hname, args.gpu), soften=args.soften, harmonize=hname != "none",
            )

            def perturb(backend, compositor=compositor, insertion=insertions[mode]):
                insert = dataset.insertion_renderer(scene, backend, [insertion], compositor=compositor)
                return frame_perturbation(insert, camera=CAM)

            tag = f"{mode}_{hname}"
            print(f"[demo] attacked drive: {tag}")
            attacked = drive(args.endpoint, scene, args.frames, args.speed, out / tag / "seq",
                             perturb=perturb)
            pairs = [side_by_side(c, a) for c, a in zip(clean, attacked)]
            save_gif(pairs, out / tag / "side_by_side.gif", fps=10)
            save_image(filmstrip(attacked[:: max(1, len(attacked) // 8)][:8], cols=4), out / tag / "filmstrip.png")
            for i in sorted({len(clean) // 3, 2 * len(clean) // 3, len(clean) - 1}):
                z = zoom(clean[i], attacked[i])
                if z is not None:
                    save_image(z, out / tag / f"zoom_{i:04d}.png")
            if obj_detect is not None:
                rows = perception_eval(clean, attacked, obj_detect)
                evals.setdefault(mode, {})[tag] = rows
                vis = [r for r in rows if r["visible"]]
                hit = sum(r["attacked"] >= 0.5 for r in vis)
                print(f"[eval] {tag}: '{label}' detected (score>=0.5) in {hit}/{len(vis)} frames where the "
                      f"object is in view; clean false hits {sum(r['clean'] >= 0.5 for r in vis)}")
            print(f"[output] {out / tag}")
    if evals:
        import json

        (out / "perception_eval.json").write_text(json.dumps(evals, indent=1))
        for mode, results in evals.items():
            plot_eval(results, out / f"perception_{mode}.png", label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
