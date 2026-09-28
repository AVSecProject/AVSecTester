#!/usr/bin/env python
"""Insert the image-attack objects into real nuScenes photos — the same world-anchored insertion as
``nurec_object_demo.py``, on a pinhole camera and across many scenes (day / night, Boston / Singapore).

For each chosen ``CAM_FRONT`` image the objects (STOP sign, person standee, person billboard) are
placed in the **ego frame** (x forward, y left, z up; nuScenes' ego origin is on the ground at the rear
axle) and rendered through the image's calibrated pinhole camera
(:class:`avsectester.simulators.camera_models.PinholeCamera`). A placement is used only if its image
footprint overlaps no annotated object (a crude occlusion check); several candidate spots are tried.

    python scripts/nuscenes_object_demo.py --nuscenes /workspace/hdd/datasets/nuscenes \
        --asset <cutouts>/person_001.png --n 8 --harmonizer libcom --eval

Writes ``tmp/nuscenes_objects/``: ``grid.png`` (rows = images; columns = clean + each object),
per-image PNGs, and with ``--eval`` ``perception_eval.json`` (COCO detector scores per object).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from avsectester.attacks.person_poster import billboard, load_cutout, standee
from avsectester.attacks.sign_spoof import RoadsideSign, sign_rgba
from avsectester.simulators.camera_models import PinholeCamera, make_pose, quat_to_matrix
from avsectester.simulators.patch_insertion import PatchCompositor, render_plane
from demo_common import COCO_PERSON, COCO_STOP_SIGN, build_coco_detector
from nurec_object_demo import make_harmonizer

REPO = Path(__file__).resolve().parents[1]
# candidate spots (x ahead, y lateral) per object, nearest-first; the first unoccluded one is used
SPOTS = {
    "stop": [(16.0, -4.5), (20.0, -5.0), (14.0, -4.0), (18.0, 4.5), (24.0, -5.5)],
    "standee": [(12.0, -3.2), (15.0, -3.5), (10.0, -3.0), (13.0, 3.2), (18.0, -3.8)],
    "billboard": [(20.0, -6.0), (24.0, -6.5), (17.0, -5.5), (20.0, 6.0), (28.0, -7.0)],
}
LABELS = {"stop": (COCO_STOP_SIGN, "stop sign"), "standee": (COCO_PERSON, "person"),
          "billboard": (COCO_PERSON, "person")}


def make_object(name: str, x: float, y: float, person: np.ndarray) -> RoadsideSign:
    if name == "stop":
        return RoadsideSign(x=x, y=y, face=sign_rgba(), width=0.75, yaw=0.15 if y < 0 else -0.15)
    if name == "standee":
        return standee(person, x, y, height=1.75, yaw=0.1 if y < 0 else -0.1)
    return billboard(person, x, y, width=1.4, yaw=0.2 if y < 0 else -0.2)


def footprint(frame, cam, cam_from_ego, obj):
    """Union mask of the object's planes in the image (no harmonization)."""
    m = np.zeros(frame.shape[:2], np.uint8)
    for corners, tex in obj.planes():
        m |= render_plane(frame, cam, cam_from_ego, corners, tex)[1]
    return m


def free_spot(name, frame, cam, cam_from_ego, boxes, person):
    """First candidate spot whose footprint is fully in the image and clear of every annotated box."""
    h, w = frame.shape[:2]
    for x, y in SPOTS[name]:
        obj = make_object(name, x, y, person)
        m = footprint(frame, cam, cam_from_ego, obj)
        ys, xs = np.where(m)
        if xs.size < 30 or xs.min() < 4 or ys.min() < 4 or xs.max() > w - 5 or ys.max() > h - 5:
            continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        if all(bx + bw < x0 or bx > x1 or by + bh < y0 or by > y1 for bx, by, bw, bh in boxes):
            return obj
    return None


def pick_images(coco: dict, n: int) -> list[dict]:
    """``CAM_FRONT`` images spread across both cities and by brightness proxy (the file's log id)."""
    front = [im for im in coco["images"] if "/CAM_FRONT/" in im["file_name"]]
    rng = np.random.default_rng(0)
    rng.shuffle(front)
    boston = [im for im in front if "-0400" in im["file_name"]]
    singapore = [im for im in front if "+0800" in im["file_name"]]
    out = []
    for a, b in zip(boston, singapore):
        out += [a, b]
        if len(out) >= 6 * n:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nuscenes", required=True)
    ap.add_argument("--ann", default="nuscenes_infos_val_mono3d.coco.json")
    ap.add_argument("--asset", required=True, help="person cut-out PNG")
    ap.add_argument("--n", type=int, default=8, help="images to show")
    ap.add_argument("--harmonizer", choices=["none", "classic", "chroma", "libcom"], default="libcom")
    ap.add_argument("--night", type=int, default=2, help="how many of the images should be dark (night)")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "tmp" / "nuscenes_objects"))
    args = ap.parse_args()

    import cv2

    root, out = Path(args.nuscenes), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    coco = json.loads((root / args.ann).read_text())
    boxes_of: dict = {}
    for a in coco["annotations"]:
        boxes_of.setdefault(a["image_id"], []).append(a["bbox"])
    person = load_cutout(args.asset)
    compositor = PatchCompositor(make_harmonizer(args.harmonizer, args.gpu))
    detectors = {k: build_coco_detector(args.gpu, threshold=0.05, labels={i: n}) for k, (i, n) in
                 {"stop sign": LABELS["stop"], "person": LABELS["standee"]}.items()} if args.eval else {}

    rows, evals, n_night, n_day = [], [], 0, 0
    for im in pick_images(coco, args.n):
        if len(rows) >= args.n:
            break
        frame = cv2.cvtColor(cv2.imread(str(root / im["file_name"])), cv2.COLOR_BGR2RGB)
        night = frame.mean() < 55
        if night and n_night >= args.night or not night and n_day >= args.n - args.night:
            continue
        cam = PinholeCamera(np.array(im["cam_intrinsic"]), im["width"], im["height"])
        q = im["cam2ego_rotation"]
        cam_from_ego = np.linalg.inv(make_pose(quat_to_matrix(*q), im["cam2ego_translation"]))
        objs = {name: free_spot(name, frame, cam, cam_from_ego, boxes_of.get(im["id"], []), person)
                for name in SPOTS}
        if any(o is None for o in objs.values()):
            continue  # need a free spot for every object to make a comparable row
        n_night += night
        n_day += not night
        row, rec = [frame], {"image": im["file_name"], "night": bool(night)}
        for name, obj in objs.items():
            att = compositor.apply_planes(frame, cam, cam_from_ego, obj.planes(), soften=0.5)
            row.append(att)
            if detectors:
                ys, xs = np.where(np.abs(att.astype(np.int16) - frame.astype(np.int16)).max(axis=2) > 12)
                box = (xs.min() - 4, ys.min() - 4, xs.max() + 4, ys.max() + 4)
                det = detectors[LABELS[name][1]]
                rec[name] = {"at": [obj.x, obj.y], "clean": _best(det(frame), box), "attacked": _best(det(att), box)}
        rows.append(row)
        evals.append(rec)
        stem = Path(im["file_name"]).stem
        cv2.imwrite(str(out / f"{stem}.png"), cv2.cvtColor(_row_image(row), cv2.COLOR_RGB2BGR))
        print(f"[nuscenes] {stem} ({'night' if night else 'day'})")

    if not rows:
        print("[warn] no image had a free spot for every object")
        return 1
    grid = np.vstack([_row_image(r) for r in rows])
    cv2.imwrite(str(out / "grid.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    if evals and detectors:
        (out / "perception_eval.json").write_text(json.dumps(evals, indent=1))
        for name in SPOTS:
            hits = sum(e[name]["attacked"] >= 0.5 for e in evals)
            fp = sum(e[name]["clean"] >= 0.5 for e in evals)
            print(f"[eval] {name}: '{LABELS[name][1]}' detected in {hits}/{len(evals)} images (clean: {fp})")
    print(f"[output] {out}")
    return 0


def _best(dets, box) -> float:
    scores = []
    for b, score, _ in dets:
        ix = max(0.0, min(b[2], box[2]) - max(b[0], box[0]))
        iy = max(0.0, min(b[3], box[3]) - max(b[1], box[1]))
        if ix * iy >= 0.5 * (b[2] - b[0]) * (b[3] - b[1]):
            scores.append(score)
    return round(max(scores, default=0.0), 3)


def _row_image(row: list, scale: float = 0.4) -> np.ndarray:
    import cv2

    tiles = []
    for img, label in zip(row, ["clean", "stop", "standee", "billboard"]):
        t = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.putText(t, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        tiles += [t, np.full((t.shape[0], 4, 3), 255, np.uint8)]
    r = np.hstack(tiles[:-1])
    return np.vstack([r, np.full((4, r.shape[1], 3), 255, np.uint8)])


if __name__ == "__main__":
    sys.exit(main())
