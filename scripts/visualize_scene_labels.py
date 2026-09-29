"""Visualize ground-truth *scene labels* on real frames — reusing ``avsectester.simulators.viz``.

Overlays a dataset's :class:`SceneGT` (from :class:`NuRecDataset` / :class:`NuScenesDataset`) on the real
camera frame as projected **3-D bounding boxes** (the right representation for a 3-D scene; edges are
subdivided so they render correctly curved under the nuRec fisheye), using only the shared viz primitives:

* nuRec: ``draw_boxes3d(frame, scene, target=...)`` on the scene's rendered ``.mp4`` frame, plus a
  :func:`filmstrip` across several frames;
* nuScenes: the full backend path — ``labels_view(...)`` wrapping ``camera_view`` over a
  ``RecordedFrameBackend``, exactly as a run would visualize a frame.

Writes PNGs under ``tmp/compare/``. Run: ``python scripts/visualize_scene_labels.py [--nuscenes]``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from avsectester.scenarios.datasets.nurec import NuRecDataset
from avsectester.scenarios.requirements import PHYSICAL_PATCH_HIDE_VEHICLE as REQ
from avsectester.simulators.viz import draw_boxes3d, filmstrip, labels_view, save_image

OUT = Path("tmp/compare")
NUREC = Path("/workspace/hdd/datasets/PhysicalAI-Autonomous-Vehicles-NuRec/sample_set/26.01_release")


def _mp4_frame(mp4: str, idx: int):
    import cv2
    cap = cv2.VideoCapture(mp4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    cap.release()
    return bgr[:, :, ::-1].copy() if ok else None


def nurec_examples(uuid: str = "023b7fcc-671c-40e3-9bd2-c66b0b073fbc") -> None:
    usdz = str(NUREC / uuid / f"{uuid}.usdz")
    mp4 = str(NUREC / uuid / "camera_front_wide_120fov.mp4")
    strip, target_saved = [], False
    for k in range(21):
        scene = next(NuRecDataset([usdz], keyframe=k / 20).scenes())
        frame = _mp4_frame(mp4, scene.source["frame_index"])
        if frame is None:
            continue
        match = REQ.match(scene)
        # projected 3-D bounding boxes (near vehicles only, to keep the frame legible); the requirement's
        # target (if any) is highlighted and always drawn regardless of distance
        overlaid = draw_boxes3d(frame, scene, target=match.target if match else None, max_distance=40.0)
        strip.append(overlaid)
        if match and not target_saved:  # first qualifying frame -> the single-frame example
            save_image(overlaid, OUT / "labels_nurec_target.png")
            print("saved", OUT / "labels_nurec_target.png",
                  f"(target vehicle d={match.target.distance:.1f}m)")
            target_saved = True
    save_image(filmstrip(strip[::3][:8], cols=4), OUT / "labels_nurec_filmstrip.png")
    print("saved", OUT / "labels_nurec_filmstrip.png", f"({len(strip)} frames labeled)")


def nuscenes_example() -> None:
    from avsectester.scenarios.datasets.nuscenes import NuScenesDataset, RecordedFrameBackend
    ds = NuScenesDataset("/workspace/hdd/datasets/nuscenes", max_samples=60)
    for scene in ds.scenes():
        match = REQ.match(scene)
        if match is None:
            continue
        backend = RecordedFrameBackend(scene.source["image_path"])
        # the standard view path: labels_view wraps camera_view, overlaying this frame's GT
        view = labels_view(lambda _obs, s=scene: s, camera="front", target=match.target, max_distance=40.0)
        save_image(view(backend.reset()), OUT / "labels_nuscenes_target.png")
        print("saved", OUT / "labels_nuscenes_target.png",
              f"(target vehicle d={match.target.distance:.1f}m)")
        return
    print("no qualifying nuScenes frame in the sampled set")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuscenes", action="store_true", help="also render a nuScenes example (slow devkit load)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    nurec_examples()
    if args.nuscenes:
        nuscenes_example()
