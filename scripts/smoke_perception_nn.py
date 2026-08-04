"""Neural-perception smoke test: load a real mmdet3d detector and run one forward pass.

Unlike ``smoke_perception.py`` (which only exercises the CUDA ops), this builds avstack's
``MMDetObjectDetector3D`` from a downloaded checkpoint and pushes a synthetic ``LidarData``
through its real inference path, proving the whole neural stack resolves and runs on the GPU.

Prereqs: the ``avsec`` conda env, a CUDA GPU, and checkpoints in place::

    ./scripts/fetch_models.sh

A synthetic random cloud yields 0 detections (no car-shaped structure); the point is that
the model loads and the forward pass completes. Real detections come from real CARLA/KITTI
point clouds. Select the model/dataset via CLI, e.g. ``--dataset carla-vehicle``.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pointpillars")
    ap.add_argument("--dataset", default="kitti")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-points", type=int, default=60000)
    args = ap.parse_args()

    import numpy as np
    import quaternion  # noqa: F401  (registers np.quaternion)
    import torch

    if not torch.cuda.is_available():
        print("NN PERCEPTION SMOKE: FAIL -> CUDA not available")
        return 1

    from avstack.calibration import LidarCalibration
    from avstack.geometry import GlobalOrigin3D, ReferenceFrame
    from avstack.geometry.datastructs import PointMatrix3D
    from avstack.modules.perception.object3d import MMDetObjectDetector3D
    from avstack.sensors import LidarData

    print(f"building detector ({args.model} / {args.dataset}) ...")
    try:
        det = MMDetObjectDetector3D(model=args.model, dataset=args.dataset, gpu=args.gpu)
    except FileNotFoundError as exc:
        print(f"NN PERCEPTION SMOKE: FAIL -> checkpoint missing ({exc}).")
        print("Run ./scripts/fetch_models.sh (and see models/README.md for CARLA weights).")
        return 1
    print(f"  loaded. classes={det.class_names} threshold={det.threshold}")

    ref = ReferenceFrame(x=np.array([0.0, 0.0, 1.73]), q=np.quaternion(1), reference=GlobalOrigin3D)
    calib = LidarCalibration(ref)
    rng = np.random.default_rng(0)
    pts = rng.uniform([-40, -40, -3, 0.0], [40, 40, 1, 1.0], size=(args.n_points, 4)).astype(np.float32)
    cloud = LidarData(0.0, 0, PointMatrix3D(pts, calib), calib, 0)

    print("running forward pass on a synthetic cloud ...")
    dets = det(cloud, frame=0)
    ok = hasattr(dets, "__len__")
    print(f"  -> returned {type(dets).__name__} with {len(dets)} detections")
    print("NN PERCEPTION SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
