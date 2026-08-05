"""Verify every avstack CARLA-trained model loads and runs a forward pass.

Builds each avstack MM detector from its downloaded checkpoint (config + weights onto the
GPU) and runs one inference, proving the checkpoint/config resolve and the neural stack
executes. LiDAR models go through avstack's full path on a synthetic cloud; 2D models run
the loaded detector's real inference path on a synthetic image.

A synthetic input yields few/zero detections (no real structure) -- the check is that the
model *loads and the forward pass completes*, not that it detects. Real detections come from
real CARLA point clouds/images (the carla-vehicle LiDAR model is separately validated live).

Prereqs: the ``avsec`` conda env, a CUDA GPU, and ``./scripts/fetch_models.sh`` already run.

    python scripts/verify_models.py            # verify all
    python scripts/verify_models.py --lidar    # LiDAR only
    python scripts/verify_models.py --cam      # 2D camera only
"""
from __future__ import annotations

import argparse
import sys
import traceback

# (kind, model, dataset)
LIDAR_MODELS = [
    ("lidar", "pointpillars", "carla-vehicle"),
    ("lidar", "pointpillars", "carla-infrastructure"),
]
CAM_MODELS = [
    ("cam", "fasterrcnn", "carla-vehicle"),
    ("cam", "fasterrcnn", "carla-infrastructure"),
    ("cam", "fasterrcnn", "carla-joint"),
    ("cam", "cascadercnn", "carla-vehicle"),
    ("cam", "cascadercnn", "carla-infrastructure"),
]


def _synthetic_cloud(n_points: int = 60000):
    import numpy as np
    import quaternion  # noqa: F401  (registers np.quaternion)
    from avstack.calibration import LidarCalibration
    from avstack.geometry import GlobalOrigin3D, ReferenceFrame
    from avstack.geometry.datastructs import PointMatrix3D
    from avstack.sensors import LidarData

    ref = ReferenceFrame(x=np.array([0.0, 0.0, 1.73]), q=np.quaternion(1), reference=GlobalOrigin3D)
    calib = LidarCalibration(ref)
    rng = np.random.default_rng(0)
    pts = rng.uniform([-40, -40, -3, 0.0], [40, 40, 1, 1.0], size=(n_points, 4)).astype("float32")
    return LidarData(0.0, 0, PointMatrix3D(pts, calib), calib, 0)


def _synthetic_image(h: int = 600, w: int = 800):
    import numpy as np

    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(h, w, 3), dtype="uint8")  # BGR uint8


def _verify_lidar(model: str, dataset: str) -> tuple[bool, str]:
    from avstack.modules.perception.object3d import MMDetObjectDetector3D

    det = MMDetObjectDetector3D(model=model, dataset=dataset, gpu=0)
    dets = det(_synthetic_cloud(), frame=0)
    n = len(dets)
    return True, f"classes={det.class_names} thr={det.threshold} -> {n} dets"


def _verify_cam(model: str, dataset: str) -> tuple[bool, str]:
    from avstack.modules.perception.object2dfv import MMDetObjectDetector2D

    det = MMDetObjectDetector2D(model=model, dataset=dataset, gpu=0)
    # run the loaded detector's real forward path directly on a numpy BGR image
    result = det.inference_detector(det.model, _synthetic_image())
    n = len(getattr(result, "pred_instances", []))
    return True, f"classes={det.class_names} thr={det.threshold} -> {n} raw preds"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lidar", action="store_true", help="verify LiDAR models only")
    ap.add_argument("--cam", action="store_true", help="verify 2D camera models only")
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("VERIFY MODELS: FAIL -> CUDA not available")
        return 1

    todo = []
    if not args.cam:
        todo += LIDAR_MODELS
    if not args.lidar:
        todo += CAM_MODELS

    results = []
    for kind, model, dataset in todo:
        label = f"{kind:5s} {model:12s} {dataset}"
        print(f"\n=== {label} ===")
        try:
            ok, info = (_verify_lidar if kind == "lidar" else _verify_cam)(model, dataset)
            print(f"  PASS  {info}")
            results.append((label, True, info))
        except Exception as exc:  # noqa: BLE001 - report per-model, keep going
            print(f"  FAIL  {type(exc).__name__}: {exc}")
            traceback.print_exc()
            results.append((label, False, f"{type(exc).__name__}: {exc}"))

    n_ok = sum(1 for _, ok, _ in results if ok)
    print("\n" + "=" * 64)
    print(f"SUMMARY: {n_ok}/{len(results)} models verified working")
    for label, ok, info in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}  {info}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
