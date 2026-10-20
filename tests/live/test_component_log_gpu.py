"""Optional GPU detector hook test on a nuScenes image.

This checks output capture rather than detection accuracy. The CARLA-trained checkpoint is evaluated
on a different image domain. Missing dataset, weights or optional dependencies skip this test.
"""

import os
from pathlib import Path

import pytest

_ROOT = os.environ.get("AVSECTESTER_NUSCENES_ROOT")
_WEIGHTS = (
    Path(__file__).resolve().parents[2] / "third_party/avstack-core/third_party/"
    "mmdetection/work_dirs/carla/faster_rcnn_r50_fpn_1x_carla_vehicle.pth"
)


def test_stage_capture_records_real_gpu_detector_output():
    # Resolve prerequisites at execution time so collecting live tests does not initialize CUDA.
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if _ROOT is None or not (Path(_ROOT) / "v1.0-trainval").is_dir() or not _WEIGHTS.is_file():
        pytest.skip("Set AVSECTESTER_NUSCENES_ROOT to nuScenes trainval data and fetch the "
                    "CARLA 2D detector weights")
    root = Path(_ROOT)
    pytest.importorskip("avstack")
    pytest.importorskip("mmdet")
    pytest.importorskip("nuscenes.nuscenes")
    import numpy as np
    from avsectester.closed_loop.avstacks.modular import _StageCapture
    from avstack.calibration import CameraCalibration
    from avstack.modules.perception.object2dfv import MMDetObjectDetector2D
    from avstack.geometry import GlobalOrigin3D
    from avstack.sensors import ImageData
    from nuscenes.nuscenes import NuScenes
    from PIL import Image

    # real image + real camera intrinsics from the nuScenes devkit (no CAN-bus dependency)
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(root), verbose=False)
    cam = nusc.get("sample_data", nusc.sample[0]["data"]["CAM_FRONT"])
    K = np.asarray(
        nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])["camera_intrinsic"],
        dtype=float,
    )
    rgb = np.asarray(Image.open(root / cam["filename"]).convert("RGB"))
    calib = CameraCalibration(
        GlobalOrigin3D, np.hstack([K, np.zeros((3, 1))]), img_shape=rgb.shape, channel_order="rgb"
    )
    img = ImageData(timestamp=0.0, frame=0, data=rgb, calibration=calib, source_ID="CAM_FRONT")

    # real GPU neural detector + the project's real component-logging post-hook
    gpu = int(os.environ.get("AVSECTESTER_TEST_GPU", "0"))
    det = MMDetObjectDetector2D(
        model="fasterrcnn", dataset="carla-vehicle", gpu=gpu, epoch="latest"
    )
    cap = _StageCapture()
    det.register_post_hook(cap)

    dets = det(img, frame=0)  # real inference on the GPU

    # The hook passes the detector output on unchanged and records a detached copy of it.
    recorded = cap.last
    assert recorded is not dets and len(recorded) == len(dets)
    for copy, detection in zip(recorded, dets):
        assert copy is not detection
        assert copy.obj_type == detection.obj_type and copy.score == detection.score
        np.testing.assert_allclose(copy.box.box2d, detection.box.box2d)
    assert cap(dets)[0] is dets
