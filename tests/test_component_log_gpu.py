"""Component logging on REAL GPU execution: the project's ``_StageCapture`` post-hook (the same class
``ModularAVStack.instrument()`` attaches) capturing the real output of a real mmdet GPU neural detector
run on a real nuScenes camera image.

This validates the capture *plumbing* on a GPU neural perception stage (not the CPU passthrough of
``test_component_log_real.py``, and not detection accuracy — CARLA-trained weights on a nuScenes image is
a domain shift, so the capture assertion, not the count, is what matters). Self-skips unless a CUDA GPU,
the real nuScenes dataset, and the CARLA-trained 2D weights are all present, so CI stays green.
"""

import os.path as osp

import pytest

pytest.importorskip("avstack")

_NUSCENES = "/workspace/hdd/datasets/nuscenes"
_WEIGHTS = (
    "third_party/avstack-core/third_party/mmdetection/work_dirs/carla/"
    "faster_rcnn_r50_fpn_1x_carla_vehicle.pth"
)


def _gpu_available():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001 - any import/driver failure just means "no GPU here"
        return False


pytestmark = pytest.mark.skipif(
    not (_gpu_available()
         and osp.isdir(osp.join(_NUSCENES, "v1.0-trainval"))
         and osp.isfile(_WEIGHTS)),
    reason="needs a CUDA GPU + the real nuScenes dataset + the CARLA-trained 2D detector weights",
)


def test_stage_capture_records_real_gpu_detector_output():
    import numpy as np
    from avsectester.stacks.modular import _StageCapture
    from avstack.calibration import CameraCalibration
    from avstack.config import MODELS
    from avstack.geometry import GlobalOrigin3D
    from avstack.sensors import ImageData
    from nuscenes.nuscenes import NuScenes
    from PIL import Image

    # real image + real camera intrinsics from the nuScenes devkit (no CAN-bus dependency)
    nusc = NuScenes(version="v1.0-trainval", dataroot=_NUSCENES, verbose=False)
    cam = nusc.get("sample_data", nusc.sample[0]["data"]["CAM_FRONT"])
    K = np.asarray(
        nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])["camera_intrinsic"],
        dtype=float,
    )
    rgb = np.asarray(Image.open(osp.join(_NUSCENES, cam["filename"])).convert("RGB"))
    calib = CameraCalibration(GlobalOrigin3D, np.hstack([K, np.zeros((3, 1))]),
                              img_shape=rgb.shape, channel_order="rgb")
    img = ImageData(timestamp=0.0, frame=0, data=rgb, calibration=calib, source_ID="CAM_FRONT")

    # real GPU neural detector + the project's real component-logging post-hook
    det = MODELS.build(dict(type="MMDetObjectDetector2D", model="fasterrcnn",
                            dataset="carla-vehicle", gpu=0, epoch="latest"))
    cap = _StageCapture()
    det.register_post_hook(cap)

    dets = det(img, frame=0)                       # real inference on the GPU

    # the capture path works on a real GPU neural stage: it recorded the detector's genuine output
    assert cap.last is not None
    assert len(cap.last) == len(dets)
