"""Optional public camera-policy inference on one real nuScenes image."""
import importlib.util
import os
from pathlib import Path

import pytest


def test_camera_policy_with_real_detector():
    import torch
    root = os.environ.get("AVSECTESTER_NUSCENES_ROOT")
    specification = importlib.util.find_spec("avstack")
    avstack_root = (Path(specification.origin).resolve().parents[1]
                    if specification is not None and specification.origin is not None
                    else Path(__file__).resolve().parents[2] / "third_party/avstack-core")
    checkpoint = (avstack_root / "third_party/mmdetection/work_dirs/carla/"
                  "faster_rcnn_r50_fpn_1x_carla_vehicle.pth")
    required = (f"CUDA and local checkpoint {checkpoint}; "
                "AVSECTESTER_NUSCENES_ROOT/v1.0-trainval with CAM_FRONT image files")
    if specification is None or specification.origin is None:
        pytest.skip(required + " (avstack SDK is unavailable)")
    if not torch.cuda.is_available():
        pytest.skip(required + " (CUDA is unavailable)")
    if root is None or not (Path(root) / "v1.0-trainval").is_dir() or not checkpoint.is_file():
        pytest.skip(required)
    pytest.importorskip("avstack.config")
    pytest.importorskip("mmdet.apis")
    pytest.importorskip("nuscenes.nuscenes")
    import numpy as np
    from nuscenes.nuscenes import NuScenes
    from PIL import Image
    from avsectester.closed_loop.avstacks.implemented.camera_forward_collision import CameraForwardCollisionAVStack
    from avsectester.data_types import Control, Observation

    dataset = NuScenes(version="v1.0-trainval", dataroot=root, verbose=False)
    camera = dataset.get("sample_data", dataset.sample[0]["data"]["CAM_FRONT"])
    with Image.open(Path(root) / camera["filename"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    avstack = CameraForwardCollisionAVStack(gpu=int(os.environ.get("AVSECTESTER_TEST_GPU", "0")))
    observation = Observation(0., 0, {"front": rgb})
    try:
        avstack.reset(observation)
        assert isinstance(avstack(observation), Control)
        assert all(len(box) == 4 and isinstance(score, float)
                   for box, score, _ in avstack.component_log()["perception"])
    finally:
        avstack.close()
