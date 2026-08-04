"""Integration: detection-level phantom attack through the REAL neural detector.

Proves the security layer (hook adapter + attack + monitor) composes with actual AI on a
real point cloud: attach PhantomDetectionAttack at the ``perception_out`` seam of a live
MMDetObjectDetector3D and confirm its output grows by exactly the phantom, traced by a
monitor. Gated on avstack + CUDA + a downloaded checkpoint + the mmdet3d demo cloud, so it
runs in the ``avsec`` env and skips cleanly elsewhere.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("avstack")
torch = pytest.importorskip("torch")

if not torch.cuda.is_available():
    pytest.skip("neural-detector test needs a CUDA GPU", allow_module_level=True)

_MM3D = "third_party/avstack-core/third_party/mmdetection3d"
_CKPT = f"{_MM3D}/checkpoints/kitti/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"
_CLOUD = f"{_MM3D}/demo/data/kitti/000008.bin"
if not (os.path.exists(_CKPT) and os.path.exists(_CLOUD)):
    pytest.skip("run scripts/fetch_models.sh (checkpoint/cloud missing)", allow_module_level=True)

import numpy as np
from avsectester.attacks.perception import PhantomDetectionAttack
from avsectester.hooks import RunContext, attach, attach_monitor
from avstack.calibration import LidarCalibration
from avstack.geometry import GlobalOrigin3D, ReferenceFrame
from avstack.geometry.datastructs import PointMatrix3D
from avstack.modules.perception.object3d import MMDetObjectDetector3D
from avstack.sensors import LidarData


def _cloud(calib):
    pts = np.fromfile(_CLOUD, dtype=np.float32).reshape(-1, 4)
    return LidarData(0.0, 0, PointMatrix3D(pts, calib), calib, 0)


def test_phantom_detection_augments_real_detector_output():
    det = MMDetObjectDetector3D(model="pointpillars", dataset="kitti", gpu=0)
    calib = LidarCalibration(
        ReferenceFrame(x=np.array([0.0, 0.0, 1.73]), q=np.quaternion(1), reference=GlobalOrigin3D)
    )

    clean = det(_cloud(calib), frame=0)
    assert len(clean) > 0  # the checkpoint genuinely fires on a real cloud

    ctx = RunContext()
    attach(det, PhantomDetectionAttack(target_xyz=[12.0, 0.0, -1.5], score=0.9), "perception_out", ctx)
    attach_monitor(det, "perception_out", ctx)
    ctx.tick(0, 0.0)
    attacked = det(_cloud(calib), frame=0)

    assert len(attacked) == len(clean) + 1
    phantom = [d for d in attacked if getattr(d, "ID", None) == 90002]
    assert len(phantom) == 1
    assert phantom[0].obj_type == "Car"
    assert list(phantom[0].position.x) == [12.0, 0.0, -1.5]
    # the monitor observed the post-attack detector output
    assert ctx.trace.records[-1].outputs == {"count": len(clean) + 1}
