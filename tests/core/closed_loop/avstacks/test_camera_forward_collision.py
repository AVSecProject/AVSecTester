"""The image policy keeps the existing plausible-box braking rule without model downloads."""

from unittest.mock import Mock

import numpy as np
import pytest

from avsectester.closed_loop.avstacks.camera_forward_collision import CameraForwardCollisionAVStack
from avsectester.closed_loop.datatypes import Observation


@pytest.mark.parametrize("boxes,brake", [
    ([], 0.),
    ([([20, 20, 60, 50], .9, "vehicle")], 1.),
    ([([0, 0, 100, 100], .9, "vehicle")], 0.),
    ([([30, 5, 40, 95], .9, "vehicle")], 0.),
])
def test_plausible_vehicle_boxes_control_braking(monkeypatch, boxes, brake):
    avstack = CameraForwardCollisionAVStack()
    infer = Mock(return_value=boxes)
    monkeypatch.setattr(avstack, "detect", infer)
    observation = Observation(0., 0, {"front": np.zeros((100, 100, 3), np.uint8)})
    command = avstack(observation)
    assert command.brake == brake
    assert command.throttle == (0. if brake else .55)
    assert avstack.component_log()["perception"] == ([] if not brake else boxes)
    assert avstack.detections() is None
    avstack.reset(observation)
    assert avstack.component_log() is None
    infer.assert_called_once()


def test_missing_camera_fails_without_loading_a_detector():
    avstack = CameraForwardCollisionAVStack()
    with pytest.raises(ValueError, match="absent"):
        avstack(Observation(0., 0, {}))
    assert avstack._detector is None


def test_nuscenes_detector_keeps_only_native_vehicle_label_indices(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    class Tensor:
        def __init__(self, data):
            self.data = np.asarray(data)
        def detach(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self.data

    instances = SimpleNamespace(
        bboxes=Tensor([[0, 0, 3, 3]] * 7), scores=Tensor([.8] * 7), labels=Tensor(range(7)),
    )
    api = ModuleType("mmdet.apis")
    api.inference_detector = lambda *args: SimpleNamespace(pred_instances=instances)
    monkeypatch.setitem(sys.modules, "mmdet.apis", api)
    avstack = CameraForwardCollisionAVStack(detector_dataset="nuscenes")
    avstack._detector = SimpleNamespace(model=object())
    assert len(avstack.detect(np.zeros((4, 4, 3), np.uint8))) == 5
