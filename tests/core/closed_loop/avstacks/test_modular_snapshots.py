"""Stage telemetry records outputs when emitted, before downstream mutation."""

from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.closed_loop.avstacks.modular import _StageCapture


def test_stage_capture_detaches_output_before_the_next_component_runs():
    capture = _StageCapture()
    output = [{"score": 0.8, "box": np.array([1.0, 2.0, 3.0])}]
    assert capture(output)[0] is output
    output[0]["score"] = 0.0
    output[0]["box"][:] = 99
    assert capture.last[0]["score"] == 0.8
    np.testing.assert_array_equal(capture.last[0]["box"], [1.0, 2.0, 3.0])
    previous = capture.last
    capture([])
    assert capture.last == [] and previous[0]["score"] == 0.8
    assert capture._recorder.log.path is None
    assert capture._recorder.log.steps == []


def test_modular_unsupported_perception_output_does_not_interrupt_driving():
    from types import SimpleNamespace
    from avsectester.closed_loop.avstacks.modular import ModularAVStack
    from avsectester.closed_loop.datatypes import Observation

    avstack = object.__new__(ModularAVStack)
    avstack._perception_capture = SimpleNamespace(last={'segmentation': np.ones((2, 2))})
    avstack._perception_capture.__call__ = lambda *_: None
    avstack.pipeline = SimpleNamespace(
        perception=SimpleNamespace(post_hooks=[]),
    )
    class Pipeline:
        perception = avstack.pipeline.perception
        def __call__(self, data, vehicle_state):
            return SimpleNamespace(throttle=.4, steer=.1, brake=0.)
    avstack.pipeline = Pipeline()
    observation = Observation(0, 0, {'front': np.zeros((2, 2, 3), np.uint8)})
    command = avstack(observation)
    assert command.throttle == .4
    assert avstack.detections() is None


@pytest.mark.parametrize("sensor_reference", [
    SimpleNamespace(reference=None),
    SimpleNamespace(reference=SimpleNamespace(reference=None, is_global_origin=True)),
], ids=["no-parent", "world-parent"])
def test_modular_detection_reference_requires_a_sensor_body_parent(sensor_reference):
    from avsectester.closed_loop.avstacks.modular import ModularAVStack
    from avsectester.closed_loop.datatypes import Observation

    avstack = object.__new__(ModularAVStack)
    observation = Observation(0, 0, {"front": np.zeros((2, 2, 3), np.uint8)},
                              calibration={"front": SimpleNamespace(reference=sensor_reference)})
    avstack._remember_capture_reference(observation)
    assert avstack._detection_reference is None


def test_modular_malformed_detection_outputs_are_unavailable():
    from types import SimpleNamespace
    from avsectester.closed_loop.avstacks.modular import ModularAVStack

    class Box:
        center = SimpleNamespace(x=(1., 2., 3.))
        l, w, h, yaw = 4., 2., 1., 0.
        def change_reference(self, reference, inplace):
            return self

    avstack = object.__new__(ModularAVStack)
    for fields in ({"obj_type": 5, "score": .8}, {"obj_type": "Car", "score": "broken"},
                   {"score": .8}, {"obj_type": "Car", "score": np.nan}):
        avstack._perception_capture = SimpleNamespace(last=[SimpleNamespace(data=Box(), **fields)])
        avstack._detection_reference = object()
        avstack._last_detections = None
        avstack._detections_ready = False
        assert avstack.detections() is None
        assert avstack.detections() is None
