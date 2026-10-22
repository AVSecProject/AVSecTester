"""AVStack components share one construction, input and output template."""
import pytest

from avsectester.closed_loop.avstacks.alpamayo import AlpamayoAVStack
from avsectester.closed_loop.avstacks.camera_forward_collision import CameraForwardCollisionAVStack
from avsectester.closed_loop.avstacks.modular import ModularAVStack
from avsectester.data_types import Control, Detection
from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter


@pytest.mark.parametrize(
    "constructor,options,option_fixtures,observation_fixture,boundary,component_keys,detection_categories", [
    pytest.param(ModularAVStack, {}, {"pipeline_cfg": "pipeline_config"},
                 "native_observation", "native_dependencies", None, ("Car",), id="modular"),
    pytest.param(AlpamayoAVStack, {"checkpoint_path": "synthetic-checkpoint", "camera_ids": ["front"],
                                  "device": "cpu"}, {}, "camera_observation",
                 "alpamayo_model_boundary", {"policy", "command"}, None, id="alpamayo"),
    pytest.param(CameraForwardCollisionAVStack, {}, {}, "camera_observation",
                 "camera_detector_boundary", {"perception"}, None, id="camera-forward-collision"),
])
def test_avstack(constructor, options, option_fixtures, observation_fixture, boundary,
                 component_keys, detection_categories, request):
    request.getfixturevalue(boundary)
    options = {**options, **{name: request.getfixturevalue(fixture)
                            for name, fixture in option_fixtures.items()}}
    observation = request.getfixturevalue(observation_fixture)()
    adapter = DrivingAdapter(lambda seed: constructor(**options))
    avstack = adapter.factory(0)
    try:
        avstack.reset(observation)
        assert isinstance(avstack(observation), Control)
        assert avstack.command_modes
        components = avstack.component_log()
        assert (components is None) == (component_keys is None)
        assert set(components or {}) == set(component_keys or ())
        assert all(value is not None for value in (components or {}).values())
        detections = avstack.detections()
        assert (detections is None) == (detection_categories is None)
        assert tuple(detection.category for detection in detections or ()) == (detection_categories or ())
        assert all(isinstance(detection, Detection) for detection in detections or ())
    finally:
        avstack.close()
