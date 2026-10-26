"""AVStack components share one construction, input and output template."""
import pytest

from avsectester.closed_loop.avstacks.implemented.alpamayo import AlpamayoAVStack
from avsectester.closed_loop.avstacks.implemented.camera_forward_collision import CameraForwardCollisionAVStack
from avsectester.closed_loop.avstacks.implemented.modular import ModularAVStack
from avsectester.data_types import Control, Detection
from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter


@pytest.mark.parametrize(
    "constructor,options,option_fixtures,observation_fixture,past_fixture,boundary,component_keys,nullable_components,detection_categories", [
    pytest.param(ModularAVStack, {}, {"pipeline_cfg": "pipeline_config"},
                 "native_observation", "empty_past", "native_dependencies", None, set(), ("Car",), id="modular"),
    pytest.param(AlpamayoAVStack, {"checkpoint_path": "synthetic-checkpoint", "camera_ids": ["front"],
                                  "device": "cpu"}, {}, "policy_observation", "policy_past",
                 "alpamayo_model_boundary", {"policy", "command", "route_waypoints", "navigation_command",
                    "history_pose_count", "selected_image_times", "navigation_note"}, {"route_waypoints"}, None, id="alpamayo"),
    pytest.param(CameraForwardCollisionAVStack, {}, {}, "camera_observation", "empty_past",
                 "camera_detector_boundary", {"perception"}, set(), None, id="camera-forward-collision"),
])
def test_avstack(constructor, options, option_fixtures, observation_fixture, past_fixture, boundary,
                 component_keys, nullable_components, detection_categories, request):
    request.getfixturevalue(boundary)
    options = {**options, **{name: request.getfixturevalue(fixture)
                            for name, fixture in option_fixtures.items()}}
    observation = request.getfixturevalue(observation_fixture)()
    adapter = DrivingAdapter(lambda seed: constructor(**options))
    avstack = adapter.factory(0)
    try:
        avstack.reset(observation, past=request.getfixturevalue(past_fixture))
        assert isinstance(avstack(observation), Control)
        assert avstack.command_modes
        components = avstack.component_log()
        assert (components is None) == (component_keys is None)
        assert set(components or {}) == set(component_keys or ())
        assert {key for key, value in (components or {}).items() if value is None} == nullable_components
        detections = avstack.detections()
        assert (detections is None) == (detection_categories is None)
        assert tuple(detection.category for detection in detections or ()) == (detection_categories or ())
        assert all(isinstance(detection, Detection) for detection in detections or ())
    finally:
        avstack.close()
