"""Provided driving systems satisfy one observation-to-command component contract."""
from types import SimpleNamespace

import pytest

from avsectester.closed_loop.avstacks.implemented.alpamayo import AlpamayoAVStack
from avsectester.closed_loop.avstacks.implemented.camera_forward_collision import CameraForwardCollisionAVStack
from avsectester.closed_loop.avstacks.implemented.modular import ModularAVStack
from avsectester.closed_loop.avstacks.implemented.rule_based import RuleBasedAVStack
from avsectester.data_types import Control, Detection
from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter


CASES = [
    pytest.param(SimpleNamespace(
        constructor=ModularAVStack, options={}, fixtures={"pipeline_cfg": "pipeline_config"},
        observation="native_observation", past=lambda request: (),
        prepare=lambda request: pytest.importorskip("avstack.config"),
        component_keys=set(), no_components=True, nullable=set(), categories=("Car",),
        no_detections=False, command_modes=frozenset({"actuator"}),
    ), id="modular"),
    pytest.param(SimpleNamespace(
        constructor=AlpamayoAVStack, options={"checkpoint_path": "synthetic-checkpoint",
            "camera_ids": ["front"], "device": "cpu"}, fixtures={}, observation="policy_observation",
        past=lambda request: request.getfixturevalue("policy_past"),
        prepare=lambda request: request.getfixturevalue("alpamayo_model_boundary"),
        component_keys={"policy", "command", "route_waypoints", "navigation_command",
            "history_pose_count", "selected_image_times", "navigation_note"},
        no_components=False, nullable={"route_waypoints"}, categories=(), no_detections=True,
        command_modes=frozenset({"trajectory"}),
    ), id="alpamayo"),
    pytest.param(SimpleNamespace(
        constructor=CameraForwardCollisionAVStack, options={}, fixtures={},
        observation="camera_observation", past=lambda request: (),
        prepare=lambda request: request.getfixturevalue("camera_detector_boundary"),
        component_keys={"perception"}, no_components=False, nullable=set(), categories=(),
        no_detections=True, command_modes=frozenset({"actuator"}),
    ), id="camera-forward-collision"),
    pytest.param(SimpleNamespace(
        constructor=RuleBasedAVStack, options={"camera_id": "front"},
        fixtures={"detector": "rule_based_detector"}, observation="rule_based_observation",
        past=lambda request: (), prepare=lambda request: request.getfixturevalue("rule_based_detector_model"),
        component_keys={"detections", "sign_estimates", "state", "target_track_id",
            "distance_to_stop_m", "acceleration_mps2", "planned_speed_mps", "braking", "tracks"},
        no_components=False, nullable=set(), categories=(), no_detections=True,
        command_modes=frozenset({"trajectory"}),
    ), id="rule-based"),
]


@pytest.mark.parametrize("case", CASES)
def test_avstack_contract(case, request):
    case.prepare(request)
    options = {**case.options, **{name: request.getfixturevalue(fixture)
                                 for name, fixture in case.fixtures.items()}}
    observation = request.getfixturevalue(case.observation)()
    adapter = DrivingAdapter(lambda seed: case.constructor(**options))
    avstack = adapter.factory(0)
    try:
        avstack.reset(observation, past=case.past(request))
        assert isinstance(avstack(observation), Control)
        assert avstack.command_modes == case.command_modes
        components = avstack.component_log()
        assert (components is None) == case.no_components
        assert set(components or {}) == case.component_keys
        assert {key for key, value in (components or {}).items() if value is None} == case.nullable
        detections = avstack.detections()
        assert (detections is None) == case.no_detections
        assert tuple(detection.category for detection in detections or ()) == case.categories
        assert all(isinstance(detection, Detection) for detection in detections or ())
    finally:
        avstack.close()
