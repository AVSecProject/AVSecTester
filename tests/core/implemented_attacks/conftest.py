"""Recorded approaches and native boundaries for the concrete-attack template."""

import numpy as np
import pytest

from avsectester.attack import AttackSegment
from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter
from avsectester.closed_loop.avstacks.implemented.modules import (
    RuleBasedAVStack, YOLOXDetector, rule_based_driving_adapter,
)
from avsectester.closed_loop.simulators.ego_dynamics import TrajectoryFollower
from avsectester.closed_loop.simulators.nurec import NuRecSimulator
from avsectester.execution.services import ClosedLoopServices, ScenarioView
from avsectester.implemented_attacks.stop_sign_patch_attack import StopSignPatchAttack
from tests.fakes import (
    CAMERA, ConstantAVStack, DeceleratingAVStack, MemoryDataset, MemorySimulator,
    NuRecImageBoundary, StopApproachDataset, stop_approach_map,
)


@pytest.fixture
def memory_dataset():
    return MemoryDataset()


@pytest.fixture
def stop_approach_dataset():
    return StopApproachDataset(stop_approach_map())


@pytest.fixture
def approach_view():
    services = []

    def create(*, line_kind="stop_sign", facing=True, stopped=True, past_steps=15,
               line_offset_y=0., linked_sign=True, sign_x=60.):
        scenario_map = stop_approach_map(
            line_kind=line_kind, facing=facing, line_offset_y=line_offset_y,
            linked_sign=linked_sign, sign_x=sign_x,
        )
        dataset = StopApproachDataset(scenario_map, stopped=stopped)
        adapter = DrivingAdapter(lambda seed: ConstantAVStack(), past_steps=past_steps)
        scope = ClosedLoopServices(dataset, MemorySimulator(), adapter)
        scope.__enter__()
        services.append(scope)
        return ScenarioView("scene", scope, 0)

    yield create
    for scope in reversed(services):
        scope.close()


@pytest.fixture
def default_driving_options():
    return {}


@pytest.fixture
def native_driving_options():
    pytest.importorskip("avstack.config")
    from avstack.datastructs import DataContainer
    from avstack.geometry import GlobalOrigin3D

    return {"perception_output": DataContainer(0, 0., [], "front", source_reference=GlobalOrigin3D)}


@pytest.fixture
def nurec_patch_services_and_segment(nurec_longitudinal_dataset):
    """Moving recorded camera, real binding, and image-independent candidate driver."""
    dataset = nurec_longitudinal_dataset
    simulator = NuRecSimulator(
        NuRecImageBoundary(CAMERA), TrajectoryFollower(), depth="none",
        route_generator="none", physics_update_mode="none", object_occlusion="geometry",
        time_step=.1,
    )
    adapter = DrivingAdapter(lambda seed: DeceleratingAVStack(), past_steps=15)
    frames = tuple(dataset.frames("scene"))
    route = np.asarray([dataset.frame_data("scene", frame).ego.pose[:3, 3] for frame in frames])

    def make_segment(*, start=15, stop=30, line_x=28.):
        sign_pose = np.diag([-1., -1., 1., 1.])
        sign_pose[:3, 3] = (28., -1., 1.8)
        segment = AttackSegment("scene", frames[start], frames[stop], {
            "sign_id": "patch-sign", "sign_pose": sign_pose,
            "sign_extent": [.05, .8, .8], "sign_native_category": "MUTCD_R1_STOP",
            "sign_lane_ids": [], "stop_line_id": "patch-line",
            "stop_line_points": np.array([[line_x, -3., 0.], [line_x, 3., 0.]]),
            "stop_line_lane_ids": [], "recorded_stop_frame": 30,
            "recorded_stop_timestamp_s": frames[30].timestamp,
            "approach_distance_m": 28. - route[start, 0], "route_points": route,
            "line_route_distance_m": line_x,
        })
        StopSignPatchAttack().validate_segment(segment)
        return segment

    with ClosedLoopServices(dataset, simulator, adapter) as services:
        yield services, make_segment


@pytest.fixture
def replica_detector_model():
    from tests.fakes import replica_detector

    return replica_detector()


@pytest.fixture
def nurec_detector_services_and_segment(nurec_patch_services_and_segment, replica_detector_model):
    """Real rule-based scoring binding over the shared synthetic recorded camera world."""
    services, make_segment = nurec_patch_services_and_segment

    def factory(seed):
        return RuleBasedAVStack(YOLOXDetector(
            "unused.pth", device="cpu", score_threshold=0., model=replica_detector_model,
        ), camera_id="front")

    factory.avstack_options = {
        "camera_id": "front", "detector": YOLOXDetector("unused.pth", device="cpu", score_threshold=0.),
    }
    adapter = rule_based_driving_adapter(factory)
    with ClosedLoopServices(services.dataset, services.simulator, adapter) as scope:
        yield scope, make_segment
