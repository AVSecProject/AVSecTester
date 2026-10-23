"""Native-boundary simulator cases shared by the component contract tests."""

import pytest

from avsectester.closed_loop.simulators.base import Initialization
from avsectester.closed_loop.simulators.carla import CarlaSimulator
from avsectester.closed_loop.simulators.nurec import NuRecRenderer, NuRecSimulator
from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator
from tests.fakes import CarlaNativeBoundary, NuRecRpcBoundary


@pytest.fixture
def carla_case(carla_dataset, monkeypatch):
    CarlaNativeBoundary().install(monkeypatch)
    simulator = CarlaSimulator({"rate": 10}, settle_iters=0)
    frames = carla_dataset.frames("scene")
    with simulator.create_runtime(seed=3) as runtime:
        sequence = runtime.generate_frame_data(carla_dataset.scenario_start("scene", frames[0]), frames)
    start = carla_dataset.scenario_start("scene", frames[2])
    initialization = Initialization(sequence.at(frames[2]), sequence.resolved_scenario, sequence.provenance)
    initial = carla_dataset.scenario_start("scene", frames[0])
    initial_state = Initialization(sequence.at(frames[0]), sequence.resolved_scenario, sequence.provenance)
    return carla_dataset, simulator, start, initialization, (12, 16, 3), initial, initial_state


@pytest.fixture
def nurec_case(nurec_dataset, monkeypatch):
    NuRecRpcBoundary().install(monkeypatch)
    # Shared simulator contracts compare inserted objects with geometric composites.
    simulator = NuRecSimulator({"dt": .1}, NuRecRenderer(cameras=["front"]), object_occlusion="geometry")
    start = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[1])
    initial = nurec_dataset.scenario_start("scene", nurec_dataset.frames("scene")[0])
    return nurec_dataset, simulator, start, None, (12, 16, 3), initial, None


@pytest.fixture
def recorded_case(nuscenes_dataset):
    simulator = RecordedReplaySimulator(camera_aliases={"CAM_FRONT": "front"})
    start = nuscenes_dataset.scenario_start("scene", nuscenes_dataset.frames("scene")[1])
    initial = nuscenes_dataset.scenario_start("scene", nuscenes_dataset.frames("scene")[0])
    return nuscenes_dataset, simulator, start, None, (80, 100, 3), initial, None


@pytest.fixture(params=["carla_case", "nurec_case", "recorded_case"])
def simulator_case(request):
    return request.getfixturevalue(request.param)
