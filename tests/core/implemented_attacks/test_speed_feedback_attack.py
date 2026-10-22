"""Feedback saturation is observable in the model input and recorded attack output."""

import pytest

from avsectester.attack import OfflineContext
from avsectester.implemented_attacks.speed_feedback_attack import SpeedFeedbackAttack
from avsectester.execution.runner import ClosedLoopRunner, RunRequest
from avsectester.execution.services import ClosedLoopServices
from tests.fakes import ConstantAVStack, MemoryDataset, MemorySimulator


@pytest.mark.parametrize("target_speed,expected_bound", [(0., 0.), (100., 2.)])
def test_bias_bounds(target_speed, expected_bound):
    dataset, simulator = MemoryDataset(), MemorySimulator()
    attack = SpeedFeedbackAttack(decision_steps=6, target_speed=target_speed,
                                gain=10., max_bias=2., initial_bias=1.)
    with ClosedLoopServices(dataset, simulator) as services:
        segment, = attack.build_metadata(services.scenario("scene", seed=0))
        artifacts = attack.run_offline(OfflineContext(services, (segment,)))
    with simulator.create_runtime() as simulator_runtime:
        log = ClosedLoopRunner().run(simulator_runtime, ConstantAVStack(), RunRequest(
            6, start=dataset.scenario_start("scene", segment.start),
            attacks=(attack.create_runtime(segment, artifacts),),
        ))
    biases = [output.value for row in log.steps for output in row.outputs if output.name == "speed_bias"]
    assert biases[0] == 1.
    assert all(0. <= bias <= 2. for bias in biases)
    assert biases[-1] == expected_bound
    for row, bias in zip(log.steps, biases):
        assert row.model_observation.ego_speed == pytest.approx(row.physical_pre.ego_speed + bias)
