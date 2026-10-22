"""Policy-specific context, trajectory timing and ownership at the model boundary."""
from types import SimpleNamespace

import numpy as np
import pytest

from avsectester.closed_loop.avstacks.alpamayo import AlpamayoAVStack
from avsectester.data_types import Observation
from avsectester.runlog import Recorder, RunLog, StepRecord
from avsectester.closed_loop.avstacks.alpamayo import alpamayo_driving_adapter
from avsectester.execution.services import ClosedLoopServices
from tests.fakes import MemoryDataset, MemorySimulator


def test_driving_adapter(alpamayo_model_boundary):
    adapter = alpamayo_driving_adapter(lambda seed: AlpamayoAVStack(
        model=alpamayo_model_boundary.create(), camera_ids=["front"], device="cpu"))
    observation = Observation(1., 10, {"front": np.zeros((4, 6, 3), np.uint8)}, ego_speed=2.)
    with ClosedLoopServices(MemoryDataset(), MemorySimulator(), adapter) as services:
        with services.session(7) as session:
            component, = session.components()
            assert component.id == "policy"
            result = session.evaluate("policy", observation)
            assert result.value.trajectory == [
                ((1., 0., 0.), (1., 0., 0., 0.), 1_100_000),
                ((2., 0., 0.), (1., 0., 0., 0.), 1_200_000),
            ]


def test_trajectory_context(alpamayo_model_boundary):
    model = alpamayo_model_boundary.create()
    avstack = AlpamayoAVStack(model=model, camera_ids=["front"], device="cpu", context_length=2)
    first = Observation(1., 10, {"front": np.zeros((4, 6, 3), np.uint8)},
                        vehicle_state=SimpleNamespace(x=4., y=-2., yaw=np.pi / 2), ego_speed=2.)
    avstack.reset(first)
    command = avstack(first)
    assert command.trajectory == [((1., 0., 0.), (1., 0., 0., 0.), 1_100_000),
                                  ((2., 0., 0.), (1., 0., 0., 0.), 1_200_000)]
    model_input = model.last_input
    assert model_input.speed == 2. and model_input.previous_plan is None
    assert len(model_input.camera_images["front"]) == 2
    times = [pose.timestamp_us for pose in model_input.ego_pose_history]
    assert times == sorted(times) and times[-1] - times[0] >= 1_500_000
    second = Observation(1.1, 11, {"front": np.ones((4, 6, 3), np.uint8)}, ego_speed=3.)
    avstack(second)
    assert model.last_input.previous_plan == "forward"
    assert model.last_input.camera_images["front"][-1].image[0, 0, 0] == 1
    recorded = avstack.component_log()
    assert recorded["command"].trajectory[-1][-1] == 1_300_000
    assert recorded["policy"].reasoning_text == "clear"
    avstack.reset(first)
    avstack(first)
    assert model.last_input.previous_plan is None


def test_close(alpamayo_model_boundary):
    observation = Observation(0., 0, {"front": np.zeros((4, 6, 3), np.uint8)})
    owned = AlpamayoAVStack(checkpoint_path="synthetic-checkpoint", device="cpu", camera_ids=["front"])
    owned.reset(observation)
    model = owned.models["policy"]
    owned(observation)
    owned.close()
    assert model.closed and owned.models == {} and owned.component_log() is None
    owned.close()
    with pytest.raises(RuntimeError, match="closed"):
        owned.reset(observation)

    borrowed_model = alpamayo_model_boundary.create()
    borrowed = AlpamayoAVStack(model=borrowed_model, camera_ids=["front"], device="cpu")
    borrowed.reset(observation)
    borrowed.close()
    assert not borrowed_model.closed and borrowed.models == {}
    with pytest.raises(ValueError, match="checkpoint_path"):
        AlpamayoAVStack(checkpoint_path="", device="cpu").reset(observation)


def test_native_records(alpamayo_model_boundary, tmp_path):
    avstack = AlpamayoAVStack(model=alpamayo_model_boundary.create(),
                              camera_ids=["front"], device="cpu")
    observation = Observation(0., 0, {"front": np.zeros((4, 6, 3), np.uint8)})
    avstack.reset(observation)
    command = avstack(observation)
    components = avstack.component_log()
    with Recorder(tmp_path) as recorder:
        recorder.record_step(StepRecord(0, components=components, produced_command=command))
    saved = RunLog.read(tmp_path).steps[0]
    assert saved.components["policy"]["selected_index"] == 0
    assert saved.components["policy"]["reasoning_text"] == "clear"
    np.testing.assert_array_equal(saved.components["policy"]["candidate_positions"],
                                  components["policy"].candidate_positions)
    assert saved.produced_command.trajectory == command.trajectory
    assert components["policy"].reasoning_text == "clear"
    avstack.close()
