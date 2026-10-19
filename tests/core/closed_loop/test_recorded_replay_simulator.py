"""Recorded replay preserves RGB pixels, advances source frames and isolates callers."""
import numpy as np
import pytest
from PIL import Image
from avsectester.datasets.base import FrameReference, FrameData, ScenarioStart, SensorDataReference
from avsectester.scenarios.scene import SceneGT, EgoState
from avsectester.closed_loop.datatypes import Control
from avsectester.closed_loop.simulators.recorded import RecordedReplaySimulator


def source(tmp_path):
    refs = []
    for i in range(3):
        path = tmp_path / f"{i}.png"
        Image.fromarray(np.full((2, 3, 3), 100 + i, np.uint8)).save(path)
        frame = FrameReference("s", 12+i, str(i), 4+i*.5)
        refs.append(FrameData(frame, SceneGT(12+i, frame.timestamp, EgoState(), {}, []),
                                   sensor_data={"front": SensorDataReference(str(path), "front", frame.timestamp)}))
    return ScenarioStart("NuScenesDataset", "s", refs[0].frame, {"frame_data": tuple(refs)})


def test_recorded_backend_advances_actual_images_and_resets_configured_origin(tmp_path):
    backend = RecordedReplaySimulator(source(tmp_path))
    assert np.all(backend.reset().sensor_data["front"] == 100)
    advanced = backend.step(Control(throttle=1, steer=.5))
    assert advanced.frame == 13 and advanced.t == 4.5
    assert np.all(advanced.sensor_data["front"] == 101)
    backend.step(Control())
    with pytest.raises(ValueError, match="terminal recorded"):
        backend.step(Control())
    assert backend.reset().frame == 12


def test_recorded_observation_edits_do_not_change_source_or_clock(tmp_path):
    backend = RecordedReplaySimulator(source(tmp_path))
    observation = backend.reset()
    observation.sensor_data["front"][:] = 0
    observation.frame, observation.t = -1, -2
    assert backend.ground_truth().frame == 12
    assert backend.ground_truth().t == 4
    assert np.all(backend.observe().sensor_data["front"] == 100)
