"""Read recorded frames from real nuScenes when AVSECTESTER_NUSCENES_ROOT is set."""
import os
from pathlib import Path

import pytest
from avsectester.datasets.nuscenes import NuScenesDataset

_ROOT = os.environ.get('AVSECTESTER_NUSCENES_ROOT')


@pytest.mark.skipif(_ROOT is None or not (Path(_ROOT) / 'v1.0-trainval').is_dir(),
                    reason='Set AVSECTESTER_NUSCENES_ROOT to nuScenes v1.0-trainval data')
def test_real_nuscenes_frames_have_actor_extent_and_calibration():
    pytest.importorskip('nuscenes.nuscenes')
    data = NuScenesDataset(str(_ROOT))
    scenario = next(iter(data.scenarios()))
    frames = data.frames(scenario.id)
    record = data.frame_data(scenario.id, frames[0])
    assert record.ego is not None and record.sensor_data
    assert any(actor.extent and min(actor.extent) > 0 for actor in record.actors.values())
    reference = next(iter(record.sensor_data.values()))
    assert {'camera', 'ego_from_sensor'} <= reference.metadata.keys()
