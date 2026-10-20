"""Read real NuRec data when AVSECTESTER_NUREC_ROOT points at the licensed sample set."""
import os
from pathlib import Path

import numpy as np
import pytest

from avsectester.datasets.nurec import NuRecDataset

_ROOT = os.environ.get('AVSECTESTER_NUREC_ROOT')
_UUID = '023b7fcc-671c-40e3-9bd2-c66b0b073fbc'
_SCENE = Path(_ROOT or '.') / _UUID / f'{_UUID}.usdz'
pytestmark = pytest.mark.skipif(_ROOT is None or not _SCENE.is_file(),
                                reason='Set AVSECTESTER_NUREC_ROOT to the licensed NuRec sample set')


def test_real_nurec_timeline_and_actor_geometry():
    data = NuRecDataset([str(_SCENE)], time_step=.1)
    scenario = next(iter(data.scenarios()))
    frames = data.frames(scenario.id)
    assert len(frames) > 20
    np.testing.assert_allclose(np.diff([frame.timestamp for frame in frames]), .1)
    record = data.frame_data(scenario.id, frames[5])
    assert record.ego is not None and record.actors
    assert any(actor.extent and min(actor.extent) > 0 for actor in record.actors.values())
    assert data.scenario_start(scenario.id, frames[5]).payload['usdz_path'] == str(_SCENE)
