"""FrameDataCache publishes complete generated sequences and returns independent copies."""

import numpy as np
import pytest

from avsectester.execution.frame_data_cache import FrameDataCache
from avsectester.closed_loop.simulators.base import FrameDataSequence
from avsectester.data_types import ActorState
from avsectester.datasets import FrameData, FrameReference, ScenarioStart
from tests.fakes import MemoryDataset


def generated_sequence():
    frames = tuple(FrameReference("scene", i, i, i * .1) for i in range(3))
    return FrameDataSequence(tuple(FrameData(frame, ego=ActorState(np.eye(4))) for frame in frames), {}, {})


def failed_generation():
    raise RuntimeError("frame data generation failed")


def test_key():
    start = ScenarioStart(MemoryDataset.__name__, "scene", FrameReference("scene", 0, 0, 0.), {"time_step": .1})
    identity = {"type": "MemorySimulator", "environment": {"world": {}}}
    key = FrameDataCache.key(start, identity, 7)
    assert key == FrameDataCache.key(start, dict(reversed(list(identity.items()))), 7)
    assert key != FrameDataCache.key(start, identity, 8)
    assert key != FrameDataCache.key(start, {**identity, "environment": {"world": {"weather": "rain"}}}, 7)


def test_get_or_generate():
    start = ScenarioStart(MemoryDataset.__name__, "scene", FrameReference("scene", 0, 0, 0.), {"time_step": .1})
    key = FrameDataCache.key(start, {"type": "MemorySimulator"}, 7)
    cache = FrameDataCache()
    with pytest.raises(RuntimeError, match="generation failed"):
        cache.get_or_generate(key, failed_generation)
    with pytest.raises(KeyError, match=key):
        cache.get(key)
    incomplete = FrameDataSequence(generated_sequence().frames[1:], {}, {})
    with pytest.raises(ValueError, match="contiguous frame positions"):
        cache.get_or_generate(key, lambda: incomplete)
    generated = cache.get_or_generate(key, generated_sequence)
    generated.frames[0].ego.pose[:] = 0
    np.testing.assert_array_equal(cache.get(key).frames[0].ego.pose, np.eye(4))
    # Reuse the published sequence.
    np.testing.assert_array_equal(cache.get_or_generate(key, failed_generation).frames[0].ego.pose,
                                  np.eye(4))
