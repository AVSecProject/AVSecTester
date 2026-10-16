"""Reference ownership and cache identity."""

from dataclasses import replace

import numpy as np
import pytest

from avsectester.datasets.base import ActorState, DataReference, FrameRef, ReferenceFrame, SourceData
from avsectester.reference import ReferenceCache, ReferenceSequence


def reference():
    frame = FrameRef("road", 0, 100, 0)
    return ReferenceFrame(frame, None, {
        "ego": ActorState(np.eye(4), (0, 0, 0)),
        "host": ActorState(np.eye(4), (1, 0, 0)),
    }, {"front": DataReference("capture", "front", 0)})


def test_cache_key_covers_recipe_simulator_and_seed_and_reads_are_detached():
    cache = ReferenceCache()
    expected = reference()
    source = SourceData("carla", "road", expected.frame, {"recipe": {"rate": 20}})
    identity = {"version": "0.9.16", "camera": "pinhole"}
    key = cache.key(source, identity, 1)
    assert key != cache.key(source, identity, 2)
    assert key != cache.key(source, {**identity, "version": "next"}, 1)
    assert key != cache.key(replace(source, payload={"recipe": {"rate": 10}}), identity, 1)
    calls = []

    def generate():
        calls.append(1)
        return ReferenceSequence((expected,), {"actual_spawn": 5}, identity)

    first = cache.materialize(key, generate)
    first.frames[0].actors["ego"].pose[0, 3] = 99
    assert cache.materialize(key, generate).frames[0].actors["ego"].pose[0, 3] == 0
    assert calls == [1]


def test_failed_materialization_is_not_cached():
    cache = ReferenceCache()
    with pytest.raises(ValueError):
        cache.materialize("key", lambda: ReferenceSequence((), {}, {}))
    with pytest.raises(KeyError):
        cache.get("key")
