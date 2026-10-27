"""Driving adapters expose declared input history without loading a driving system."""

from dataclasses import replace

import pytest

from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter, ScorerDescriptor
from tests.fakes import ConstantAVStack, HistoryAVStack


def test_past_steps():
    def construct(seed):
        raise RuntimeError("Reading the history declaration must not construct a driving system")

    assert DrivingAdapter(construct).past_steps == 0
    assert DrivingAdapter(construct, past_steps=3).past_steps == 3
    construct.avstack_type = HistoryAVStack
    assert DrivingAdapter(construct, past_steps=3).past_steps == 2
    construct.avstack_type = ConstantAVStack
    assert DrivingAdapter(construct, past_steps=3).past_steps == 0
    construct.avstack_type = int
    assert DrivingAdapter(construct, past_steps=3).past_steps == 3
    with pytest.raises(ValueError, match="past_steps must be a nonnegative integer"):
        DrivingAdapter(construct, past_steps=True)


def test_scorer_descriptor():
    descriptor = ScorerDescriptor(
        "detector.confidence", {"image": "CHW float tensor [0,1]"},
        {"score": "scalar confidence"}, {"score": "tensor"}, ("image",),
        sensor_id="front", decision_threshold=.5,
    )
    assert descriptor.sensor_id == "front" and descriptor.decision_threshold == .5
    assert replace(descriptor, sensor_id=None, decision_threshold=None).sensor_id is None
    with pytest.raises(ValueError, match="sensor_id"):
        replace(descriptor, sensor_id=3)
    with pytest.raises(ValueError, match="sensor_id"):
        replace(descriptor, sensor_id="")
    with pytest.raises(ValueError, match="decision_threshold"):
        replace(descriptor, decision_threshold=True)
    with pytest.raises(ValueError, match="decision_threshold"):
        replace(descriptor, decision_threshold=1.1)
    with pytest.raises(ValueError, match="decision_threshold"):
        replace(descriptor, decision_threshold=float("nan"))
