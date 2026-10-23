"""Driving adapters expose declared input history without loading a driving system."""

import pytest

from avsectester.closed_loop.avstacks.driving_adapter import DrivingAdapter
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
