"""Driving adapters expose declared input history without loading a driving system."""

from dataclasses import replace

import pytest

from avsectester.closed_loop.avstacks.driving_adapter import ComponentDescriptor, DrivingAdapter, ScorerDescriptor
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
        "detector.confidence", {"images": "BCHW float tensor [0,1]"},
        {"scores": "scalar confidence"}, {"scores": "tensor"}, ("images",),
        sensor_id="front", decision_threshold=.5,
    )
    assert descriptor.sensor_id == "front" and descriptor.decision_threshold == .5
    assert descriptor.inputs == {"images": "BCHW float tensor [0,1]"}
    assert descriptor.outputs == {"scores": "scalar confidence"}
    assert descriptor.differentiable_inputs == ("images",) and descriptor.output_shape == ()
    assert descriptor.direction == "minimize"
    assert replace(descriptor, direction="maximize").direction == "maximize"
    with pytest.raises(ValueError, match="direction"):
        replace(descriptor, direction="invalid")
    assert replace(descriptor, sensor_id=None, decision_threshold=None).sensor_id is None
    with pytest.raises(ValueError, match="sensor_id"):
        replace(descriptor, sensor_id=3)
    with pytest.raises(ValueError, match="decision_threshold"):
        replace(descriptor, decision_threshold=1.1)


def test_decision_time_step():
    class TimeStepAVStack(ConstantAVStack):
        decision_time_step = .2

    def factory(seed):
        raise AssertionError("Reading declarations must not construct the driving system")

    assert DrivingAdapter(factory).decision_time_step is None
    assert DrivingAdapter(factory, decision_time_step=.3).decision_time_step == .3
    factory.avstack_type = TimeStepAVStack
    assert DrivingAdapter(factory, decision_time_step=.3).decision_time_step == .2
    factory.avstack_type = int
    assert DrivingAdapter(factory, decision_time_step=.3).decision_time_step == .3


def test_bindings():
    descriptor = ComponentDescriptor("policy", {"observation": "Observation"},
                                     {"command": "Control"}, {"command": "control"})
    callback = lambda stack, inputs: stack(inputs)
    adapter = DrivingAdapter(lambda seed: ConstantAVStack(), components=[(descriptor, callback)])
    assert adapter.component_bindings == {"policy": (descriptor, callback)}
    with pytest.raises(ValueError, match="unique"):
        DrivingAdapter(lambda seed: ConstantAVStack(),
                       components=[(descriptor, callback), (descriptor, callback)])
    with pytest.raises(ValueError, match="Differentiable inputs"):
        DrivingAdapter(lambda seed: ConstantAVStack(), components=[
            (replace(descriptor, differentiable_inputs=("missing",)), callback)])
