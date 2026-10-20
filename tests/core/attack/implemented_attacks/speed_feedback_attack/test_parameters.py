"""Attack construction rejects bad parameters before touching data or simulators."""
from types import SimpleNamespace

import pytest

from avsectester.attack.implemented_attacks.speed_feedback_attack import SpeedFeedbackAttack
from avsectester.attack.runtime import OnlineContext


@pytest.mark.parametrize("options,exception", [
    ({"decision_steps": True}, ValueError),
    ({"decision_steps": 0}, ValueError),
    ({"decision_steps": 1.5}, ValueError),
    ({"decision_steps": 2, "keep_tail": 1}, TypeError),
])
def test_bad_segment_options_fail_at_attack_construction(options, exception):
    with pytest.raises(exception):
        SpeedFeedbackAttack(target_speed=4., **options)

@pytest.mark.parametrize("count", [0, True, -1, 1.5])
def test_count_limits_fail_during_construction(count):
    with pytest.raises(ValueError, match="max_segments"):
        SpeedFeedbackAttack(decision_steps=2, target_speed=4., max_segments=count)

def test_speed_feedback_checks_converted_numbers_and_records_bias():
    attack = SpeedFeedbackAttack(decision_steps=2, target_speed="5", gain="0.5",
                                max_bias="3", initial_bias="1")
    runtime = attack.create_runtime(None, None)
    recorded = []
    context = OnlineContext(1, "step", .1, 1,
        physical_state=SimpleNamespace(ego_speed=2.),
        _output_recorder=lambda context, name, value: recorded.append((name, value)))
    runtime.step(context)
    assert recorded == [("speed_bias", 2.5)]
    assert runtime.bias == 2.5
    assert not hasattr(runtime, "feedback")
    with pytest.raises(ValueError, match="finite"):
        SpeedFeedbackAttack(decision_steps=1, target_speed="nan")
