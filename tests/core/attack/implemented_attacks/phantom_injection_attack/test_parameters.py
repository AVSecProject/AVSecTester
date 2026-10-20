"""Attack construction rejects bad parameters before touching data or simulators."""

import numpy as np
import pytest

from avsectester.attack.implemented_attacks.phantom_injection_attack import PhantomInjectionAttack


@pytest.mark.parametrize("options,exception", [
    ({"decision_steps": True}, ValueError),
    ({"decision_steps": 0}, ValueError),
    ({"decision_steps": 1.5}, ValueError),
    ({"decision_steps": 2, "keep_tail": 1}, TypeError),
])
def test_bad_segment_options_fail_at_attack_construction(options, exception):
    with pytest.raises(exception):
        PhantomInjectionAttack(**options)

@pytest.mark.parametrize("count", [0, True, -1, 1.5])
def test_count_limits_fail_during_construction(count):
    with pytest.raises(ValueError, match="max_segments"):
        PhantomInjectionAttack(decision_steps=2, max_segments=count)

def test_phantom_uses_canonical_length_width_height_and_a_centre_position():
    attack = PhantomInjectionAttack(decision_steps=1)
    assert attack.extent == (4., 1.8, 1.6)
    np.testing.assert_allclose(attack.target_xyz, (6., 0., -.7))
    assert attack.target_xyz[2] - attack.extent[2] / 2 == pytest.approx(-1.5)
    runtime = attack.create_runtime(None, None)
    with pytest.raises(ValueError, match="source_reference"):
        runtime.inject([], None)
