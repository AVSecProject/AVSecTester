"""Public attack contracts and selected intervals."""

import numpy as np
import pytest

from avsectester.attack import Attack, AttackSegment, OfflineContext, OnlineContext
from avsectester.datasets import FrameReference


def test_save_and_load_artifacts(tmp_path):
    attack = Attack()
    artifacts = {"image": np.full((2, 3, 4), 120, np.uint8)}
    attack.save_artifacts(artifacts, tmp_path)
    restored = attack.load_artifacts(tmp_path)
    np.testing.assert_array_equal(restored["image"], artifacts["image"])


def test_default_phases(segment):
    attack = Attack()
    assert attack.run_offline(OfflineContext(None, (segment,))) is None
    attack.validate_segment(segment)
    assert attack.deployment_ticks(segment) == 0
    runtime = attack.create_runtime(segment, None)
    context = OnlineContext(0, "setup", 0., 0)
    assert runtime.setup(context) == () and runtime.step(context) == ()
    assert runtime.stop(context) is None
    with pytest.raises(NotImplementedError, match="metadata.load"):
        attack.build_metadata(None)


def test_attack_segment():
    start = FrameReference("scene", 2, "frame-2", .2)
    stop = FrameReference("scene", 5, "frame-5", .5)
    segment = AttackSegment("scene", start, stop, user_data={"host": "lead", "label": "visible"})
    assert segment.decision_steps == 3
    assert segment.user_data == {"host": "lead", "label": "visible"}
    with pytest.raises(ValueError, match="stop"):
        AttackSegment("scene", stop, start)


def test_configuration_parameters():
    class ConfiguredAttack(Attack):
        def __init__(self, count=2):
            self.count = count

    assert ConfiguredAttack().configuration_parameters() == {"count": 2}

    class VariadicAttack(Attack):
        def __init__(self, **options):
            pass

    class HiddenParameterAttack(Attack):
        def __init__(self, value=1):
            pass

    with pytest.raises(TypeError, match="variadic"):
        VariadicAttack().configuration_parameters()
    with pytest.raises(TypeError, match="no matching attribute"):
        HiddenParameterAttack().configuration_parameters()
