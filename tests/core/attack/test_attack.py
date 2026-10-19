"""The user's attack class: one metabuilder and two phases with trivial defaults."""

import pytest

from avsectester.attack import Attack, OfflineContext
from avsectester.attack.runtime import AttackRuntime


class Phantom(Attack):
    def build_metadata(self, scenario):
        return [scenario]


def test_offline_and_online_defaults_are_trivial():
    attack = Phantom()
    assert attack.run_offline(OfflineContext(services=None, metadata=("segment",))) is None
    attack.assets = {"patch": "texture"}
    assert attack.run_offline(OfflineContext(None, ())) == {"patch": "texture"}
    runtime = attack.create_runtime("segment", attack.assets)
    assert type(runtime) is AttackRuntime
    assert runtime.setup(None) == () and runtime.step(None) == ()


def test_every_attack_builds_its_metadata():
    with pytest.raises(TypeError, match="build_metadata"):
        Attack()
