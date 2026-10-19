"""The attack: the user's ``Attack`` (metabuilder and offline phase) and its online ``AttackRuntime``.

:mod:`~avsectester.attack.attack` holds ``Attack``, ``AttackSegment`` and ``OfflineContext``;
:mod:`~avsectester.attack.runtime` holds ``AttackRuntime``, ``OnlineContext`` and the attack
effects; :mod:`~avsectester.attack.segments` holds the ``split_segments`` helper for
``build_metadata``. The user-facing names are re-exported here.
"""

from avsectester.attack.attack import Attack, AttackSegment, OfflineContext
from avsectester.attack.segments import split_segments
from avsectester.attack.runtime import (
    AttackEffect,
    AttackRuntime,
    AVStackAttackEffect,
    OnlineContext,
    ResultSubscription,
    SensorAttackEffect,
    WorldAttackEffect,
)

__all__ = [
    "AVStackAttackEffect",
    "Attack",
    "AttackEffect",
    "AttackRuntime",
    "AttackSegment",
    "OfflineContext",
    "OnlineContext",
    "ResultSubscription",
    "SensorAttackEffect",
    "WorldAttackEffect",
    "split_segments",
]
