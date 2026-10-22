"""The attack: the user's ``Attack`` (metabuilder and offline phase) and its online ``AttackRuntime``.

:mod:`~avsectester.attack.attack` holds ``Attack``, ``AttackSegment`` and ``OfflineContext``;
:mod:`~avsectester.attack.runtime` holds ``AttackRuntime``, ``OnlineContext`` and the attack
actions; :mod:`~avsectester.attack.segments` holds the ``split_segments`` helper for
``build_metadata``. The user-facing names are re-exported here.
"""

from avsectester.attack.attack import Attack, AttackSegment, OfflineContext
from avsectester.attack.segments import split_segments
from avsectester.attack.runtime import (
    AttackAction,
    AttackRuntime,
    AVStackAttackAction,
    InsertObject,
    OnlineContext,
    ResultSubscription,
    SensorAttackAction,
    WorldAttackAction,
)

__all__ = [
    "AVStackAttackAction",
    "Attack",
    "AttackAction",
    "AttackRuntime",
    "AttackSegment",
    "InsertObject",
    "OfflineContext",
    "OnlineContext",
    "ResultSubscription",
    "SensorAttackAction",
    "WorldAttackAction",
    "split_segments",
]
