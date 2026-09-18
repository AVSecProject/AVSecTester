"""CARLA closed-loop security experiment — compose a CarlaBackend with a ModularAVStack and drive it.

The implementations now live with their interface category — the world backend in
:mod:`avsectester.simulators.carla`, the AV stack in :mod:`avsectester.stacks.modular` — and this
module is the CARLA-specific *orchestration* that wires them: :func:`run_scenario` runs one pass (clean
or attacked) and :func:`prepare_scenario` resolves random choices once for a paired run. Running a
scenario clean and then attacked, and diffing the driving record, is the whole test
(see :mod:`avsectester.metric`).

``CarlaBackend``, ``ModularAVStack``, ``prepare_scenario``, ``set_perception_gpu`` and ``_spawn_config``
are re-exported here for backward compatibility; import them from their home modules in new code.
"""

from __future__ import annotations

# Re-imported here so callers/tests can reach them as scenario.* (they are the SAME registry/module
# objects the relocated CarlaBackend/ModularAVStack use, so patching a shared object's .build/.sleep
# reaches those). This makes the orchestration module require the CARLA stack, as before the split.
import secrets  # noqa: F401
import time  # noqa: F401
from copy import deepcopy

from avcarla.config import CARLA  # noqa: F401  (tests patch scenario.CARLA.build)
from avstack.config import (  # noqa: F401  (tests patch scenario.{HOOKS,PIPELINE}.build)
    HOOKS,
    PIPELINE,
)

from avsectester.backend import run
from avsectester.plane import Trace
from avsectester.simulators.carla import (
    CarlaBackend,
    _spawn_config,
    prepare_scenario,
    set_perception_gpu,
)
from avsectester.stacks.modular import ModularAVStack

__all__ = [
    "CarlaBackend",
    "ModularAVStack",
    "_spawn_config",
    "prepare_scenario",
    "run_scenario",
    "set_perception_gpu",
]


def run_scenario(
    scenario: dict,
    attacks: list[dict] | None = None,
    frames: int = 40,
    settle_iters: int = 100,
    patches: list[dict] | None = None,
) -> Trace:
    """Run the CARLA + modular demo through the generic interface and return the driving Trace.

    Assembles a :class:`~avsectester.simulators.carla.CarlaBackend` (world + ego + traffic) and a
    :class:`~avsectester.stacks.modular.ModularAVStack` (the AV box), attaches any modular ``attacks``
    as hooks on the stack's pipeline, and drives them with :func:`avsectester.backend.run`. ``patches``
    are world-level physical-patch attacks applied by the backend at reset (pass them only for the
    attacked run). ``replay_scenario`` (actual spawn transforms) is carried on the returned Trace for a
    paired run; strict-spawn replay is always used.
    """
    scenario = deepcopy(scenario)
    backend = CarlaBackend(scenario, settle_iters=settle_iters, patches=patches)
    stack = ModularAVStack(scenario["ego"]["pipeline"])
    for atk in attacks or []:
        stack.attach(atk["stage"], atk["hook"])
    stack.attach_counter()
    try:
        trace = run(backend, stack, frames)
        trace.replay_scenario = backend.replay_scenario
        for record, n in zip(trace.records, stack.detection_counts):
            record.n_detections = n
    finally:
        backend.close()
    return trace
