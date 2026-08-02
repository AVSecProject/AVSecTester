"""Closed-loop vulnerability search (PLAN.md Phase 7, proposal Thrust 3).

Strategies implement core.interfaces.SearchStrategy and register with the SEARCH registry.
The fitness signal is the activation/escalation detector from avsectester.monitors.

TODO(phase7): EvolutionaryFuzzer over joint (scenario x attack) parameters; later, a
generative-scenario hook (e.g. NVIDIA AlpaSim) behind the same interface.
"""

__all__: list = []
