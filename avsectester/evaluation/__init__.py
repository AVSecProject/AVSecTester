"""Evaluation layer — run attacks over qualifying scenarios and score them.

:mod:`~avsectester.evaluation.robustness` is the attack-robustness harness: for each scenario a
:class:`~avsectester.scenarios.source.ScenarioSource` yields, it runs the attack under a suite of sensor
corruptions (:mod:`avsectester.simulators.augment`) and scores each with
:func:`avsectester.metric.impact`, reporting how the attack's success rate degrades with corruption.
"""

from .robustness import RobustnessReport, evaluate_robustness

__all__ = ["RobustnessReport", "evaluate_robustness"]
