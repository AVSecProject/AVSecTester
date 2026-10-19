"""AVSecTester — adversarial security-testing framework for autonomous-vehicle systems.

The framework packages follow its components:

* :mod:`avsectester.datasets` — scenarios and per-frame data from recorded datasets or CARLA
  recipes, and the start data a simulator needs. A dataset never depends on a simulator.
* :mod:`avsectester.attack` — the user's ``Attack`` (metabuilder and offline phase),
  ``AttackSegment`` records, and the online ``AttackRuntime`` with its attack effects.
* :mod:`avsectester.closed_loop` — the closed loop: data types, the ``Simulator`` base and the
  supplied simulators (``simulators``), the ``AVStack`` interface and the supplied driving
  systems (``avstacks``), Vehicle I/O, stage hooks, the ``ClosedLoopRunner``, and the closed-loop
  services used by the metabuilder and the offline phase.
* :mod:`avsectester.runlog` — the run log and its Parquet persistence.
* :mod:`avsectester.evaluation` — driving-impact metric, component logging and robustness runs.

Attack implementations and their helpers live in :mod:`avsectester.attacks`,
:mod:`avsectester.rendering`, :mod:`avsectester.perturbations` and :mod:`avsectester.insertion`.

The legacy experiment path used by the CLI remains until phase 4: ``backend`` (the legacy loop
``run``), ``preparation``, ``interventions``, ``runner``, ``carla_runner``, ``conditions``,
``experiment`` and ``scenarios``. The AV modules, geometry, sensors, CARLA bridge and the Alpamayo
model come from avstack, avcarla and alpasim_driver.
"""

__version__ = "0.1.0"
