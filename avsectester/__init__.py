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
* :mod:`avsectester.evaluation` — metrics, reports and visualization from RunLog.
* :mod:`avsectester.experiment` — configuration, preparation and treatment execution.

Concrete attacks live in :mod:`avsectester.attack.implemented_attacks`. Reusable insertion and
optimization tools live in :mod:`avsectester.attack.insertion` and
:mod:`avsectester.attack.optimization`. Attack interfaces do not import those tools or concrete
implementations; concrete attacks compose the framework interfaces and tools.
"""

__version__ = "0.1.0"
