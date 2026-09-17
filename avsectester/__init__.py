"""AVSecTester — adversarial security-testing framework for autonomous-vehicle systems.

The framework is two roles joined by a pure data plane, so **any** world backend mixes with **any**
AV stack and an attack is a transform on the stream between them:

* :mod:`avsectester.plane`   — the sim<->stack contract, pure data: ``Observation`` (down) and
  ``Control`` (up), plus the ``Trace``/``FrameRecord`` driving record. No avstack/torch imports.
* :mod:`avsectester.backend` — the interfaces: ``WorldBackend`` (``reset``/``step(control)``) and
  ``AVStack`` (``__call__(obs) -> Control``), and ``run(backend, stack, frames, perturb=None)``
  which drives the loop; ``perturb`` is the single universal attack seam.

The **2×2** of interchangeable implementations:

* World backends — ``CarlaBackend`` (:mod:`avsectester.scenario`; real avcarla closed loop) and
  ``NuRecBackend`` (:mod:`avsectester.simulators`; in-process NVIDIA NuRec neural reconstruction).
* AV stacks — ``ModularAVStack`` (:mod:`avsectester.scenario`; an avstack ``ModularDrivingPipeline``)
  and ``AlpamayoAVStack`` (:mod:`avsectester.stacks`; the real end-to-end Alpamayo-1.5 policy).

Plus :mod:`avsectester.attacks` (avstack ``HOOKS`` hooks — the modular white-box seam),
:mod:`avsectester.metric` (clean-vs-attacked driving-impact verdict), :mod:`avsectester.viz` (the
impact plot), and :mod:`avsectester.simulators.viz` (per-simulation scene views). The AV modules,
geometry, sensors, CARLA bridge, and the Alpamayo model come from avstack/avcarla/alpasim_driver;
AVSecTester adds only the interface, the attack/metric seams, and the two new backend/stack halves.
"""

__version__ = "0.1.0"
