"""AVSecTester — adversarial security-testing framework for autonomous-vehicle systems.

Backend and stack adapters exchange observations and controls. Compatible sensor, state and
control representations can be composed through the same driving loop:

* :mod:`avsectester.plane`   — the sim<->stack contract, pure data: ``Observation`` (down) and
  ``Control`` (up), plus the ``Trace``/``FrameRecord`` driving record. No avstack/torch imports.
* :mod:`avsectester.backend` — the interfaces: ``WorldBackend`` (``reset``/``step(control)``) and
  ``AVStack`` (``__call__(obs) -> Control``), and ``run`` which drives the loop.
* :mod:`avsectester.runtime` — ordered stage handlers and optional per-run lifecycle for attacks,
  defenses and localization adapters. Only stages supported by the selected adapters can be used.

The supplied backend and stack adapters:

* World backends (:mod:`avsectester.simulators`) — ``CarlaBackend`` (``simulators.carla``; real
  avcarla closed loop) and ``NuRecBackend`` (``simulators.nurec``; local ego dynamics with
  reconstructed views from an external rendering service).
* AV stacks (:mod:`avsectester.stacks`) — ``ModularAVStack`` (``stacks.modular``; an avstack
  ``ModularDrivingPipeline``) and ``AlpamayoAVStack`` (``stacks.alpamayo``; end-to-end Alpamayo-1.5).
* Composition — :mod:`avsectester.scenario` wires a CARLA backend + modular stack into one run.
  The CLI pairs clean and attacked runs using the prepared scenario and actual spawn replay.

Plus :mod:`avsectester.attacks` (avstack ``HOOKS`` hooks — the modular white-box seam),
:mod:`avsectester.metric` (clean-vs-attacked driving-impact verdict + its ``plot_impact`` figure), and
:mod:`avsectester.simulators.viz` (per-simulation scene views). The AV modules,
geometry, sensors, CARLA bridge, and the Alpamayo model come from avstack/avcarla/alpasim_driver;
AVSecTester provides the connecting interfaces, scenario selection, insertion geometry,
intervention execution and evaluation.
"""

__version__ = "0.1.0"
