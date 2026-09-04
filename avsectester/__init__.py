"""AVSecTester — adversarial security-testing framework for autonomous-vehicle systems.

A thin security layer on top of avstack (avstack-lab). The AV stack, geometry, sensors, CARLA
bridge, and dataset adapters all come from avstack/avcarla; AVSecTester adds only:

* :mod:`avsectester.scenario` — build a real closed-loop CARLA scenario from config (an avcarla
  ``CarlaMobileActor`` running an avstack ``ModularDrivingPipeline``) and drive it;
* :mod:`avsectester.attacks`  — attacks as avstack ``HOOKS`` hooks on pipeline stages;
* :mod:`avsectester.metric`   — diff a clean vs attacked run into a driving-impact verdict.

No parallel environment/system/attack abstractions and no mock: the framework uses avstack's own
interfaces directly.
"""

__version__ = "0.0.1"
