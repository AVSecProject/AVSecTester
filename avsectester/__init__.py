"""Adversarial testing for autonomous driving, organized by internal dependency layer.

* ``geometry``, ``validation``, ``type_paths`` and ``serialization`` provide low-level
  mathematics, checks, callable imports, JSON serialization and independent snapshots.
* ``data_types`` contains shared observations, commands, states, detections and data targets.
* ``datasets``, ``maps`` and ``rendering`` provide source data, semantic maps and object rendering.
* ``closed_loop.simulators`` and ``closed_loop.avstacks`` export their interfaces; concrete
  simulators and driving systems live in implementation modules. Their ``base`` modules own the
  simulator/runtime and AVStack/ObservationPipeline contracts. ``closed_loop`` also provides stage
  hooks and environments; driving-component declarations live in ``avstacks.driving_adapter``.
* ``attack`` defines attack interfaces and reusable insertion and optimization tools;
  ``runlog`` records and reads closed-loop traces.
* ``execution`` owns the runner, isolated services, attack dispatch and frame-data cache.
* ``implemented_attacks`` composes those interfaces and tools into concrete attacks.
* ``evaluation`` computes metrics and formats reports and visualizations from recorded data.
* ``experiment`` loads configuration, prepares attacks and executes treatment comparisons.
* ``cli`` exposes experiment execution and analysis commands.

Internal imports point to the same or a lower layer. The sole upward type-only reference is
from the Attack interface to the execution services supplied to its user callbacks. Optional
native and model dependencies retain their lazy imports in the adapters that use them.
"""

__version__ = "0.1.0"
'Framework package version.'


__all__ = [
    '__version__',
]
