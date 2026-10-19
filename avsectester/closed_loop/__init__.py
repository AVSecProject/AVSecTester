"""The closed loop: a simulator and a driving system, stepped by one runner.

Modules: ``datatypes`` (Observation, Control, WorldSnapshot), ``simulator`` (Simulator,
SensorView) with implementations in ``simulators``, ``avstack`` (AVStack) with implementations in
``avstacks``, ``vehicle_io``, ``hooks`` (stage hooks), ``runner`` (ClosedLoopRunner), and the
metabuilder and offline-phase access in ``services``, ``driving_adapters``, ``frame_data`` and
``resources``.

This package imports none of them: datasets and the legacy scenario package import closed-loop
modules while they initialize, so eager imports here would be circular. Import from the modules.
"""
