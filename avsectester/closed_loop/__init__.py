"""The closed loop: a simulator and a driving system, stepped by one runner.

Modules: ``datatypes`` (Observation, Control, WorldSnapshot), ``simulator`` (Simulator,
SensorView) with implementations in ``simulators``, ``avstack`` (AVStack) with implementations in
``avstacks``, ``vehicle_io``, ``hooks`` (stage hooks), ``runner`` (ClosedLoopRunner), and the
metabuilder and offline-phase access in ``services``, ``driving_adapters``, ``frame_data`` and
``resources``.

Import public types from their modules. Keeping this package initializer lightweight avoids
loading simulator and driving-system implementations when only data contracts are needed.
"""
