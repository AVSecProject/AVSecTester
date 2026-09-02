"""AVSecTester — adversarial security-testing framework for autonomous-vehicle systems.

Built as a thin security layer on top of avstack (avstack-lab): the AV stack, geometry,
sensors, CARLA bridge, and dataset adapters come from avstack; AVSecTester adds the minimal
security-testing contract in ``core`` — Frame, Environment, System, Attack/Defense, Metric —
plus attack/defense plugins, environments (mock + CARLA), and metrics.
"""

__version__ = "0.0.1"
