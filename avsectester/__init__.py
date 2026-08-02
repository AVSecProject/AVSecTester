"""AVSecTester — closed-loop adversarial stress-testing framework for AV systems.

AVSecTester builds its *security* layer on top of `avstack` (avstack-lab):

    - avstack-core (`avstack`)      : reconfigurable AV modules, geometry, sensors,
                                      registry/config, pipeline + hooks, RSS metric.
    - lib-avstack-carla (`avcarla`) : closed-loop CARLA bridge.
    - avstack-api (`avapi`)         : KITTI / nuScenes / CARLA dataset adapters.

AVSecTester adds: attacks, defenses, runtime instrumentation, the attack-escalation
DAG, security metrics, closed-loop vulnerability search, root-cause analysis and audit
reporting, and agent-assisted integration workflows.

Design principle (attacks/defenses/monitors are non-invasive): they attach to avstack
modules as **pre/post hooks** (the `HOOKS` registry + `@apply_hooks`), so we never fork
avstack internals to intercept data.
"""

__version__ = "0.0.1"
