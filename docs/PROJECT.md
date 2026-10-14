# AVSecTester

AVSecTester gives security researchers a common platform for implementing attacks and defenses
and observing their effects on a single autonomous vehicle in closed-loop simulation.

Researchers supply the driving system, scenario requirements and intervention logic. The
framework connects these parts, runs the experiment and captures driving and component outputs.
It does not impose a universal attack-success criterion or require every attack to use the
same scene conditions.

## Current capabilities

| Capability | Implementation and guide |
|---|---|
| Shared driving loop | `WorldBackend` provides observations and executes commands. `AVStack` makes driving decisions. See [Interfaces](INTERFACE.md). |
| CARLA simulation | `CarlaBackend` owns avcarla actors, sensors and reactive NPC traffic. `ModularAVStack` executes the avstack pipeline outside the backend. |
| Reconstructed scenes | `NuRecBackend` requests NuRec images from an external `nre-ga` service and advances the ego with local planar dynamics. Other actors follow recorded trajectories. It does not run the full AlpaSim simulation or a reactive traffic policy. |
| Driving models | `ModularAVStack` exposes perception, tracking, planning and control. `AlpamayoAVStack` wraps the AlpaSim Alpamayo driver. Compatible input and command adapters are required when combining a backend and stack. |
| Attack and defense preparation | `Intervention.prepare(scenario)` creates reusable artifacts. `hooks(artifacts)` constructs each run's handlers. Artifacts can be explicitly saved, loaded and reused. See [Interventions](INTERVENTIONS.md). |
| Attack and defense execution | Ordered `Runtime` handlers support effective sensor, localization, component, rendering, world and command stages. Stateful handlers have lifecycle callbacks. See [Interventions](INTERVENTIONS.md). |
| Scenario preparation and conditions | Unfiltered configurations and selected cases produce a `ScenarioInstance`. `run_prepared` applies shared world/sensor conditions and runs named treatments. |
| Initial-case selection | Dataset filtering and CARLA candidate creation share composable filters, fixed actor bindings and initial-window checks. Custom filters can access native metadata and provider APIs. See [Scenario selection](SCENARIOS.md). |
| Patches and objects | Explicit world or host-local placements, three orientation modes, shared geometry and visibility evidence. See [Image attacks](IMAGE_ATTACKS.md). |
| Outputs and metrics | Physical driving traces, component snapshots, images/GIFs, a stop-based impact metric and corruption-condition summaries. See [Component logging](COMPONENT_LOGGING.md) and [Robustness evaluation](AUGMENTATION.md). |

The supplied driving paths are CARLA with the modular pipeline and NuRec with Alpamayo.
The Python interfaces also support custom adapters. The YAML CLI runs CARLA experiments as single
runs or paired comparisons, with clean, attack, defense or combined treatments. It accepts the
supplied modular stack and user stack factories. NuRec with Alpamayo runs through the demo scripts
in the AlpaSim driver environment.
Native physical panels and modular attacks are adapters into the same runtime.

## Experiment workflow

1. Install the required environment, services and weights using [Setup](SETUP.md).
2. Define or select a scene and its actor bindings, filters and operating conditions. All treatments
   of one experiment share those operating conditions. Separate experiments may use different ones.
3. Prepare attack and defense artifacts, or register runtime handlers directly. Supply a
   compatible driving-stack factory and the intended treatment order.
4. Use `run_configured` for YAML/Python mappings or `run_prepared` for a `ScenarioInstance`.
   Single runs execute once. Comparisons start from the same scene and actor setup, with shared
   conditions, mounting definitions and identities. Interventions are enabled in their intended
   treatments.
5. Collect physical traces, component outputs and scene views. Interpret metrics according to
   the experiment's success definition and the simulator's limitations.

Use the [script index](../scripts/README.md) for runnable examples and
[Development](DEVELOPMENT.md) for extension points and repository structure.

## Longer-term goals

The broader project aims to provide complete experiment reports, attack-propagation analysis,
reusable vulnerability records and automation through an AI harness. Component testing,
hardware-in-the-loop, in-vehicle network and V2X adapters are also possible extensions.
These are project goals, not interfaces implemented in the current repository.
