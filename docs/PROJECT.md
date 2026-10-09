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
| Shared driving loop | `WorldBackend` provides observations and executes controls. `AVStack` makes driving decisions. See [Interfaces](INTERFACE.md). |
| CARLA simulation | `CarlaBackend` connects avcarla actors, sensors and reactive NPC traffic to a modular avstack pipeline. |
| Reconstructed scenes | `NuRecBackend` requests NuRec images from an external `nre-ga` service and advances the ego with local planar dynamics. Other actors follow recorded trajectories. It does not run the full AlpaSim simulation or a reactive traffic policy. |
| Driving models | `ModularAVStack` exposes perception, tracking, planning and control. `AlpamayoAVStack` wraps the AlpaSim Alpamayo driver. Compatible input and control adapters are required when combining a backend and stack. |
| Attack and defense execution | Ordered `Runtime` handlers support effective sensor, localization, component, rendering, world and command stages. Stateful handlers have lifecycle callbacks. See [Interventions](INTERVENTIONS.md). |
| Initial-case selection | Dataset filtering and CARLA candidate creation share composable conditions, fixed actor bindings and initial-window checks. Custom filters can access native metadata and provider APIs. See [Scenario selection](SCENARIOS.md). |
| Patches and objects | Explicit world or host-local placements, three orientation modes, shared geometry and visibility evidence. See [Image attacks](IMAGE_ATTACKS.md). |
| Outputs and metrics | Physical driving traces, component snapshots, images/GIFs, a stop-based impact metric and corruption-condition summaries. See [Component logging](COMPONENT_LOGGING.md) and [Robustness evaluation](AUGMENTATION.md). |

The supplied driving paths are CARLA with the modular pipeline and NuRec with Alpamayo.
The Python interfaces also support custom adapters. The YAML CLI currently runs CARLA/modular
clean and attacked pairs, rather than configuring every backend and runtime feature.

## Experiment workflow

1. Install the required environment, services and weights using [Setup](SETUP.md).
2. Define or select an initial case and its actor bindings. Conditions can differ between attacks.
3. Create a backend and a compatible driving stack. Define attack and optional defense handlers
   at the stages those adapters actually expose.
4. Recreate the selected origin for each comparison run. Keep the same mounting definitions
   and identities, with the intervention enabled only in its intended runs.
5. Collect physical traces, component outputs and scene views. Interpret metrics according to
   the experiment's success definition and the simulator's limitations.

Use the [script index](../scripts/README.md) for runnable examples and
[Development](DEVELOPMENT.md) for extension points and repository structure.

## Longer-term goals

The broader project aims to provide complete experiment reports, attack-propagation analysis,
reusable vulnerability records and automation through an AI harness. Component testing,
hardware-in-the-loop, in-vehicle network and V2X adapters are also possible extensions.
These are project goals, not interfaces implemented in the current repository.
