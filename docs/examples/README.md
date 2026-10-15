# Examples

> **Audience:** Researchers who want to see how existing attacks use the framework's interfaces.  
> **Outcome:** Understand how each supplied attack divides its work between preparation and execution, where it acts and how to run it.  
> **Prerequisites:** [Architecture](../concepts/architecture.md) and [Implement an attack](../guides/attacks.md).

## How to read these examples

The supplied attacks use the same [intervention](../concepts/architecture.md#terminology) interfaces
as attacks you write yourself. Each example is described in the same five parts:

- **What it does:** the attack's effect.
- **Preparation:** the work done before runs and the artifact it produces.
- **Execution:** the runtime stage, the attack surface and the handler that runs there.
- **Configure and run:** the YAML keys or Python calls, and the scripts that exercise the example.
- **Code:** the module paths.

| Example | Stage | Attack surface | Simulator and driving system | Execution |
|---|---|---|---|---|
| [Phantom injection](phantom-injection.md) | `perception.post` | Perception output inside the driving system | CARLA with the modular driving system | Static |
| [Physical patch](physical-patch.md) | `world.setup` | The world: a textured panel on a vehicle | CARLA | Static |
| [Image insertion](image-insertion.md) | `observation` | Camera images in the model input | CARLA and NuRec | Static. Placement follows current poses |
| [Speed-feedback adaptive attack](speed-feedback-attack.md) | `observation` | Reported speed in the model input | Any simulator | Adaptive |

Run commands from the repository root. Simulator, model and weight setup are described in
[Installation](../getting-started/installation.md). [scripts/README.md](../../scripts/README.md) indexes
every script.

## Defenses

No defense example is supplied. Defenses use the same preparation and execution interfaces as the
attacks above, with `kind: defense`. See [Implement a defense](../guides/defenses.md).

**Not yet implemented.** Reference defense algorithms and metrics for recorded defense outputs.
