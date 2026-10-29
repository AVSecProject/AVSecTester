# Supplied examples

> **Audience:** Users choosing a supplied experiment configuration.
>
> **Outcome:** Select a dataset, simulator, driving system and attack workflow.
>
> **Prerequisites:** [Installation](../getting-started/installation.md) and
> [Quickstart](../getting-started/quickstart.md).

Run configurations through Compose after preparing data, weights and the local Compose
environment. See [Writing attacks](../guides/attacks.md) for lifecycle phases.

## Configurations

| Configuration | `Simulator` | Driving system | Data and weights | Example |
|---|---|---|---|---|
| [carla_phantom.yaml](../../configs/carla_phantom.yaml) | CARLA | `ModularAVStack` | CARLA Town03, `fetch_models.sh` | [Phantom injection](phantom-injection.md) |
| [carla_patch.yaml](../../configs/carla_patch.yaml) | CARLA | `CameraForwardCollisionAVStack` | CARLA Town03, `fetch_models.sh` | [Physical patch](physical-patch.md) |
| [carla_patch_optimized.yaml](../../configs/carla_patch_optimized.yaml) | CARLA | `CameraForwardCollisionAVStack` | CARLA Town03, `fetch_models.sh` | [Physical patch](physical-patch.md), white-box attack, algorithm used: `PGD` |
| [carla_object_insertion.yaml](../../configs/carla_object_insertion.yaml) | CARLA | `CameraForwardCollisionAVStack` | CARLA Town03, `fetch_models.sh` | [Image insertion](image-insertion.md) |
| [nurec_patch.yaml](../../configs/nurec_patch.yaml) | NuRec | `AlpamayoAVStack` | `data/nurec`, scene `023b7fcc-671c-40e3-9bd2-c66b0b073fbc`, Alpamayo | [Physical patch](physical-patch.md) |
| [nurec_object_insertion.yaml](../../configs/nurec_object_insertion.yaml) | NuRec | `AlpamayoAVStack` | `data/nurec`, scene `023b7fcc-671c-40e3-9bd2-c66b0b073fbc`, Alpamayo | [Image insertion](image-insertion.md) |
| [nurec_speed_feedback.yaml](../../configs/nurec_speed_feedback.yaml) | NuRec | `AlpamayoAVStack` | `data/nurec`, scene `023b7fcc-671c-40e3-9bd2-c66b0b073fbc`, Alpamayo | [Speed feedback](speed-feedback-attack.md) |
| [nuscenes_object_insertion.yaml](../../configs/nuscenes_object_insertion.yaml) | Recorded playback | `CameraForwardCollisionAVStack` | `data/nuscenes`, `fetch_models.sh` | [Image insertion](image-insertion.md) |
| [nurec_stop_sign_patch_modular_2601.yaml](../../configs/nurec_stop_sign_patch_modular_2601.yaml) | NuRec | `RuleBasedAVStack` | `data/nurec/26.01_release`, COCO YOLOX-m | [Stop-sign patch](stop-sign-patch.md), white-box attack, algorithm used: `PGD` |
| [nurec_stop_sign_patch_modular_2604.yaml](../../configs/nurec_stop_sign_patch_modular_2604.yaml) | NuRec | `RuleBasedAVStack` | `data/nurec/26.04_release`, COCO YOLOX-m | [Stop-sign patch](stop-sign-patch.md), white-box attack, algorithm used: `PGD` |
| [nurec_stop_sign_patch_alpamayo_2601.yaml](../../configs/nurec_stop_sign_patch_alpamayo_2601.yaml) | NuRec | `AlpamayoAVStack` | `data/nurec/26.01_release`, Alpamayo | [Stop-sign patch](stop-sign-patch.md), black-box attack, algorithm used: `NES` |
| [nurec_stop_sign_patch_alpamayo_2604.yaml](../../configs/nurec_stop_sign_patch_alpamayo_2604.yaml) | NuRec | `AlpamayoAVStack` | `data/nurec/26.04_release`, Alpamayo | [Stop-sign patch](stop-sign-patch.md), black-box attack, algorithm used: `NES` |

The `_2601` and `_2604` suffixes select NuRec 26.01 and 26.04. In stop-sign configuration names,
`modular` selects `RuleBasedAVStack`.

Run commands from the repository root. YAML paths are relative to the YAML file. Use the running
CARLA experiment service or start an ephemeral NuRec experiment container:

```bash
docker compose exec carla-experiment avsectester run configs/carla_phantom.yaml
docker compose run --rm nurec-experiment run configs/nurec_patch.yaml
```

See [Quickstart](../getting-started/quickstart.md) for first-run startup and shutdown steps.
[Experiment configuration](../reference/configuration.md) describes path resolution and defaults.

## Attack impact

Attack impacts and driving results must be measured in the target environment. Workflow tests
establish preparation, execution and evaluation behavior. Report scientific impact separately
using the [evaluation workflow](../guides/evaluation.md).

**Not yet implemented.** Published effectiveness claims for supplied attacks. See
[Attack impact](../development/TODO.md#attack-impact).
