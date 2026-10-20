# Contributing

> **Audience:** Code, test and documentation contributors.
>
> **Outcome:** Follow ownership rules and run the required checks.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## Design principles

- Reuse native simulator actors, sensors, models and registries rather than reimplementing them.
- Keep one ClosedLoopRunner and one RunLog recorder for all experiments and candidate rollouts.
- Separate physical state from model-visible observations and commands.
- Keep selection, offline preparation and online deployment in one Attack workflow.
- Store online history in a fresh AttackRuntime per run; pass artifacts explicitly.
- Keep Dataset stateless; generated frame data belongs to preparation caches.
- Preserve dependency direction: framework packages do not import concrete attacks.
- Put shared camera/pose geometry in geometry/, image transport in closed_loop/datatypes.py,
  sensor transforms in closed_loop/sensor_transforms.py and driving scorers beside AVStack.
- Keep world and sensor environments explicit. Do not silently approximate one with the other.
- Reject unsupported capabilities before acquiring native resources when possible.
- Keep model, SDK and GPU imports lazy outside shared interface/data modules.

## Repository layout

| Path | Responsibility |
|---|---|
| avsectester/datasets/ | Dataset records and CARLA/NuRec/nuScenes providers |
| avsectester/attack/ | Attack, metadata, runtime/effects, placement and segment helper |
| avsectester/attack/insertion/ | Reusable insertion rendering, visibility and harmonization |
| avsectester/attack/optimization/ | Reusable offline optimization algorithms |
| avsectester/attack/implemented_attacks/ | One folder per concrete Attack |
| avsectester/closed_loop/ | Simulator/runtime, AVStack, runner, services, hooks and environment |
| avsectester/closed_loop/simulators/ | CARLA, NuRec and recorded playback implementations |
| avsectester/closed_loop/avstacks/ | Modular, Alpamayo, camera-forward-collision, native hooks, mmdet3d.py and model scorers |
| avsectester/geometry/ | Canonical camera and pose geometry |
| avsectester/experiment/ | Configuration, preparation, execution and persistence |
| avsectester/runlog/ | Step/event records, codecs and Parquet persistence |
| avsectester/evaluation/ | Log-only metrics, reports and visualization |
| configs/ | Supplied experiment configurations |
| scripts/ | Preparation, validation and visualization tools |
| tests/ | Core, avstack and explicitly selected live groups |
| third_party/ | External source submodules |

## Add a component

| Component | Contract and guide |
|---|---|
| Simulator | Simulator + SimulatorRuntime; [Simulators](../guides/simulators.md#write-a-custom-simulator-adapter) |
| Driving system | AVStack class/function via avstack.type; [Driving systems](../guides/driving-systems.md#use-a-custom-driving-system) |
| Offline component/scorer | DrivingAdapter bindings; [Driving systems](../guides/driving-systems.md#components-and-scorers-for-offline-work) |
| Attack | Attack + fresh AttackRuntime; [Attacks](../guides/attacks.md) |
| Dataset | Stateless Dataset methods; [Scenarios](../guides/scenarios.md) |
| Sensor environment | apply(rgb, rng); [Environments](../guides/environments.md#custom-sensor-environments) |
| World setting | Simulator declarations/value validation; [Native CARLA settings](../guides/environments.md#native-carla-settings) |
| Metric | Metric.evaluate over ComparisonGroup; [Evaluation](../guides/evaluation.md#add-a-metric) |

Every added component needs tests for its observable behavior and documentation at the owning
interface. Preserve effect timing, caller-owned resource cleanup and fresh per-run histories.

## Tests and CI

```bash
python -m ruff check avsectester scripts tests
python -m mypy
NO_ALBUMENTATIONS_UPDATE=1 python -m pytest tests/core -q
NO_ALBUMENTATIONS_UPDATE=1 python -m pytest tests/avstack -q
```

The core tests need CPU dependencies; avstack tests additionally need native avstack packages and
the CARLA Python client. Neither offline group starts a simulator. Ruff checks E9 and F; mypy uses
the explicit file list in pyproject.toml. See [Tests](../../tests/README.md).

For image dependencies, mount the checkout into the prepared Docker image:

```bash
docker run --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/app -w /app \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e MPLBACKEND=Agg \
  -e NO_ALBUMENTATIONS_UPDATE=1 -e NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint python avsectester:latest \
  -m pytest tests/core tests/avstack -q -p no:cacheprovider
```

The image supplies dependencies; the checkout supplies code. Rebuild after dependency changes. These
offline checks do not load model checkpoints. For neural inference in a standalone container with an
overridden entrypoint, also expose the existing detector model tree with the read-only mount in
[Devices, mounts and rebuilds](../getting-started/installation.md#devices-mounts-and-rebuilds). The
Compose entrypoint normally performs that model-path setup; overriding it bypasses the step.
[CI](../../.github/workflows/ci.yml) covers core tests, avstack tests, lint and Parquet recording
with only runlog-python312 running under Python 3.12. Live physics, rendering RPC/model
compatibility and attack effectiveness need separate environment validation. Report missing
prerequisites as unverified, not passed.

## Documentation rules

| Type | Location | Content |
|---|---|---|
| Getting started | getting-started/ | Complete steps to a working result |
| Concepts | concepts/ | Ownership, data flow and terms |
| Guides | guides/ | Tasks, common cases and links to full reference |
| Examples | examples/ | Concrete supplied Attack workflow and configuration |
| Reference | reference/ | Lookup tables, signatures, defaults and constraints |
| Development | development/ | Contributor rules and open work |

Check API names, signatures, YAML keys, defaults and paths against source. Each fact has an owner
document; link to it instead of maintaining parallel tables. Use Simulator for configuration,
SimulatorRuntime for a live world, AVStack for driving, DrivingAdapter for component/scorer
bindings, HookSet for hooks, metabuilder for attack selection and environment for world/sensors.

Mark unavailable capabilities with `**Not yet implemented.**` at the point where they appear. Link
to a matching TODO item when one exists. Do not describe planned capabilities as current.

Use English, sentence-case headings and roughly 100-character prose lines. Begin task pages with
Audience, Outcome and Prerequisites. End guides with Next steps. Update links when headings move.
Self-contained offline examples carry `<!-- runnable -->` immediately before the code fence and are
checked by tests/core/test_docs.py: Python examples are executed; YAML examples must be either an
`environment:` mapping, validated like an experiment environment, or a complete experiment
configuration, loaded with load_configuration. Other runnable blocks fail the test.
External-service examples state prerequisites; they must still parse against the current API. Do
not make every example non-runnable to avoid verification.

## Roadmap

[TODO](TODO.md#todo) lists only incomplete capabilities. Scientific effect-level validation is
distinct from workflow execution and offline regression tests.
