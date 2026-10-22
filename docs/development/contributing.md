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
- Follow the dependency layers; shared interfaces and reusable tools do not import concrete attacks.
- Put shared camera/pose geometry in geometry/, image transport in data_types.py,
  sensor corruptions in closed_loop/sensor_corruptions.py and driving scorers beside AVStack.
- Keep world and sensor environments explicit. Do not silently approximate one with the other.
- Reject unsupported capabilities before acquiring native resources when possible.
- Keep model, SDK and GPU imports lazy outside shared interface/data modules.

## Repository layout

| Path | Responsibility |
|---|---|
| avsectester/geometry/ | Canonical poses, cameras, boxes, projection and mesh intersections |
| avsectester/validation.py, type_paths.py, serialization.py | Shared validation, type imports, JSON serialization and snapshots |
| avsectester/data_types.py | Shared observations, commands, world/actor states, detections and targets |
| avsectester/datasets/ | Stateless recorded data providers and CARLA recipes |
| avsectester/maps/ | Semantic-map records and clipgt/OpenDRIVE readers |
| avsectester/rendering/ | Object assets, textures, surface sampling, harmonizers and composition |
| avsectester/closed_loop/ | Shared hooks and environments plus simulator/driving-system subpackages |
| avsectester/closed_loop/simulators/base.py | Simulator/runtime interfaces, capture lifecycle and sensor views |
| avsectester/closed_loop/simulators/ego_dynamics.py, avsectester/closed_loop/simulators/object_composition.py | Ego integration and capture composition |
| avsectester/closed_loop/avstacks/base.py, avsectester/closed_loop/avstacks/driving_adapter.py | AVStack, ObservationPipeline and component/scorer contracts |
| avsectester/closed_loop/simulators/carla/ | CARLA simulator, scenarios, sensors, conversions and recipe replay |
| avsectester/closed_loop/simulators/nurec/ | NuRec simulator, renderer and depth |
| avsectester/closed_loop/avstacks/ | Driving implementations, native Modular hooks, registration and model scorers |
| avsectester/attack/ | Attack interfaces, segment metadata and runtime/action contracts |
| avsectester/attack/insertion/ | Insertion actions, assets, placement, visibility and compositing tools |
| avsectester/attack/optimization/ | Generic optimizers, perturbations, objectives and SessionScorer |
| avsectester/runlog/ | Step/event records, payload shards and Parquet readers/writers |
| avsectester/execution/ | Runner, services, attack dispatch and frame cache |
| avsectester/implemented_attacks/ | One folder per concrete Attack, with attack-specific code and textures |
| avsectester/evaluation/ | Metrics, report records, statistics and visualization |
| avsectester/experiment/ | Configuration, preparation, treatments, output layout and evaluation orchestration |
| avsectester/experiment/output_directory.py | Output layout and non-RunLog file reads/writes |
| avsectester/experiment/type_registry.py, configuration_summary.py | Constructor discovery and configuration descriptions |
| avsectester/experiment/result_evaluation.py | Experiment-result evaluation above the standalone evaluation package |
| avsectester/cli.py | Command-line commands and presentation |
| configs/ | Supplied experiment configurations |
| scripts/ | Preparation, validation and visualization tools |
| tests/ | Core, avstack and explicitly selected live groups |
| third_party/ | External source submodules |

Simulator and driving-system package initializers export their interfaces. Import concrete classes
from their own implementation modules.

### Dependency layers

Apply the same rule to module imports, delayed imports and TYPE_CHECKING imports: internal imports
point to the same layer or a lower layer.

| Layer | Modules | Allowed internal imports |
|---|---|---|
| L0 | geometry/, validation.py, type_paths.py, serialization.py | geometry may import validation; serialization may import type_paths |
| L0′ | data_types.py | L0 |
| L1 | datasets/, maps/, rendering/ | L0–L0′; datasets may import maps, maps must not import datasets |
| L2 | closed_loop/ and all simulator/AVStack implementations | L0–L1 |
| L3 | attack/, runlog/ | L0–L2; attack may import runlog |
| L4 | execution/ | L0–L3 |
| L5 | implemented_attacks/ | L0–L4 |
| L6 | evaluation/ | L0–L5 |
| L7 | experiment/ | L0–L6 |
| L8 | cli.py | L0–L7 |

The only upward reference is attack/attack.py using execution.services for ScenarioView and
ClosedLoopServices annotations under TYPE_CHECKING. SerializationRegistry stays at L0 so driving
adapters can snapshot component outputs. String dataclass registrations keep lower layers from
importing the classes they encode. Optional third-party libraries are separate from this internal
table and load lazily in implementations.

Attack core does not import insertion/optimization tools or concrete attacks. Tools never import
each other or concrete attacks, and concrete attacks do not import one another. rendering/ never
imports attack/ or closed_loop/. Keep shared plane texture sampling in rendering/surfaces.py rather
than copying it into either consumer.

## Add a component

| Component | Owning contract and guide |
|---|---|
| Simulator | closed_loop.simulators.base.Simulator + SimulatorRuntime; [Simulators](../guides/simulators.md#write-a-custom-simulator-adapter) |
| Driving system | closed_loop.avstacks.base.AVStack via avstack.type; [Driving systems](../guides/driving-systems.md#use-a-custom-driving-system) |
| Observation processing | closed_loop.avstacks.base.ObservationPipeline via avstack.observation_pipeline; [Driving specification](../reference/configuration.md#driving-specification) |
| Offline component/scorer | closed_loop.avstacks.driving_adapter.DrivingAdapter bindings; [Driving systems](../guides/driving-systems.md#components-and-scorers-for-offline-work) |
| Attack | attack.attack.Attack + attack.runtime.AttackRuntime; place implementations in implemented_attacks/ or your own package; [Attacks](../guides/attacks.md) |
| Dataset | datasets.base.Dataset; [Scenarios](../guides/scenarios.md) |
| Map reader | maps.scenario_map.ScenarioMap returned by Dataset.scenario_map; [Semantic maps](../guides/scenarios.md#semantic-maps) |
| Sensor corruption | closed_loop.sensor_corruptions.Corruption or apply(rgb, rng); [Environments](../guides/environments.md#custom-sensor-corruptions) |
| World setting | closed_loop.simulators.base.Simulator declarations and value validation; [Native CARLA settings](../guides/environments.md#native-carla-settings) |
| Object asset or harmonizer | rendering.objects.ObjectAsset and rendering.harmonizers.Harmonizer; [World objects](../guides/attacks.md#insert-a-world-object) |
| Metric | evaluation.metric.Metric.evaluate over ComparisonGroup; [Evaluation](../guides/evaluation.md#add-a-metric) |

Every added component joins its family's template test, gets tests for behavior unique to it and
is documented at the owning interface. Preserve action timing, caller-owned resource cleanup and
fresh per-run histories.

Filesystem constructor parameters must use Path, str | Path or an os.PathLike annotation, including
sequence element annotations when appropriate. Keep names in Path-containing annotations available
at runtime. The loader resolves these options for all type/options constructors; it does not infer
paths from parameter names or ordinary nested dictionaries. See
[filesystem parameters](../reference/configuration.md#constructor-filesystem-parameters).

## Tests and CI

```bash
python -m ruff check avsectester scripts tests
python -m mypy
NO_ALBUMENTATIONS_UPDATE=1 python -m pytest tests/core -q
NO_ALBUMENTATIONS_UPDATE=1 python -m pytest tests/avstack tests/core/closed_loop/avstacks \
  tests/core/implemented_attacks/test_implemented_attacks.py -q
```

The core tests need CPU dependencies; avstack tests additionally need native avstack packages and
the CARLA Python client. Neither offline group starts a simulator. Ruff checks E9 and F; mypy uses
the explicit file list in pyproject.toml.

Write as many tests as you need to verify a change, but keep them outside the repository. Commit
only tests that follow the test design in [Writing tests](../../tests/README.md#writing-tests).

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
compatibility and attack impact need separate environment validation. Report missing
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
checked by tests/core/test_docs.py: Python examples are executed; runnable YAML is a complete
experiment configuration loaded with load_configuration. Other runnable blocks fail the test.
External-service examples state prerequisites; they must still parse against the current API. Do
not make every example non-runnable to avoid verification.

## Roadmap

[TODO](TODO.md#todo) lists only incomplete capabilities. Scientific attack-impact validation is
distinct from workflow execution and offline regression tests.
