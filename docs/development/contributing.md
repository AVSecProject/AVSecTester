# Contributing

> **Audience:** Contributors changing code, tests or documentation.  
> **Outcome:** Follow the design principles, find code by component, run tests and write documentation consistently.  
> **Prerequisites:** [Architecture](../concepts/architecture.md).


## Design principles

- **No parallel definitions.** Adapters reuse the provider's actors, sensors, model components,
  configuration registries and native hooks. They do not reimplement simulator actors or
  driving-model modules. This repository owns dataset/experiment orchestration, the attack workflow,
  resource isolation and the messages exchanged between the simulator and the driving system. When a
  missing capability belongs to avstack, add it to the avstack fork.
- **One driving loop over plain data types.** `closed_loop/datatypes.py` defines the messages
  (`Observation`, `Control`, `WorldSnapshot`) and imports no simulator or model package.
  `closed_loop/simulator.py` defines the simulator interface (`Simulator`, `SimulatorRuntime`),
  `closed_loop/avstack.py` the driving-system interface (`AVStack`), and `ClosedLoopRunner` in
  `closed_loop/runner.py` drives the loop. Paired runners, component logging and scene recording
  all drive this same loop.
- **Physical state stays separate from model-visible data.** Handlers that change the observation
  never move the vehicle. World changes use explicit simulator operations. See
  [Physical state and model-visible data](../concepts/architecture.md#physical-state-and-model-visible-data).
- **One attack workflow.** Implement `Attack.build_metadata`, optional `run_offline` and
  `create_runtime`. Keep online state in one fresh `AttackRuntime`, with world, sensor and
  AVStack effects applied by the common runner. Reuse that deployment in candidate rollouts.
  Existing concrete attacks and defenses retain their compatibility stage-hook path.
- **Datasets do not own execution.** Recorded data and recipes belong to datasets. Generated
  frame data belongs to preparation caches. Simulators receive `ScenarioStart`, never attack
  selection records.
- **Keep dependencies directed.** Framework interfaces must not import concrete attacks or
  their insertion/compositing implementations. Shared cameras and actor poses belong to
  `geometry/`. Sensor image transport belongs to `closed_loop/datatypes.py`. Model scorers
  belong beside the driving adapter.
- **Environment is explicit.** Resolve experiment base, enabled attack settings and treatment
  overrides in that order. World settings stay world settings. Sensor transforms apply after
  attack sensor effects. Do not silently substitute one for the other.
- **Fail before the world starts.** Unsupported stages and incompatible command modes raise
  errors before the simulator resets. Adapters declare the stages and command modes they support.
  The interfaces never convert between incompatible sensor, state or command formats.
- **Keep heavy imports lazy.** The core modules import without PyTorch, avstack, CARLA or
  AlpaSim. Adapters such as `AlpamayoAVStack` import their dependencies on first use, so the
  offline tests run in the [local core installation](../getting-started/installation.md#local-core-installation).

## Repository layout

Paths are relative to the repository root. Component names follow
[Components](../concepts/architecture.md#components). The complete module
list with public names is in [Module map](../reference/python-api.md#module-map).

| Path | Component | Contents |
|---|---|---|
| `avsectester/datasets/` | Dataset | `Dataset`, `FrameReference`, `FrameData`, `ScenarioStart`; NuRec, nuScenes and CARLA datasets |
| `avsectester/attack/` | Attack | `Attack`, `AttackSegment`, `OfflineContext` (`attack.py`); `AttackRuntime`, `OnlineContext` and the attack effects (`runtime.py`) |
| `avsectester/experiment/` | Experiment | Three configuration types, metadata selection, offline preparation, persistence and independent treatment execution |
| `avsectester/geometry/` | Shared geometry | Camera models, canonical actor poses and pose transforms |
| `avsectester/closed_loop/` | Closed loop | `datatypes.py` (`Observation`, `Control`, `WorldSnapshot`, `Trace`), `simulator.py` (`Simulator`, `SimulatorRuntime`, `SensorView`, legacy `WorldBackend`), `avstack.py` (`AVStack`), `vehicle_io.py`, `hooks.py` (`HookSet`, `Runtime`, `Hook`, `Plugin`, contexts), `runner.py` (`ClosedLoopRunner`), `services.py` and `driving_adapters.py` (closed-loop services), `frame_data.py`, `environment.py`, `resources.py` |
| `avsectester/closed_loop/simulators/` | Closed loop (simulator) | `CarlaSimulator`, `NuRecSimulator`, `RecordedReplaySimulator` and their runtimes (`CarlaSimulatorRuntime`, `NuRecSimulatorRuntime`, `RecordedReplaySimulatorRuntime`), renderers and ego dynamics, scene visualization |
| `avsectester/closed_loop/avstacks/` | Closed loop (driving system) | `ModularAVStack`, `AlpamayoAVStack`, CARLA dataset registration, detector scorers |
| `avsectester/runlog/` | Run log | `RunLog`, `Recorder` and readers (`log.py`); native-object codecs (`codecs.py`) |
| `avsectester/evaluation/` | Evaluation | Driving-impact metric and plot (`metric.py`), component logging, robustness evaluation |
| `avsectester/attacks/` | Attack implementations | Phantom injection (`pipeline/`), physical patch (`patch/`), object-insertion payloads (`object_insertion/`), speed-feedback attack (`adaptive.py`) |
| `avsectester/attacks/optim/` | Attack implementations (optimization) | PGD, NES, perturbations, scorers, data sources, CARLA-rendered perturbation |
| `avsectester/insertion.py`, `avsectester/rendering/`, `avsectester/perturbations/` | Attack implementations (helpers) | Insertion specification, compositing, harmonizers, visibility and legacy rendering/perturbation adapters |
| `avsectester/backend.py` | Compatibility loop entry point | `run`, still called by the legacy experiment path and evaluation |
| `avsectester/runner.py`, `legacy_experiment.py`, `carla_runner.py`, `preparation.py`, `interventions.py`, `conditions.py`, `scenarios/` | Compatibility experiment path | Named treatments (`run_prepared`, `run_pair`), configured experiments (`run_configured`), CARLA assembly, `Intervention`, `ScenarioConditions`, scenario selection |
| `avsectester/cli.py` | Experiment | The `avsectester` command |
| `configs/` | Closed-loop experiment | Supplied CARLA scenario configurations |
| `scripts/` | All | Demos, model and asset preparation, optimization, validation and visualization entry points. See [scripts/README.md](../../scripts/README.md) |
| `tests/` | All | `core`, `avstack` and `live` test groups. See [tests/README.md](../../tests/README.md) |
| `third_party/` | Simulator, driving system | Git submodules: the avstack projects and `libcom` |

Each framework component is a package whose base interfaces sit next to their implementations,
for example `closed_loop/simulator.py` and `closed_loop/simulators/`. Shared camera/pose geometry
lives in `geometry/`. Sensor-environment structure lives in `closed_loop/environment.py`; image
transforms live in `closed_loop/sensor_transforms.py`. Concrete insertion rendering and harmonization
remain in `rendering/`
until their attack implementations migrate. Keep those helpers out of framework dependencies.

## Add a component

Each extension point is documented in the guide that owns it. Start from that section.

| To add | Implement | Guide |
|---|---|---|
| Simulator | A `Simulator` and its `SimulatorRuntime` | [Write a custom simulator adapter](../guides/simulators.md#write-a-custom-simulator-adapter) |
| Driving system | An `AVStack` class or factory selected by `avstack.type` | [Use a custom driving system](../guides/driving-systems.md#use-a-custom-driving-system) |
| Component outputs of a driving system | The component-logging stack interface | [Component logging interface](../guides/driving-systems.md#component-logging-interface) |
| Attack | `Attack` and a fresh `AttackRuntime` returning attack effects | [Write your first attack](../guides/attacks.md#write-your-first-attack) |
| Defense (legacy path) | An `Intervention` configured with `kind: defense` | [Package and configure a defense](../guides/defenses.md#package-and-configure-a-defense) |
| World condition | A declared `supported_world_settings` key and its simulator implementation | [World conditions](../guides/conditions.md#native-carla-settings) |
| Sensor condition | An environment sensor transform, selected by `type` (a built-in class name or `package.module:name`) | [Custom sensor conditions](../guides/conditions.md#experiment-environment) |
| Attack selection | `Attack.build_metadata`, returning segments and recording reasons | [Use the selected scenario](../guides/attacks.md#use-the-selected-scenario) |
| Dataset | Stateless scenarios, frame data, native metadata and `ScenarioStart` | [Scenario preparation](../concepts/architecture.md#scenario-preparation) |
| Inserted asset | An asset with `planes()`, placed by an `Insertion` | [Specify inserted objects](../guides/scenarios.md#specify-inserted-objects) |
| Optimization component | A `DataSource`, `Perturbation` or `Scorer` | [Current optimization interfaces](../guides/optimization.md#current-optimization-interfaces) |
| Evaluation measure | A function over `Trace` records | [Driving traces](../guides/evaluation.md#driving-traces) |

For every new component, also:

- Add offline tests in the test group that matches its dependencies. See [Tests and CI](#tests-and-ci).
- Document it in the owning guide or reference file, following [Documentation rules](#documentation-rules).

## Tests and CI

Run these commands from the repository root. The test groups, their dependencies and the live
tests are described in [tests/README.md](../../tests/README.md).

```bash
python -m ruff check avsectester scripts tests        # syntax, undefined names, unused imports
python -m mypy                                        # data contracts and runtime
NO_ALBUMENTATIONS_UPDATE=1 python -m pytest tests/core -q   # core group, no avstack needed
NO_ALBUMENTATIONS_UPDATE=1 python -m pytest -q              # core and avstack groups
```

- Ruff selects only the `E9` and `F` rules. Formatting is not checked.
- mypy checks the closed-loop data types, hooks, simulator and AVStack boundaries, Vehicle I/O,
  runner, services, environment handling, geometry, attack interfaces, RunLog, codecs and
  dataset base listed in `pyproject.toml`.
  Imports outside these files are not followed.
- `python -m pytest` runs the `core` and `avstack` groups. The `avstack` group needs the avstack
  packages, installed as in [tests/README.md](../../tests/README.md#offline-avstack-tests).
- Live tests need a CARLA server, models or datasets. Select them explicitly, for example with
  `--run-carla`.

To test local changes with the dependencies of the Docker image, mount the checkout into a
container. No host Python environment, CARLA server or GPU is needed. Run this from the
repository root with the avstack submodules initialized:

```bash
docker run --rm -u "$(id -u):$(id -g)" \
  -v "$PWD":/app -w /app \
  -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -e MPLBACKEND=Agg -e NO_ALBUMENTATIONS_UPDATE=1 -e NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint python avsectester:latest \
  -m pytest tests/core tests/avstack -q -p no:cacheprovider
```

The mounted checkout supplies the code. The image supplies the dependencies, so rebuild it when
they change.

[CI](../../.github/workflows/ci.yml) runs on pull requests, pushes to `main` and manual dispatch,
on Ubuntu 22.04. It runs four jobs: `core-tests`, `avstack-tests` and `lint` with Python 3.11,
and `runlog-python312` for Parquet recording under the AlpaSim driver's interpreter version.
Lint uses Ruff 0.15.22 and mypy 1.20.2. CI installs CPU dependencies and does not use the Docker image.
Details are in [GitHub Actions](../../tests/README.md#github-actions).

When you write tests:

- Test the behavior you changed, including downstream consumption of replaced values, state
  reset between runs, unsupported capabilities and resource cleanup.
- Use in-memory providers and `StubRenderer` for scenario and preparation tests.
- Add a live test only for simulator or renderer behavior that an offline fixture cannot check.

## Documentation rules

The documentation is organized by what a reader needs to do. Each file has one type:

| Type | Location | How to write |
|---|---|---|
| Getting started | `docs/getting-started/` | Steps to a working result. Commands are complete and copyable |
| Concepts | `docs/concepts/` | Explanation of components, data flow and terms |
| Guide | `docs/guides/` | One task area. Explain how and when, show the common cases and link to reference tables for the complete set |
| Examples | `docs/examples/` | How the supplied attacks use the interfaces, one file per example in the same structure |
| Reference | `docs/reference/` | Complete lookup material: tables, signatures, keys and defaults. No tutorials |
| Development | `docs/development/` | Material for contributors |

Follow these rules:

- **Accuracy.** Check every API name, signature, default, YAML key, stage name and path against
  the code. Leave out a claim you cannot verify.
- **Single source.** Each fact has one owner document. Elsewhere, give at most a one-sentence
  summary and link to the owner. Do not copy tables between files. Link to the most specific
  heading that owns the fact.
- **Terminology.** Use the terms in [Terminology](../concepts/architecture.md#terminology)
  exactly. Use `simulator` for reusable configuration, `simulator_runtime` for a live world,
  `AVStack` for the driving system, `DrivingAdapter` for component/scorer bindings and
  `HookSet` for stage hooks. Use "metabuilder" for attack
  selection, "environment" for world and sensor settings, "treatment" for a named run and
  "command" for the driving output. Use legacy names only when explaining compatibility APIs.
- **Unimplemented features.** Mark them exactly as
  `**Not yet implemented.** <one sentence>. See [TODO](<relative path>/development/TODO.md#<anchor>).`
  Add the "See TODO" sentence only when a matching entry exists in [TODO.md](TODO.md).
- **Runnable examples.** Code examples must run against the current API. State caller-supplied
  placeholders in a comment. Prefer examples that run offline with `NuRecSimulator`, `StubRenderer`
  and a minimal `AVStack`, and execute them before committing.
- **Structure.** Start each page with its Audience, Outcome and Prerequisites block. End guides
  with `## Next steps`. Keep a guide under about 400 lines. Headings are link targets, so update
  every link when you rename one.
- **Style.** English, sentence-case headings, prose wrapped at about 100 characters, short
  declarative sentences, code identifiers in backticks, relative links between documents and
  fenced code blocks with a language. No emojis and no changelog wording, such as "now" or "new".

## Roadmap

Planned work is tracked in [TODO.md](TODO.md). Implementation status by component is summarized in
[Implementation status](../concepts/architecture.md#implementation-status).
