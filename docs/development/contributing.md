# Contributing

> **Audience:** Contributors changing code, tests or documentation.  
> **Outcome:** Follow the design principles, find code by component, run tests and write documentation consistently.  
> **Prerequisites:** [Architecture](../concepts/architecture.md).


## Design principles

- **No parallel definitions.** Adapters reuse the provider's actors, sensors, model components,
  configuration registries and native hooks. They do not reimplement simulator actors or
  driving-model modules. This repository owns scenario preparation, intervention preparation and
  execution, and the messages exchanged between the simulator and the driving system. When a
  missing capability belongs to avstack, add it to the avstack fork.
- **One driving loop over a pure data plane.** `plane.py` defines the messages (`Observation`,
  `Control`, `Trace`) and imports no simulator or model package. `backend.py` defines the two
  adapter interfaces, `WorldBackend` and `AVStack`, and the loop `run`. Paired runners, component
  logging and scene recording all drive this same loop.
- **Physical state stays separate from model-visible data.** Handlers that change the observation
  never move the vehicle. World changes use explicit simulator operations. See
  [Physical state and model-visible data](../concepts/architecture.md#physical-state-and-model-visible-data).
- **Attacks and defenses share one interface.** Both are interventions: optional preparation
  that produces artifacts, and runtime handlers registered at stages. Prefer extending that
  interface over adding a separate execution path.
- **Conditions are not interventions.** World and sensor conditions apply identically to every
  treatment of a scenario. Keep their settings independent of attack and defense variables.
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
| `avsectester/plane.py` | Vehicle I/O | Data plane: `Observation`, `Control`, `WorldSnapshot`, `StateEstimate`, `Trace`, `FrameRecord` |
| `avsectester/backend.py` | Closed-loop experiment | `WorldBackend`, `AVStack`, command-mode validation and the driving loop `run` |
| `avsectester/runner.py`, `avsectester/experiment.py` | Closed-loop experiment | Named treatments from one origin (`run_prepared`, `run_pair`) and configured experiments (`run_configured`) |
| `avsectester/carla_runner.py`, `avsectester/cli.py` | Closed-loop experiment | CARLA and modular driving-system assembly (`run_scenario`, `run_scenario_pair`) and the `avsectester` command |
| `avsectester/scenarios/` | Scenario preparation | Scenario sources, configured setup, requirements, filters, roles, dataset and CARLA providers, natural-language requirements |
| `avsectester/conditions.py`, `avsectester/perturbations/` | Scenario preparation (conditions) | `ScenarioConditions`, `SensorCondition`, image corruptions |
| `avsectester/insertion.py`, `avsectester/rendering/` | Scenario preparation (insertions) | Insertion specification, cameras, compositing, harmonizers, visibility |
| `avsectester/simulators/` | Simulator | `CarlaBackend`, `NuRecBackend`, renderers and ego dynamics, scene visualization |
| `avsectester/stacks/` | Driving system | `ModularAVStack`, `AlpamayoAVStack`, CARLA dataset registration |
| `avsectester/runtime.py` | Attack and defense execution | `Runtime`, `Hook`, `Plugin`, contexts, stage validation |
| `avsectester/preparation.py`, `avsectester/interventions.py` | Attack and defense preparation | `Intervention`, `PreparedIntervention`, `Treatment`, runtime adapters and configuration helpers |
| `avsectester/attacks/` | Attacks | Phantom injection (`pipeline/`), physical patch (`patch/`), object-insertion payloads (`object_insertion/`), speed-feedback attack (`adaptive.py`) |
| `avsectester/attacks/optim/` | Optimization | PGD, NES, perturbations, scorers, data sources, CARLA-rendered perturbation |
| `avsectester/metric.py`, `avsectester/evaluation/` | Evaluation | Driving-impact metric and plot, component logging, robustness evaluation |
| `configs/` | Closed-loop experiment | Supplied CARLA scenario configurations |
| `scripts/` | All | Demos, model and asset preparation, optimization, validation and visualization entry points. See [scripts/README.md](../../scripts/README.md) |
| `tests/` | All | `core`, `avstack` and `live` test groups. See [tests/README.md](../../tests/README.md) |
| `third_party/` | Simulator, driving system | Git submodules: the avstack projects and `libcom` |

Shared geometry, compositing and harmonization live in `rendering/`, not in an attack family.
Simulator adapters live in `simulators/`, and observation transforms in `perturbations/`.

## Add a component

Each extension point is documented in the guide that owns it. Start from that section.

| To add | Implement | Guide |
|---|---|---|
| Simulator | A `WorldBackend` adapter | [Write a custom simulator adapter](../guides/simulators.md#write-a-custom-simulator-adapter) |
| Driving system | An `AVStack` adapter, used through `stack_factory` or `stack.factory` | [Use a custom driving system](../guides/driving-systems.md#use-a-custom-driving-system) |
| Component outputs of a driving system | The component-logging stack interface | [Component logging interface](../guides/driving-systems.md#component-logging-interface) |
| Attack | Runtime handlers, packaged as an `Intervention` | [Write your first attack](../guides/attacks.md#write-your-first-attack), [Package the attack as an intervention](../guides/attacks.md#package-the-attack-as-an-intervention) |
| Defense | An `Intervention` configured with `kind: defense` | [Package and configure a defense](../guides/defenses.md#package-and-configure-a-defense) |
| World condition | Support in the simulator adapter's `configure_conditions` | [World conditions](../guides/conditions.md#world-conditions) |
| Sensor condition | A `SensorCondition` with a handler factory | [Custom sensor conditions](../guides/conditions.md#custom-sensor-conditions) |
| Selection filter | A `Constraint` that returns a `FilterResult` | [Custom filters and native access](../guides/scenarios.md#custom-filters-and-native-access) |
| Dataset or candidate provider | Candidate initial states and a factory that recreates the selected origin | [Support another dataset or provider](../guides/scenarios.md#support-another-dataset-or-provider) |
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
- mypy checks the files listed in `pyproject.toml`: `plane.py`, `backend.py`, `runtime.py` and
  `runner.py`. Imports outside these files are not followed.
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
on Ubuntu 22.04 with Python 3.11. It runs three jobs: `core-tests`, `avstack-tests` and `lint`
(Ruff 0.15.22 and mypy 1.20.2). CI installs CPU dependencies and does not use the Docker image.
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
  exactly. Say "simulator" and "driving system" for the components, and "backend" and "stack"
  only for the Python adapters. Say "filter" for selection checks and "condition" only for world
  and sensor settings. Say "intervention" for an attack or defense, "treatment" for a named run,
  "command" for the driving system's output and "driving step" for one decision cycle.
- **Unimplemented features.** Mark them exactly as
  `**Not yet implemented.** <one sentence>. See [TODO](<relative path>/development/TODO.md#<anchor>).`
  Add the "See TODO" sentence only when a matching entry exists in [TODO.md](TODO.md).
- **Runnable examples.** Code examples must run against the current API. State caller-supplied
  placeholders in a comment. Prefer examples that run offline with `NuRecBackend`, `StubRenderer`
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
