# Documentation

Start with [Architecture](concepts/architecture.md). It describes every component, shows how data
flows between them and defines the terms used in all other documents.

## Find what you need

| I want to… | Read, in order |
|---|---|
| Run a demo first | [Installation](getting-started/installation.md) → [Quickstart](getting-started/quickstart.md) |
| Implement a new attack | [Architecture](concepts/architecture.md) → [Implement an attack](guides/attacks.md). Look up details in [Runtime](reference/runtime.md) and [Configuration](reference/configuration.md). See [Examples](examples/README.md) for supplied attacks |
| Implement a new defense | [Architecture](concepts/architecture.md) → [Implement a defense](guides/defenses.md) → [Runtime](reference/runtime.md) |
| Optimize an attack artifact, such as a patch texture | [Implement an attack](guides/attacks.md) → [Optimize attack artifacts](guides/optimization.md) → [Physical patch example](examples/physical-patch.md) |
| Test an attack under bad weather or sensor degradation | [Apply operating conditions](guides/conditions.md) → [Evaluate results](guides/evaluation.md) |
| Select scenarios that meet requirements | [Prepare scenarios](guides/scenarios.md) → [Configuration](reference/configuration.md#selection-keys) |
| Configure CARLA or NuRec scenes | [Configure simulators](guides/simulators.md) → [Configuration](reference/configuration.md) |
| Integrate a new simulator or driving system | [Architecture](concepts/architecture.md) → [Configure simulators](guides/simulators.md#write-a-custom-simulator-adapter) or [Configure driving systems](guides/driving-systems.md#use-a-custom-driving-system) → [Runtime](reference/runtime.md) |
| Contribute code or documentation | [Contributing](development/contributing.md) → [Tests](../tests/README.md) → [TODO](development/TODO.md) |

## All documents

**Getting started**

| Document | Contents |
|---|---|
| [Installation](getting-started/installation.md) | Docker, local environments, the NuRec and Alpamayo environment, model weights and services |
| [Quickstart](getting-started/quickstart.md) | Run a first comparison, read its output and run offline without a GPU |

**Concepts**

| Document | Contents |
|---|---|
| [Architecture](concepts/architecture.md) | Components, data flow, supported combinations, implementation status and terminology |

**Guides**

| Document | Contents |
|---|---|
| [Run experiments](guides/experiments.md) | Entry points, treatments and modes, configured and custom runs, reproducibility, outputs |
| [Prepare scenarios](guides/scenarios.md) | Configured scenarios, inserted objects, filters and requirements, selection, visibility, custom providers |
| [Apply operating conditions](guides/conditions.md) | World and sensor conditions, corruption operators, ordering and randomness |
| [Configure simulators](guides/simulators.md) | CARLA and NuRec setup, seeds and spawning, world changes from handlers, custom simulators |
| [Configure driving systems](guides/driving-systems.md) | Modular and Alpamayo driving systems, models, command compatibility, custom driving systems |
| [Implement an attack](guides/attacks.md) | Writing, testing, packaging, configuring and running an attack |
| [Implement a defense](guides/defenses.md) | Writing, ordering, configuring and evaluating a defense |
| [Optimize attack artifacts](guides/optimization.md) | Where optimization fits and the current optimization interfaces |
| [Evaluate results](guides/evaluation.md) | Driving traces, the driving-impact metric, component outputs and robustness evaluation |

**Examples**

| Document | Contents |
|---|---|
| [Examples](examples/README.md) | How to read the examples, and which supplied attack acts where |
| [Phantom injection](examples/phantom-injection.md) | A fabricated detection added to the modular driving system's perception output |
| [Physical patch](examples/physical-patch.md) | A textured panel deployed on a vehicle in CARLA, with optional texture optimization |
| [Image insertion](examples/image-insertion.md) | Signs, pedestrians, signal boards and patches inserted into camera images |
| [Speed-feedback adaptive attack](examples/speed-feedback-attack.md) | A stateful attack that adapts to feedback during a run |

**Reference**

| Document | Contents |
|---|---|
| [Configuration](reference/configuration.md) | Every configuration key |
| [CLI](reference/cli.md) | Commands, options, output and exit codes |
| [Runtime](reference/runtime.md) | Stages, stage values, execution order, ordering rules, handler lifecycle and contexts |
| [Python API](reference/python-api.md) | Public classes and functions by module |

**Development**

| Document | Contents |
|---|---|
| [Contributing](development/contributing.md) | Design principles, repository layout, tests and documentation rules |
| [TODO](development/TODO.md) | Open development work |
| [Tests](../tests/README.md) | Test suites, CI and live tests |
| [Scripts](../scripts/README.md) | Index of demo, preparation, optimization and validation scripts |
