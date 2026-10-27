# Contributing

> **Audience:** Code, test and documentation contributors.
>
> **Outcome:** Follow package boundaries and contribution rules.
>
> **Prerequisites:** [Architecture](../concepts/architecture.md).

## Design principles

Use native simulator and model implementations behind framework interfaces. Keep selection,
preparation, per-run state and recorded evidence in their owning components. The
[architecture](../concepts/architecture.md) defines package roles and isolation guarantees.

## Repository layout

| Directory | Content |
|---|---|
| configs/ | Supplied experiment configurations |
| scripts/ | Preparation, validation and visualization commands |
| tests/ | Core, native and opt-in external checks |
| docs/ | Guides, examples and references |
| `third_party/` | External implementation checkouts |

The framework package layout and import direction are defined in
[dependency layers](../concepts/architecture.md#dependency-layers).

## Add a component

Implement the public interface in a user-owned package and reference it through a type path. See
[attacks](../guides/attacks.md), [simulators](../guides/simulators.md),
[driving systems](../guides/driving-systems.md) and [datasets](../guides/datasets.md).

## Tests and CI

Follow [Writing tests](../../tests/README.md#writing-tests). The
[test guide](../../tests/README.md) owns commands, prerequisites, optional external checks and
CI job descriptions.

## Documentation rules

Consult this table before adding information. Keep the complete explanation in its owner
document; elsewhere use one sentence and a link.

| Information | Owner document |
|---|---|
| YAML keys and defaults, split, metadata file format, `device/time_step` injection, paths and recording defaults | [reference/configuration.md](../reference/configuration.md) |
| Saved-preparation compatibility | [reference/configuration.md](../reference/configuration.md#saved-preparation-compatibility) |
| CLI commands, options, exit codes, output layout and automatic services | [reference/cli.md](../reference/cli.md) |
| Decision order, hook stages, action semantics, record types and `RunLog` | [reference/runtime.md](../reference/runtime.md) |
| Signatures, errors, formulas, map formats and custom-component extension contracts | [reference/python-api.md](../reference/python-api.md) |
| Terms, package roles, dependency layers, architecture figure, isolation and world ownership | [concepts/architecture.md](../concepts/architecture.md) |
| Supported `dataset/simulator/AVStack` combinations | [concepts/architecture.md](../concepts/architecture.md) |
| Installation, images, Compose, `.env` and model/trafficsim weights | [getting-started/installation.md](../getting-started/installation.md) |
| First CARLA and NuRec runs | [getting-started/quickstart.md](../getting-started/quickstart.md) |
| `Simulator` behavior: insertion, occlusion, past observations, traffic, dynamics and navigation | [guides/simulators.md](../guides/simulators.md) |
| Driving-system behavior | [guides/driving-systems.md](../guides/driving-systems.md) |
| Writing attacks, phases, deployment tick budgets, `AttackSegment` and `user_data` | [guides/attacks.md](../guides/attacks.md) |
| Optimization tools and offline reports | [guides/optimization.md](../guides/optimization.md) |
| Metrics, comparison-group membership and report display formats | [guides/evaluation.md](../guides/evaluation.md) |
| Prepare/execute, metadata save/load/validation timing, resume semantics and saved maps | [guides/experiments.md](../guides/experiments.md) |
| Environments, metabuilder environments and sensor/hook randomness | [guides/environments.md](../guides/environments.md) |
| Datasets, semantic maps and `ScenarioStart` contents | [guides/datasets.md](../guides/datasets.md) |
| Supplied configuration list and the impact-must-be-measured statement | [examples/README.md](../examples/README.md) |
| Supplied attack workflows and treatment names | [Stop-sign patch](../examples/stop-sign-patch.md), [Physical patch](../examples/physical-patch.md), [Image insertion](../examples/image-insertion.md), [Phantom injection](../examples/phantom-injection.md), [Speed feedback](../examples/speed-feedback-attack.md) |
| Implemented-attack directory rules | [../../avsectester/implemented_attacks/README.md](../../avsectester/implemented_attacks/README.md) |
| Unfinished work and Not yet implemented targets | [development/TODO.md](TODO.md) |
| Defense status | [guides/defenses.md](../guides/defenses.md) |
| Test commands, prerequisites and CI | [../../tests/README.md](../../tests/README.md) |

Check API names, signatures, YAML keys, defaults and paths against source. Use `Simulator` for
configuration, `SimulatorRuntime` for a live world, `AVStack` for driving, `DrivingAdapter` for
bindings, `HookSet` for hooks, environment for world or sensor settings, and metabuilder for
segment selection.

Use `offline phase` and `online phase` for attack lifecycle phases.
Name access as `white-box attack` or `black-box attack`; identify its optimizer as
`algorithm used: PGD` or `algorithm used: NES`.

Use English, sentence-case headings and prose lines of at most 100 characters. Tables, code and
URLs are exempt. Task pages begin with Audience, Outcome and Prerequisites. Guides end with
Next steps.

Describe behavior and defaults. Keep measurements, investigation notes and version histories out
of the documentation. Mark missing capabilities with **Not yet implemented.** and link to a
specific [TODO](TODO.md#todo) heading.

Self-contained examples place `<!-- runnable -->` immediately before their code fence. Python
and py blocks execute offline; runnable YAML blocks load as complete experiment configurations.
External-service examples state their prerequisites. Other Python blocks must parse and use
resolvable framework imports.

Public modules declare `__all__`. The API reference lists exactly those names with checked
signatures. Internal modules are implementation details.

After changing a public docstring, run `scripts/development/write_api_reference.py`.

## Roadmap

[TODO](TODO.md) lists remaining capabilities.

