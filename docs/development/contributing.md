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

| Path | Content |
|---|---|
| `configs/` | Supplied experiment configurations |
| `scripts/` | Preparation, validation and visualization commands |
| `scripts/website/` | Site build hooks, built-site checks and homepage media tools |
| `tests/` | Core, native and opt-in external checks |
| `docs/` | Guides, examples and references |
| `models/` | Model weight locations. Weights are downloaded, not tracked |
| `benchmarks/` | Repository benchmarks |
| `third_party/` | External implementation checkouts |
| `website/` | Site configuration, pinned build dependencies, homepage source and media |

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
document. Elsewhere use one sentence and a link.

| Information | Owner document |
|---|---|
| YAML keys and defaults, split, metadata file format, device and time_step injection, paths and recording defaults | [Configuration reference](../reference/configuration.md) |
| Saved-preparation compatibility | [Configuration reference](../reference/configuration.md#saved-preparation-compatibility) |
| CLI commands, options, exit codes, output layout and automatic services | [CLI reference](../reference/cli.md) |
| Decision order, hook stages, action semantics, record types and `RunLog` | [Runtime reference](../reference/runtime.md) |
| Signatures, errors, formulas, map formats and custom-component extension contracts | [Python API reference](../reference/python-api.md) |
| Terms, package roles, dependency layers, architecture figure, isolation and world ownership | [Architecture](../concepts/architecture.md) |
| Supported dataset, simulator and driving-system combinations | [Architecture](../concepts/architecture.md) |
| Installation, images, Compose, `.env` and model/trafficsim weights | [Installation](../getting-started/installation.md) |
| First CARLA and NuRec runs | [Quickstart](../getting-started/quickstart.md) |
| Simulator behavior: insertion, occlusion, past observations, traffic, dynamics and navigation | [Simulators](../guides/simulators.md) |
| Driving-system behavior | [Driving systems](../guides/driving-systems.md) |
| Writing attacks, phases, deployment tick budgets, `AttackSegment` and `user_data` | [Implement an attack](../guides/attacks.md) |
| Optimization tools and offline reports | [Optimize attack artifacts](../guides/optimization.md) |
| Metrics, comparison-group membership and report display formats | [Evaluate experiments](../guides/evaluation.md) |
| Prepare/execute, metadata save/load/validation timing, resume semantics and saved maps | [Experiments](../guides/experiments.md) |
| Environments, metabuilder environments and sensor/hook randomness | [Experiment environments](../guides/environments.md) |
| Datasets, semantic maps and `ScenarioStart` contents | [Datasets and semantic maps](../guides/datasets.md) |
| Supplied configuration list and the impact-must-be-measured statement | [Supplied examples](../examples/README.md) |
| Supplied attack workflows and treatment names | [Stop-sign patch attack](../examples/stop-sign-patch.md), [Physical patch](../examples/physical-patch.md), [Image object insertion](../examples/image-insertion.md), [Phantom injection](../examples/phantom-injection.md), [Speed-feedback attack](../examples/speed-feedback-attack.md) |
| Implemented-attack directory rules | [Attack implementations](../../avsectester/implemented_attacks/README.md) |
| Unfinished work and Not yet implemented targets | [TODO](TODO.md) |
| Defense status | [Defenses](../guides/defenses.md) |
| Test commands, prerequisites and CI | [Tests](../../tests/README.md) |
| Site navigation | [website/mkdocs.yml](../../website/mkdocs.yml) |
| Homepage text and media | [website/content/](../../website/content/) |
| Homepage media sources | [Website media](../../scripts/README.md#website-media) |

Check API names, signatures, YAML keys, defaults and paths against source. Use `Simulator` for
configuration, `SimulatorRuntime` for a live world, `AVStack` for driving, `DrivingAdapter` for
bindings, `HookSet` for hooks, environment for world or sensor settings, and metabuilder for
segment selection.

Use offline phase and online phase for attack lifecycle phases. Call an attack that uses model
gradients a white-box attack and one that only queries rollouts a black-box attack.
Identify its optimizer only as "algorithm used: PGD" or "algorithm used: NES".

Use English, sentence-case headings and prose lines of at most 100 characters. Tables, code and
URLs are exempt. Task pages begin with Audience, Outcome and Prerequisites. Guides end with
Next steps.

Describe behavior and defaults. Keep measurements, investigation notes and version histories out
of the documentation. Mark missing capabilities with **Not yet implemented.** and link to a
specific heading in [TODO](TODO.md).

Self-contained examples place `<!-- runnable -->` immediately before their code fence. Python
and py blocks execute offline. Runnable YAML blocks load as complete experiment configurations.
External-service examples state their prerequisites. Other Python blocks must parse and use
resolvable framework imports.

Public modules declare `__all__`. The API reference lists exactly those names with checked
signatures. Internal modules are implementation details.

After changing a public docstring, follow the generator command in
[Repository tools](../../scripts/README.md#api-reference-generation).

The project site publishes docs/ unchanged under /docs/, except the TODO page, which stays on
GitHub. Site links to it open it there. Every page must render correctly both on GitHub and on the
site, so use only standard Markdown and GitHub tables. Do not use Material for MkDocs syntax such
as admonitions or content tabs. Add every new docs page to `nav` in
[website/mkdocs.yml](../../website/mkdocs.yml). The strict site build fails on omitted pages. To
preview the site, create a separate virtual environment, run
`pip install -r website/requirements.txt` and then `mkdocs serve -f website/mkdocs.yml`. This is a
documentation tool and runs outside the Compose experiment services.

## Roadmap

[TODO](TODO.md) lists remaining capabilities.

