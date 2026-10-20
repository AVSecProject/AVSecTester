# Documentation

Start with [Installation](getting-started/installation.md) and
[Quickstart](getting-started/quickstart.md). Read [Architecture](concepts/architecture.md) before
extending the framework.

## Reading paths

| Goal | Documents |
|---|---|
| Run supplied experiments | Quickstart → CLI → Configuration |
| Write an attack | Architecture → Attacks → supplied attack examples |
| Optimize a patch | Attacks → Optimization → Physical patch |
| Integrate a simulator or driving system | Simulators or Driving systems → Runtime → Python API |
| Analyze existing results | Evaluation → Python API |

## All documents

| Document | Purpose |
|---|---|
| [Installation](getting-started/installation.md) | Dependencies, services, weights and assets |
| [Quickstart](getting-started/quickstart.md) | First run, outputs and saved-result evaluation |
| [Architecture](concepts/architecture.md) | Components, ownership and terminology |
| [Experiments](guides/experiments.md) | Preparation, execution, persistence and treatment comparisons |
| [Attacks](guides/attacks.md) | User attack API and effect lifecycle |
| [Defenses](guides/defenses.md) | Current scope and future work |
| [Scenarios](guides/scenarios.md) | Dataset timelines, actor identities and metadata |
| [Environments](guides/environments.md) | World and sensor environments |
| [Simulators](guides/simulators.md) | Supplied simulators and custom simulator runtime implementation |
| [Driving systems](guides/driving-systems.md) | Modular and Alpamayo integration |
| [Optimization](guides/optimization.md) | User objectives, scorers and patch optimization |
| [Evaluation](guides/evaluation.md) | RunLog metrics, reports and visualization |
| [Attack examples](examples/README.md) | Supplied Attack implementations |
| [CLI](reference/cli.md) | Commands and exit codes |
| [Configuration](reference/configuration.md) | YAML keys and object construction |
| [Runtime](reference/runtime.md) | Stages, causal order and logging |
| [Python API](reference/python-api.md) | Modules and public interfaces |
| [Contributing](development/contributing.md) | Design rules, checks and documentation conventions |
| [TODO](development/TODO.md#todo) | Open development work |

See also [scripts](../scripts/README.md) and [tests](../tests/README.md).
