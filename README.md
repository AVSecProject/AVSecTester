# AVSecTester

**Adversarial security-testing framework for autonomous-vehicle systems.**

AVSecTester runs attacks and defenses against real driving systems in closed-loop simulation and
measures their effect on driving. The same prepared scenario is run under different treatments,
such as clean and attacked, and the runs are compared.

The framework does not reimplement simulators or driving models. The modular driving system is an
[avstack](https://github.com/avstack-lab) pipeline, the end-to-end driving system is NVIDIA's
Alpamayo model, and the simulators are CARLA and NVIDIA NuRec. AVSecTester provides the interfaces
between them, scenario preparation, attack and defense integration, and evaluation.

See [Architecture](docs/concepts/architecture.md) for each component and the terms used in the
documentation.

## Quick start

Run a phantom-detection attack against a CARLA-trained PointPillars detector in a closed-loop
CARLA drive. This needs NVIDIA GPUs and Docker with the NVIDIA runtime.

Before starting, check the
[Docker prerequisites](docs/getting-started/installation.md#prerequisites) to configure the NVIDIA
runtime as Docker's default and adjust both `NVIDIA_VISIBLE_DEVICES` values in `docker-compose.yml`
to match your machine.

```bash
git clone https://github.com/AVSecProject/AVSecTester.git
cd AVSecTester
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --depth 1 -- \
  third_party/avstack-core third_party/avstack-api third_party/lib-avstack-carla
git -C third_party/avstack-core -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init --depth 1 -- third_party/mmdetection third_party/mmdetection3d
./scripts/preparation/fetch_models.sh
docker compose up -d --build
docker compose exec avsectester avsectester run configs/carla_scenario.yaml --frames 40
```

The command runs a clean drive and an attacked drive from the same start and prints their driving
statistics and a verdict. Continue with the [Quickstart](docs/getting-started/quickstart.md) to
read the output, or with [Installation](docs/getting-started/installation.md) for other
environments, including NuRec with Alpamayo.

## Documentation

| I want to… | Start here |
|---|---|
| Install and run a first experiment | [Installation](docs/getting-started/installation.md), [Quickstart](docs/getting-started/quickstart.md) |
| Understand the components and terms | [Architecture](docs/concepts/architecture.md) |
| Prepare attacks and execute experiments from Python | [Run experiments](docs/guides/experiments.md#prepare-and-execute-attacks), [Configuration](docs/reference/configuration.md#experiment-configuration-python-api) |
| Implement an attack or a defense | [Implement an attack](docs/guides/attacks.md), [Implement a defense](docs/guides/defenses.md) |
| See how the supplied attacks work | [Examples](docs/examples/README.md) |
| Look up configuration keys, CLI options or runtime stages | [Reference](docs/README.md#all-documents) |
| Contribute | [Contributing](docs/development/contributing.md) |

The [documentation index](docs/README.md) lists every document and a reading path for each task.

## Status

**Early alpha.** The repository supplies CARLA and NuRec simulators, the modular and Alpamayo
driving systems, scenario preparation and selection, world and sensor conditions, shared attack and
defense interfaces with example attacks, and evaluation tools. The Python experiment API supports
attack-defined selection, offline preparation and independent treatments with reusable artifacts.
The CLI and supplied examples use the existing configured-run API. Built-in optimizers still use
their existing data path. See
[Implementation status](docs/concepts/architecture.md#implementation-status) and
[TODO](docs/development/TODO.md).

**Not yet implemented.** Reference defense algorithms and selectable success criteria. See
[TODO](docs/development/TODO.md#defense-examples-and-output-metrics) and
[driving-impact metrics](docs/development/TODO.md#driving-impact-metrics).

**Not yet implemented.** Broader project goals include complete experiment reports,
attack-propagation analysis, reusable vulnerability records and automation through an AI harness.
Component testing, hardware-in-the-loop, in-vehicle network and V2X adapters are possible extensions.

## License

MIT — see [`LICENSE`](LICENSE).
