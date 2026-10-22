# AVSecTester

**Adversarial security testing for autonomous driving.**

AVSecTester selects attack segments, prepares reusable artifacts, runs clean and attacked
closed-loop treatments, and evaluates their saved logs. It integrates existing simulators and
driving systems: CARLA with the modular [avstack](https://github.com/avstack-lab) pipeline, and
NVIDIA NuRec with Alpamayo. nuScenes supports recorded open-loop playback.

## Quick start

The CARLA experiment needs NVIDIA GPUs, model checkpoints and Docker with the NVIDIA runtime. Before
starting, follow the [Docker prerequisites](docs/getting-started/installation.md#prerequisites) and
adjust both `NVIDIA_VISIBLE_DEVICES` entries in `docker-compose.yml` for your machine.

```bash
git clone https://github.com/AVSecProject/AVSecTester.git
cd AVSecTester
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --depth 1 -- \
  third_party/avstack-core third_party/avstack-api third_party/lib-avstack-carla
git -C third_party/avstack-core -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init --depth 1 -- third_party/mmdetection third_party/mmdetection3d
./scripts/preparation/fetch_models.sh
docker compose up -d --build
docker compose exec avsectester avsectester run configs/carla_phantom.yaml
```

Set the server address and model device on the configured components; set treatments and output
under runs.treatments and runs.output. `run`
prepares, executes and evaluates the experiment. `evaluate` recalculates metrics from saved logs
after the simulator closes. See [Quickstart](docs/getting-started/quickstart.md).

## Documentation

| Task | Start here |
|---|---|
| Set up an environment | [Installation](docs/getting-started/installation.md) |
| Understand the data flow | [Architecture](docs/concepts/architecture.md) |
| Run an experiment | [Experiments](docs/guides/experiments.md), [CLI](docs/reference/cli.md) |
| Write an attack | [Attack guide](docs/guides/attacks.md), [Examples](docs/examples/README.md) |
| Configure simulator, driving system and metrics | [Configuration](docs/reference/configuration.md) |
| Add an adapter or contribute | [Contributing](docs/development/contributing.md) |

The [documentation index](docs/README.md) lists all guides and references.

## Status

**Early alpha.** Experiments use `Attack`, `AttackRuntime`, `Simulator`, `SimulatorRuntime`,
`AVStack`, `HookSet` and `RunLog`. Supplied configurations cover phantom injection, physical
patches, image object insertion and speed feedback. Metrics and visualizations consume saved logs.
End-to-end execution checks do not establish attack impact.

**Not yet implemented.** Defense treatments and reference defense algorithms. See
[TODO](docs/development/TODO.md#defenses).

**Not yet implemented.** Reactive NuRec traffic and a physical trajectory controller. See
[TODO](docs/development/TODO.md#vehicle-dynamics).

## License

MIT — see [LICENSE](LICENSE).
