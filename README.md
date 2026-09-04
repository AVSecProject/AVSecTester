# AVSecTester

**Adversarial security-testing framework for autonomous-vehicle systems.**

AVSecTester runs an attack against a **real** AV pipeline in closed-loop CARLA simulation and
measures the effect on driving. The AV stack is not reimplemented here — it *is* an
[avstack](https://github.com/avstack-lab) pipeline running in the sim through avstack's own CARLA
bridge. An attack is an avstack **hook** attached to a pipeline stage. Running the same scenario
clean and attacked, and diffing the driving record, is the whole test.

## ▶ Demo — one command

Bring up a CARLA server + the GPU image and run a phantom-detection attack against **real neural
perception** in a closed-loop drive:

```bash
git clone --recurse-submodules <repo> && cd AVSecTester
git submodule update --init third_party/avstack-core
cd third_party/avstack-core && git submodule update --init --depth 1 \
  third_party/mmdetection third_party/mmdetection3d third_party/mmsegmentation && cd -
./scripts/fetch_models.sh          # CARLA-trained weights → ./models
docker compose up -d --build       # start a CARLA 0.9.15 server + the AVSecTester shell
docker compose exec avsectester avsectester run configs/carla_scenario.yaml --frames 40   # run it
```

A CARLA-trained **PointPillars** detector runs on a live **CarlaLidar**; a fabricated detection is
injected at the perception stage (an avstack hook — no pixels touched):

```
[clean]    peak_speed=5.19 final_speed=5.17 brake_frames=0     # ego cruises, detects real NPCs
[attacked] final_speed=0.00 brake_frames=38                    # phantom → emergency stop
=> ATTACK SUCCEEDED (forced an unsafe stop)
```

**Visualize it:** add `--plot results/impact.png` (needs the `viz` extra) to save a clean-vs-attacked
plot of ego speed + brake over time — the green (clean) line cruises while the red (attacked) line
brakes to a full stop.

`avsectester run configs/carla_scenario.yaml` **is** the demo — one command, clean vs attacked, the
impact verdict, and (with `--plot`) the figure. Details in [`docs/DOCKER.md`](docs/DOCKER.md).

## How it works

The framework is deliberately tiny — it adds a security layer, nothing more:

| Piece | What it is |
|-------|------------|
| **Scenario** (`avsectester/scenario.py`) | Builds an avcarla `CarlaClient` + `CarlaMobileActor` (ego) + `CarlaNpc` traffic **from config**, drives the loop, returns a driving `Trace`. |
| **Pipeline** | The ego's brain is an avstack `ModularDrivingPipeline`: neural perception → tracking → planning → control. Real avstack modules, built from config. |
| **Attack** (`avsectester/attacks/`) | An avstack `HOOKS` hook attached to a pipeline stage (e.g. `PhantomInjection` on `perception`). |
| **Metric** (`avsectester/metric.py`) | Diffs a clean vs attacked `Trace` into a driving-impact verdict. |

There is **no** parallel environment/system/attack machinery and **no** mock — AVSecTester uses
avstack's own interfaces (`CARLA`/`PIPELINE`/`MODELS`/`HOOKS` registries, `register_post_hook`)
directly. The closed-loop driving stack (`ModularDrivingPipeline`, `ForwardCollisionPlanner`) lives
in the avstack fork where it belongs; see [`docs/INTERFACE.md`](docs/INTERFACE.md) and
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

## Built on avstack

Vendored under `third_party/` as git submodules (forked so the closed-loop pieces can live upstream):

- **avstack-core** — reconfigurable AV modules, geometry, sensors, registry/config, hooks
- **lib-avstack-carla** (`avcarla`) — closed-loop CARLA 0.9.15 bridge (client, actors, sensors)
- **avstack-api** — KITTI / nuScenes / CARLA dataset adapters

## The one command

There is a single entry point — the same one used in the demo above:

```bash
avsectester run configs/carla_scenario.yaml --frames 40 [--gpu 1] [--plot results/impact.png]
```

It builds the scenario, runs it clean then attacked, prints the impact verdict, and (with `--plot`)
saves the figure. The scenario config is the whole experiment: the `CarlaClient`, the ego (sensors +
`ModularDrivingPipeline`), the NPC traffic, and the attack hooks.

- `--frames` — steps per run (needs enough for the clean ego to reach cruising speed; 40 is good).
- `--gpu` — perception CUDA device. The config targets GPU 0 (right in Docker, where the ego gets a
  dedicated GPU); on a single host where CARLA already holds GPU 0, pass `--gpu 1`.
- `--plot` — save the clean-vs-attacked driving-impact figure (needs the `viz` extra).

Exit code encodes the verdict: `0` attack succeeded, `2` inconclusive (clean never drove), `1` no
meaningful impact.

## Status

**Early alpha.** The neural CARLA closed loop is verified end-to-end (the demo above). The offline
suite (`tests/`) covers the attack hook and the driving pipeline without needing CARLA. A
scenario-search engine and richer attack/defense hooks are planned — see
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

## License

MIT — see [`LICENSE`](LICENSE).
