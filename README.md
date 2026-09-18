# AVSecTester

**Adversarial security-testing framework for autonomous-vehicle systems.**

AVSecTester runs an attack against a **real** AV pipeline in closed-loop simulation and measures the
effect on driving. It is built as two roles joined by a pure data plane — a **world backend** that
senses and actuates, and an **AV stack** (the box under test) that turns observations into control —
so any backend mixes with any stack, and an **attack is just a transform on the stream** between
them. Running the same scenario clean and attacked, and diffing the driving record, is the whole test.

Nothing in the AV stack is reimplemented: the modular pipeline *is* an
[avstack](https://github.com/avstack-lab) pipeline, and the end-to-end policy *is* NVIDIA's real
Alpamayo model — AVSecTester only adds the interface, the attack/metric seams, and the two new
simulator/stack halves.

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

`avsectester run configs/carla_scenario.yaml` **is** the modular demo — one command, clean vs
attacked, the impact verdict, and (with `--plot`) the figure. Details in [`docs/DOCKER.md`](docs/DOCKER.md).

## The interface: one loop, a 2×2 of parts

The framework is a pure **data plane** and two interfaces (`avsectester/plane.py`,
`avsectester/backend.py` — no CARLA/torch/avstack imports, serializable):

- **`Observation`** flows *down* (sensor data + calibration + ego state); **`Control`** flows *up*
  (throttle/steer/brake, or a `trajectory`). World state stays hidden in the backend.
- **`WorldBackend`** = `reset()` / `step(control) -> Observation` (owns the world + shared vehicle
  dynamics); **`AVStack`** = `__call__(obs) -> Control` (the box; knows nothing of modular vs
  end-to-end).
- **`run(backend, stack, frames, perturb=None) -> Trace`** drives the loop. `perturb: Observation ->
  Observation` is the **single universal attack seam** — it works against any stack, black-box
  included — and the `Trace` records the backend's *true* ego state, not the perturbed view.

Any world backend composes with any AV stack:

|                         | **ModularAVStack** (avstack pipeline) | **AlpamayoAVStack** (end-to-end policy) |
|-------------------------|---------------------------------------|-----------------------------------------|
| **CarlaBackend** (CARLA closed loop) | the one-command demo above — phantom-injection at perception | Alpamayo driving a CARLA world |
| **NuRecBackend** (NuRec neural reconstruction) | modular stack on reconstructed camera/lidar | ✔ verified end-to-end — Alpamayo on photoreal NuRec frames |

```python
from avsectester.backend import run
from avsectester.simulators import NuRecBackend, NuRecRenderer, TrajectoryFollower
from avsectester.stacks import AlpamayoAVStack

backend = NuRecBackend({"dt": 0.1, "ego0": {"speed": 5.0}},
                       renderer=NuRecRenderer(endpoint="127.0.0.1:50051", scene_id="01d503d4"),
                       dynamics=TrajectoryFollower())
trace = run(backend, AlpamayoAVStack(device="cuda:1"), frames=8)   # real Alpamayo on real NuRec imagery
```

See [`docs/INTERFACE.md`](docs/INTERFACE.md) for the full contract and
[`scripts/alpamayo_nurec_demo.py`](scripts/alpamayo_nurec_demo.py) for the runnable Alpamayo+NuRec demo.

## The parts

| Piece | What it is |
|-------|------------|
| **Data plane** (`avsectester/plane.py`) | `Observation` / `Control` / `Trace` — the pure sim↔stack contract. |
| **Interfaces** (`avsectester/backend.py`) | `WorldBackend`, `AVStack`, and the `run(...)` loop with the `perturb` attack seam. |
| **CarlaBackend + ModularAVStack** (`avsectester/scenario.py`) | avcarla `CarlaClient`/`CarlaMobileActor`/`CarlaNpc` + an avstack `ModularDrivingPipeline` (perception → tracking → planning → control), built from config. |
| **NuRecBackend** (`avsectester/simulators/nurec.py`) | In-process NVIDIA **NuRec** neural-reconstruction world; pluggable `Renderer` (`StubRenderer` black frames for CI, `NuRecRenderer` → the `nre-ga` gRPC renderer); `KinematicBicycle`/`TrajectoryFollower` dynamics; `checkpoint()`/`restore()`. |
| **AlpamayoAVStack** (`avsectester/stacks/alpamayo.py`) | Wraps the real **Alpamayo-1.5-10B** end-to-end policy as an `AVStack`: camera frames → trajectory `Control`. |
| **Attacks** (`avsectester/attacks/`) | Universal: `perturb(Observation)`. Modular white-box: an avstack `HOOKS` hook on a pipeline stage (e.g. `PhantomInjection` on `perception`). |
| **Metric** (`avsectester/metric.py`) | Diffs a clean vs attacked `Trace` into a driving-impact verdict. |
| **Visualization** | `avsectester/metric.py:plot_impact` — the clean-vs-attacked impact plot (metric view); `avsectester/simulators/viz.py` — the **simulator-agnostic** scene pipeline (`record_run`, `detections_view`, `filmstrip`, `save_gif` over a generic RGB *view*), with per-simulator view adapters in `simulators/<sim>.py` (e.g. `simulators/carla.py`: `camera_view` for `ImageData`, `lidar_bev`). |

## Built on avstack + AlpaSim

Vendored under `third_party/` as git submodules (forked so the closed-loop pieces can live upstream):

- **avstack-core** — reconfigurable AV modules, geometry, sensors, registry/config, hooks
- **lib-avstack-carla** (`avcarla`) — closed-loop CARLA 0.9.15 bridge (client, actors, sensors)
- **avstack-api** — KITTI / nuScenes / CARLA dataset adapters

The NuRec + Alpamayo halves reuse NVIDIA's [AlpaSim](https://github.com/NVlabs/alpasim) pieces: the
`nre-ga` NuRec renderer serves reconstructed scenes over gRPC, and `alpasim_driver` provides the
Alpamayo model. Those run in a dedicated Python-3.12 driver env; see [`docs/SETUP.md`](docs/SETUP.md).

## Status

**Early alpha.** Two closed loops are verified end-to-end: the neural CARLA modular loop (the demo
above) and the **real Alpamayo-1.5-10B policy driving on real NuRec-rendered imagery**. The offline
suite (`tests/`) covers the interface, the attack hook, the driving pipeline, the in-process NuRec
backend, and the scene viz without needing CARLA or a GPU. A scenario-search engine and richer
attack/defense hooks are planned — see [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

## License

MIT — see [`LICENSE`](LICENSE).
