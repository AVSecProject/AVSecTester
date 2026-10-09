# AVSecTester

**Adversarial security-testing framework for autonomous-vehicle systems.**

AVSecTester runs an attack against a **real** AV pipeline in closed-loop simulation and measures the
effect on driving. It is built as two roles joined by a pure data plane — a **world backend** that
senses and actuates, and an **AV stack** (the box under test) that turns observations into control —
so backend and stack adapters share one driving loop. Attack and defense handlers modify the
inputs, components, commands or world operations supported by those adapters. Paired runs measure
the resulting driving changes.

Nothing in the AV stack is reimplemented: the modular pipeline *is* an
[avstack](https://github.com/avstack-lab) pipeline, and the end-to-end policy *is* NVIDIA's real
Alpamayo model. AVSecTester provides the connecting interfaces, scenario selection, insertion
geometry, attack integration and evaluation.

## Documentation

| Task | Guide |
|---|---|
| Install dependencies and start simulator services | [Setup](docs/SETUP.md), [Docker](docs/DOCKER.md) |
| Understand backend, stack and attack contracts | [Interfaces](docs/INTERFACE.md) |
| Implement and combine custom attacks or defenses | [Attack and defense interfaces](docs/INTERVENTIONS.md) |
| Select initial cases, bind actors and add custom filters | [Scenario selection](docs/SCENARIOS.md) |
| Insert patches or objects and inspect visibility | [Image attacks](docs/IMAGE_ATTACKS.md) |
| Run corruption conditions and interpret outcome summaries | [Robustness evaluation](docs/AUGMENTATION.md) |
| Capture intermediate component outputs | [Component logging](docs/COMPONENT_LOGGING.md) |
| Extend the framework and run tests | [Development](docs/DEVELOPMENT.md), [Tests](tests/README.md) |

## ▶ Demo — one command

Bring up a CARLA server + the GPU image and run a phantom-detection attack against **real neural
perception** in a closed-loop drive:

```bash
git clone --recurse-submodules https://github.com/AVSecProject/AVSecTester.git && cd AVSecTester
git submodule update --init third_party/avstack-core
cd third_party/avstack-core && git submodule update --init --depth 1 \
  third_party/mmdetection third_party/mmdetection3d third_party/mmsegmentation && cd -
./scripts/preparation/fetch_models.sh          # CARLA-trained weights → ./models
docker compose up -d --build       # start a CARLA 0.9.16 server + the AVSecTester shell
docker compose exec avsectester avsectester run configs/carla_scenario.yaml --frames 40   # run it
```

A CARLA-trained **PointPillars** detector runs on a live **CarlaLidar**; a fabricated detection is
injected at the perception stage (an avstack hook — no pixels touched):

The command reports clean and attacked driving statistics and an impact verdict. Values depend on
the scene, model and run length. An attack can succeed, fail, or be inconclusive when the clean run
does not establish a driving baseline.

**Visualize it:** add `--plot results/impact.png` (needs the `viz` extra) to save a clean-vs-attacked
plot of ego speed and brake over time for both runs.

`avsectester run configs/carla_scenario.yaml` **is** the modular demo — one command, clean vs
attacked, the impact verdict, and (with `--plot`) the figure. Details in [`docs/DOCKER.md`](docs/DOCKER.md).

## The interface: one loop, pluggable parts

The framework is a pure **data plane** and two interfaces (`avsectester/plane.py`,
`avsectester/backend.py` — no CARLA/torch/avstack imports). Native sensor and state payloads
retain adapter-specific types and require appropriate codecs for transport between processes:

- **`Observation`** flows *down* (sensor data + calibration + ego state); **`Control`** flows *up*
  (throttle/steer/brake, or a `trajectory`). Physical world state is kept independent of
  model-visible estimates and is available through `backend.ground_truth()`.
- **`WorldBackend`** = `reset()` / `step(control) -> Observation` (owns the world + shared vehicle
  dynamics); **`AVStack`** = `__call__(obs) -> Control` (the box; knows nothing of modular vs
  end-to-end).
- **`run(backend, stack, frames, perturb=None, *, runtime=None) -> Trace`** drives the loop.
  `Runtime` registers ordered attack/defense handlers at supported input, component, world and
  command stages. `perturb(Observation)` remains a shorthand for final input processing.
  `Trace` records physical state independently of the attacked view.

Backend and stack adapters compose when their sensor, state and control formats are compatible:

|                         | **ModularAVStack** (avstack pipeline) | **AlpamayoAVStack** (end-to-end policy) |
|-------------------------|---------------------------------------|-----------------------------------------|
| **CarlaBackend** (CARLA closed loop) | Supplied driving path, with phantom injection at perception | Requires a trajectory-to-actuator controller and compatible camera inputs |
| **NuRecBackend** (NuRec neural reconstruction) | Requires compatible modules and avstack sensor/state adapters | Supplied driving path, using `TrajectoryFollower` |

```python
from avsectester.backend import run
from avsectester.simulators import NuRecBackend, NuRecRenderer, TrajectoryFollower
from avsectester.stacks import AlpamayoAVStack

backend = NuRecBackend({"dt": 0.1, "ego0": {"speed": 5.0}},
                       renderer=NuRecRenderer(endpoint="127.0.0.1:50051", scene_id="01d503d4"),
                       dynamics=TrajectoryFollower())
trace = run(backend, AlpamayoAVStack(device="cuda:1"), frames=8)   # real Alpamayo on real NuRec imagery
```

See [`docs/INTERFACE.md`](docs/INTERFACE.md) for the backend/stack contract,
[`docs/INTERVENTIONS.md`](docs/INTERVENTIONS.md) for attack/defense callbacks, and
[`scripts/demos/nurec/alpamayo_nurec_demo.py`](scripts/demos/nurec/alpamayo_nurec_demo.py) for the runnable Alpamayo+NuRec demo.

## The parts

| Piece | What it is |
|-------|------------|
| **Data plane** (`avsectester/plane.py`) | `Observation` / `Control` / `Trace` — the pure sim↔stack contract. |
| **Interfaces** (`avsectester/backend.py`) | `WorldBackend`, `AVStack`, and one driving loop. `Runtime` adds ordered handlers and lifecycle management. |
| **CarlaBackend + ModularAVStack** (`avsectester/scenario.py`) | avcarla `CarlaClient`/`CarlaMobileActor`/`CarlaNpc` + an avstack `ModularDrivingPipeline` (perception → tracking → planning → control), built from config. |
| **NuRecBackend** (`avsectester/simulators/nurec.py`) | Local planar ego dynamics with a pluggable renderer (`StubRenderer` for offline use, `NuRecRenderer` for the external `nre-ga` service). Other actors follow recorded trajectories. Supports `checkpoint()`/`restore()`. |
| **AlpamayoAVStack** (`avsectester/stacks/alpamayo.py`) | Wraps the real **Alpamayo-1.5-10B** end-to-end policy as an `AVStack`: camera frames → trajectory `Control`. |
| **Attacks** (`avsectester/attacks/`) | Stage handlers through `Runtime`, native avstack hooks, physical patches and image insertion. Attack and defense share the handler contract. |
| **Selection and insertion** (`avsectester/scenarios/`, `insertion.py`) | Initial-case filters, stable actor bindings, explicit placements and camera visibility. |
| **Metric** (`avsectester/metric.py`) | Diffs a clean vs attacked `Trace` into a driving-impact verdict. |
| **Visualization** | `avsectester/metric.py:plot_impact` — the clean-vs-attacked impact plot (metric view); `avsectester/simulators/viz.py` — the **simulator-agnostic** scene pipeline (`record_run`, `detections_view`, `filmstrip`, `save_gif` over a generic RGB *view*), with per-simulator view adapters in `simulators/<sim>.py` (e.g. `simulators/carla.py`: `camera_view` for `ImageData`, `lidar_bev`). |

## Built on avstack + AlpaSim

Vendored under `third_party/` as git submodules (forked so the closed-loop pieces can live upstream):

- **avstack-core** — reconfigurable AV modules, geometry, sensors, registry/config, hooks
- **lib-avstack-carla** (`avcarla`) — closed-loop CARLA 0.9.16 bridge (client, actors, sensors)
- **avstack-api** — KITTI / nuScenes / CARLA dataset adapters

The NuRec + Alpamayo halves reuse NVIDIA's [AlpaSim](https://github.com/NVlabs/alpasim) pieces: the
`nre-ga` NuRec renderer serves reconstructed scenes from a separate container over gRPC, and
`alpasim_driver` provides Alpamayo in a dedicated Python 3.12 environment. This integration does
not run the full AlpaSim simulation. See [`docs/SETUP.md`](docs/SETUP.md).

## Status

**Early alpha.** The repository provides CARLA and NuRec backends, modular and Alpamayo stack
adapters, initial-case selection, insertion rendering and component logging. Backend/stack
combinations require compatible sensor inputs and model dependencies. Selection is available
through the Python API, while the YAML CLI runs configured CARLA experiments. Automated searches
for worst-case driving outcomes and broader reporting remain development work. See
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) and [test instructions](tests/README.md).

## License

MIT — see [`LICENSE`](LICENSE).
