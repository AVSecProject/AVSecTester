# AVSecTester

**Adversarial security-testing framework for autonomous-vehicle systems.**

AVSecTester runs an attack against a real AV pipeline — in closed-loop simulation or over a
dataset — and measures the effect on driving. Attacks act at named **seams** of the pipeline
(e.g. the detector's output); the framework runs a clean baseline and an attacked pass and
reports whether the attack propagated to a driving consequence, plus whether a defense mitigates it.

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
docker compose exec avsectester python scripts/smoke_carla_neural.py 40   # run the attack
```

What it does — a CARLA-trained **PointPillars** detector runs on a live **CarlaLidar**; a
fabricated detection is injected at the `perception_out` seam (no pixels touched):

```
[clean]    frames=40 ... mean_detections=5.2 brake_frames=0     # ego cruises, detects real NPCs
[attacked] frames=40 ... final_speed=0.0     brake_frames=39    # phantom → emergency stop
SMOKE: PASS (phantom forced an unsafe stop)
```

**See it:** `docker compose exec avsectester python scripts/record_carla.py 40` records per-frame
CARLA screenshots + a LiDAR bird's-eye view (real detections green, injected phantom red) into
`results/` — the phantom shows as a red box floating in the brake corridor with no points beneath
it. Details in [`docs/DOCKER.md`](docs/DOCKER.md).

No GPU/CARLA? The simulator-free path runs the same loop locally against a mock AV pipeline:

```bash
pip install -e third_party/avstack-core ".[dev]"
avsectester run configs/mock_experiment.yaml   # clean vs attacked vs attacked+defended
```

## Built on avstack

AVSecTester is the *security layer* on top of [avstack-lab](https://github.com/avstack-lab),
vendored under `third_party/` as git submodules:

- **avstack-core** — reconfigurable AV modules, geometry, sensors, registry/config
- **lib-avstack-carla** — closed-loop CARLA 0.9.15 bridge
- **avstack-api** — KITTI / nuScenes / CARLA dataset adapters

Six small core interfaces carry the whole framework — `Frame`, `Environment` (dataset⇄sim),
`System` (the AV pipeline; fires attacks at its seams), `Attack`/`Defense`, `Metric` — and we
never fork avstack internals. See [`docs/INTERFACE.md`](docs/INTERFACE.md) and
[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

## Layout

```
avsectester/   core · envs · attacks · defenses · metrics · reports ·
               config · viz · (search · agent · knowledge — planned)
third_party/   avstack-core · avstack-api · lib-avstack-carla · nuCarla   (git submodules)
configs/       example experiment configs
docs/          DOCKER.md · DEVELOPMENT.md · INTERFACE.md · SETUP.md
```

## Run an experiment

An experiment is one YAML config (environment + system + attack + defense + metric). The runner
runs a **clean** baseline and an **attacked** pass, scores them with the metric, and (if a
defense is declared) an **attacked+defended** pass to measure mitigation.

```bash
avsectester run configs/mock_experiment.yaml          # simulator-free (MockEnv/MockSystem)
avsectester run configs/carla_neural_experiment.yaml  # closed-loop CARLA + neural perception
```

It prints an impact report — did the attack induce braking + a stop the clean run never had —
and, with a defense, whether it was mitigated. The baseline `ScoreGateDefense` mitigates the
low-confidence object spoof.

Docker is the reproducible path for the CARLA stack; the manual conda install is in
[`docs/SETUP.md`](docs/SETUP.md).

## Status

**Early alpha.** The minimal core (environment → system → attack at seams → metric) is
implemented; the neural CARLA closed loop is **verified end-to-end** (the demo above); the mock
path drives the offline test suite. Richer escalation/attribution analysis, a scenario-search
engine, and an AI harness are planned — see [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

## License

MIT — see [`LICENSE`](LICENSE).
