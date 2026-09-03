# AVSecTester

**Closed-loop adversarial stress-testing framework for autonomous-vehicle systems.**

Given an AV stack, AVSecTester audits known attack mechanisms and reports feasible
vulnerabilities, root causes, realistic impacts, and candidate mitigations. Unlike AV
testing tools that focus on functional safety or environmental robustness, it treats
adversarial attacks as **first-class system entities** and evaluates their activation,
targeting, persistence, cross-component propagation, detectability, mitigation, and
safety consequences — via an **attack-escalation-path** abstraction.

## Built on avstack

AVSecTester is the *security layer* on top of [avstack-lab](https://github.com/avstack-lab),
vendored under `third_party/` as git submodules:

- **avstack-core** — reconfigurable AV modules, geometry, sensors, registry/config, hooks, RSS metric
- **lib-avstack-carla** — closed-loop CARLA 0.9.15 bridge
- **avstack-api** — KITTI / nuScenes / CARLA dataset adapters

Attacks/defenses act at named **seams** of an AV pipeline (`System`) running in an
**environment** (simulation or dataset) — no forking of avstack internals. Six small core
interfaces (`Frame`, `Environment`, `System`, `Attack`/`Defense`, `Metric`) carry the whole
framework; see [`docs/INTERFACE.md`](docs/INTERFACE.md) and [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md).

## Layout

```
avsectester/   core · envs · attacks · defenses · metrics · reports ·
               config · viz · (search · agent · knowledge — planned)
third_party/   avstack-core · avstack-api · lib-avstack-carla · nuCarla   (git submodules)
configs/       example experiment configs
docs/          DEVELOPMENT.md · INTERFACE.md · SETUP.md · DOCKER.md
```

## Quickstart

Fastest path — a reproducible, torch-free Docker image that runs the end-to-end mock attack
(see [`docs/DOCKER.md`](docs/DOCKER.md)):

```bash
git clone --recurse-submodules <repo> && cd AVSecTester
git submodule update --init third_party/avstack-core
docker build -t avsectester:mock .
docker run --rm avsectester:mock          # clean vs attacked vs attacked+defended
```

Or a local install:

```bash
conda create -y -n avsec python=3.10 && conda activate avsec
pip install -e third_party/avstack-core ".[dev]"   # mock example; full GPU/CARLA stack: docs/SETUP.md
pytest
avsectester run configs/mock_experiment.yaml
```

## Run an experiment

An experiment is one YAML config (environment + system + attack + defense + metric). The runner
runs a **clean** baseline and an **attacked** pass, scores them with the metric, and (if a
defense is declared) an **attacked+defended** pass to measure mitigation.

```bash
# CARLA-free: MockEnv + MockSystem (needs the full avstack stack, no simulator)
avsectester run configs/mock_experiment.yaml

# closed-loop CARLA with neural perception (needs a CARLA 0.9.15 server on :2000 + weights)
avsectester run configs/carla_neural_experiment.yaml
```

It prints an impact report — did the attack induce braking + a stop the clean run never had —
and, with a defense, whether it was mitigated. The same phantom stops the ego on either
environment; the baseline `ScoreGateDefense` mitigates the low-confidence object spoof.

Runnable demos/smokes live in [`scripts/`](scripts/).

Status: **early alpha** — the minimal core (environment → system → attack at seams → metric)
runs on `MockEnv`/`MockSystem`; the CARLA env/system is ported but not yet re-verified live.
See `docs/DEVELOPMENT.md`.

## License

MIT — see [`LICENSE`](LICENSE).
