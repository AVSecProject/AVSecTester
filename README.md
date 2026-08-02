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
- **lib-avstack-carla** — closed-loop CARLA 0.9.13 bridge
- **avstack-api** — KITTI / nuScenes / CARLA dataset adapters

Attacks, defenses, and monitors attach to avstack pipelines as **pre/post hooks** — no
forking of avstack internals. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Layout

```
avsectester/   core · backends · attacks · defenses · monitors ·
               metrics · search · reports · agent · knowledge
third_party/   avstack-core · avstack-api · lib-avstack-carla   (git submodules)
configs/       example experiment specs
docs/          ARCHITECTURE.md · SETUP.md
```

## Quickstart

```bash
git clone --recurse-submodules <repo> && cd AVSecTester
conda create -y -n avsec python=3.10 && conda activate avsec
pip install -e ".[dev]"          # core-only; full avstack stack: see docs/SETUP.md
pytest
avsectester version
```

## Run an experiment

An experiment is one YAML spec (system + scenario + attack + defense + metrics). The engine
runs a **clean** baseline and an **attacked** pass, scores them with the escalation metric,
and (if a defense is declared) an **attacked+defended** pass to measure mitigation.

```bash
# CARLA-free: uses MockBackend (needs the full avstack stack, no simulator)
avsectester run configs/mock_experiment.yaml

# closed-loop CARLA (needs a CARLA 0.9.15 server on :2000 — see docs/SETUP.md)
avsectester run configs/carla_experiment.yaml
```

Both print an **attack-escalation report** — verdict, activation/propagation/persistence/
safety metrics, and the `attack_surface → perception → tracking → control → consequence`
DAG with per-stage evidence. The same LiDAR-spoof phantom escalates to an unsafe stop on
either backend; the baseline `ScoreGateDefense` mitigates it.

Runnable demos live in [`scripts/`](scripts/) (closed-loop drive, attack escalation, smoke tests).

Status: **early alpha** — end-to-end engine (spec → backend → attack/defense hooks → escalation
DAG → report) works on `MockBackend` and `CarlaBackend`. See `docs/ARCHITECTURE.md` for design.

## License

MIT — see [`LICENSE`](LICENSE).
