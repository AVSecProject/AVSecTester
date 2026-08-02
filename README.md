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
python3.10 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # core-only; full avstack stack: see docs/SETUP.md
pytest
avsectester version
```

Status: **pre-alpha scaffold.** See `docs/ARCHITECTURE.md` for design and the development
plan for the phased roadmap.

## License

MIT — see [`LICENSE`](LICENSE).
