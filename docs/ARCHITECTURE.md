# AVSecTester Architecture

AVSecTester is a **security layer built on top of [avstack](https://github.com/avstack-lab)**.
avstack supplies the AV stack, geometry, sensors, CARLA bridge, dataset adapters, and an
mmengine-style registry/config system; AVSecTester adds everything security-specific.

```
   natural-language      ┌──────────────────────────────────────────────────────────┐
   goal / research  ────►│              AI agent harness  (agent/)                    │
   paper                 │  generate structured ExperimentSpec · implement / adapt    │
                         │  attack & defense code · pick mode+simulator · orchestrate │
                         │  runs · read traces + DAG · summarize verdict              │
                         └───────────────┬──────────────────────────┬────────────────┘
                                         │ spec                      │ critical cases
                                         ▼                           ▼
                         ┌──────────────────────────────┐ ┌──────────────────────────┐
                         │   Experiment engine (core)    │◄│  Testing handler (search)│
   ExperimentSpec ──────►│   spec · threat model ·       │ │  fuzzing engine + sim    │
   (hand-written or      │   escalation DAG · registries │ │  broker, LLM-assisted →  │
    harness-generated)   │                               │ │  critical scenarios      │
                         └───────────────┬───────────────┘ └──────────────────────────┘
                                         │
                         attacks ─┐   defenses ─┐   monitors ─┐
                                  │             │            │
                                  ▼   pre/post hooks (HOOKS)  ▼
                         ┌───────────────────────────────────────┐
        backends adapt ─►│           AV pipeline (ego AV)        │
                         │  perception→tracking→…→planning→ctrl  │
                         └───────────────────────────────────────┘
                                         │ traces
                                         ▼
                         metrics · reports · knowledge (AV vuln dataset)
                                         │
        ┌────────────────┬───────────────┼────────────────────┬─────────────────────┐
        ▼                ▼               ▼                    ▼                     ▼
   componential      CARLA           AlpaSim          other simulators     Block Harbor VSEC
   (isolated       (avcarla,      (AI-generative                          (hardware-in-the-
    module)        closed-loop)    scenarios)                              loop, real ECUs)
```

## Key integration decisions

1. **Build on avstack** (confirmed). avstack already bridges CARLA 0.9.15 + mmdetection3d +
   datasets; its `security-sandbox` is stale, so the security layer is greenfield.
2. **Non-invasive interception via hooks.** avstack modules run `_apply_pre_hooks` /
   `_apply_post_hooks` (the `@apply_hooks` decorator, `HOOKS` registry). AVSecTester
   attacks/defenses/monitors are hook-shaped callables (`hook(*io) -> io`) attached to
   pipeline modules — **we do not fork avstack internals**. Metrics already work this way
   in avstack (`MetricsHook`).
3. **Shared registry/config.** We reuse avstack's `Registry` so plugins build-from-config
   the same way avstack modules do, with a local fallback shim when avstack isn't installed.
4. **One standard, many modes.** A single `ExperimentSpec` runs unchanged across the testing
   modes — componential (isolated module), closed-loop simulation (simulator-agnostic), and
   hardware-in-the-loop — all attaching at the same backend seam. Today `CarlaBackend`
   (closed-loop) and `MockBackend` (simulator-free) are implemented.

## AI agent harness (`agent/`)

The harness makes the framework largely automated and moves it toward a push-button experience.
It sits above the engine and drives it from intent rather than from a hand-written spec.

- **Generate structured input.** Given a natural-language goal or a research paper, agents draft
  a valid `ExperimentSpec` (target system config, scenario, threat model, attack, defense,
  metrics) that the engine can run directly. The structured schema is the contract, so a
  generated spec is validated and reproducible like any other.
- **Implement or adapt code when needed.** If an attack or defense does not yet exist as a plugin,
  agents write or adapt it against the stable hook seam (`apply(data, ego_state=…) → data`) and
  register it, so a research idea becomes a runnable, registry-buildable plugin without forking
  the AV stack.
- **Select mode, simulator, and scenarios**, orchestrate the clean / attacked / defended runs,
  read the diagnosis traces and escalation DAG, and summarize what happened and why.
- **Validation gates.** Harness-produced artifacts pass the same schema validation, budget checks,
  and tests as hand-written ones before a verdict is trusted or indexed.

## Testing handler (`search/`)

A pluggable handler that generates the *cases* the engine runs, so security risk is pressure-
tested rather than checked on a single hand-picked scenario.

- **Fuzzing engine.** Evolutionary / search-based generation over the scenario and attack-
  parameter space (route, traffic, weather, timing, attacker power) to find the conditions under
  which an attack escalates — and the boundary where it stops.
- **Simulation broker.** Abstracts "give me the next scenario to run" from any one simulator, so
  the same search loop drives CARLA, AlpaSim, or another engine through the backend interface.
- **LLM / AI assisted.** The handler can be steered by the AI harness: agents propose critical
  or adversarial scenarios (edge cases, corner geometries, worst-case timings) and seed or bias
  the search, combining learned priors with systematic exploration.
- **Feeds the engine.** Each generated case becomes an `ExperimentSpec` variant; results and their
  escalation DAGs flow back to rank cases by severity and into the vulnerability dataset.

## Simulator bridges

Simulation is **simulator-agnostic** behind the backend interface (`Backend` ABC:
`build / step / run / close` + `add_perception_hook`). A new simulator is a new backend; nothing
above it changes.

- **CARLA** — `CarlaBackend`, the closed-loop reference via avcarla (implemented).
- **AlpaSim (and other AI-generative engines)** — a planned `AlpaSimBackend` that bridges an
  AI-generative scenario simulator, so scenarios can be both realistic and automatically
  diversified. It adapts the simulator's sensor/actor/step API to the same `Backend` contract and
  reuses the identical attack/defense/monitor seams and escalation metric.
- **Coordinate frames** are normalized per backend (see below), so a bridge's handedness /
  units are contained at the boundary.

## Hardware-in-the-loop interface

HIL testing attaches at the **same backend seam** so one standard covers real hardware.

- **Block Harbor VSEC** (`https://vsec.blockharbor.io/`) — a planned HIL backend that bridges real
  ECUs, buses, and vehicle hardware. The user configures the hardware under test and the benchmark
  setting; the system exercises it either as an isolated component or inside a closed-loop
  simulated drive, and generates the same audit report and escalation DAG.
- **Why the same seam.** Because attacks, defenses, monitors, and the escalation metric live above
  the backend interface, a verdict produced in simulation and one produced on hardware are
  directly comparable — this is what makes cross-mode consistency checkable.

## Package map

| Package | Role | PLAN phase |
|---|---|---|
| `core` | experiment spec, threat model, escalation DAG, interfaces, registries | 1 |
| `backends` | mode/simulator adapters: CARLA · Mock · (planned) AlpaSim · (planned) HIL/VSEC | 2 |
| `attacks` | attack plugins + engine (hook-injected) | 3 |
| `monitors` | instrumentation, traces, clean-vs-attacked diff → escalation DAG | 4 |
| `defenses` | defense/mitigation plugins | 5 |
| `metrics` | activation/targeting/persistence/safety/detectability/mitigability/practicality | 6 |
| `search` | testing handler: fuzzing engine + simulation broker, LLM-assisted critical-case generation | 7 |
| `reports` | root-cause attribution + audit reports | 8 |
| `agent` | AI harness: spec generation, attack/defense implementation, run orchestration, validation gates | 9 |
| `knowledge` | reusable vulnerability-path store / AV vulnerability dataset | 10 |

## Coordinate frames (correctness-critical)

CARLA is **left-handed** (X-fwd, Y-right, Z-up); mmdet3d/KITTI/nuScenes are **right-handed**.
All frame conversions go through avstack `geometry/refchoc`, with explicit round-trip tests —
silent frame bugs corrupt attack results without failing loudly. Each new simulator or HIL bridge
normalizes to the same convention at its backend boundary.

See `dev/PLAN.md` (local, gitignored) for the full task checklist.
