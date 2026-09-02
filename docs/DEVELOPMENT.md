# AVSecTester — Development Guide

AVSecTester is an **adversarial security-testing framework for autonomous-vehicle systems**,
built as a thin layer on [avstack](https://github.com/avstack-lab) (the AV stack, geometry,
sensors, CARLA bridge, dataset adapters). It runs an attack against an AV pipeline — in
simulation or over a dataset — and measures the effect on driving.

Concept-first: Section 1 is the architecture; Section 2 covers each component (design goal,
then implementation). **[Now]** = built; **[Planned]** / **[TODO]** = intended. For exact
signatures see [`docs/INTERFACE.md`](INTERFACE.md).

---

## 1. Architecture

The whole framework is six small interfaces in `core`, and a runtime loop over them:

```
   run(env, system):
     frame = env.reset()
     while not done:
        outcome = system.process(frame)        # attacks fire at its seams
        trace.add(outcome.record)
        frame, done = env.step(outcome.control) # simulation uses it; dataset ignores it

   Environment ──frames──▶ System ──control──▶ Environment        Metric(clean, attacked)
   (MockEnv / CarlaEnv /   (MockSystem / CarlaSystem;                     ▲
    DatasetEnv[Planned])    fires Attack/Defense at Seams)   Trace ───────┘
```

**The six core types** (`core/`):
- **`Frame`** — one time-step of AV data (sensors, ego, calibration, ground_truth, meta). The
  single unit that flows; dataset and simulation both produce it.
- **`Environment`** — a sequential source of frames (`reset`/`step`). **The only thing that
  differs between dataset and simulation:** a dataset ignores the control fed back; a simulator
  applies it and advances the world.
- **`System`** — the AV pipeline under test. It processes a frame into control and **fires
  attacks/defenses at its seams by calling their `apply` directly** (`System.fire`).
- **`Seam`** — the statically-named injection points (`raw_lidar`, `perception_out`, …). A seam
  is just a name; the system knows how to fire plugins there — so *seam and hook are one thing*
  and there is no separate hook object.
- **`Attack`** (offline `prepare(data)→artifact` + runtime `apply(payload, ctx)`) / **`Defense`**
  (runtime `apply`).
- **`Metric`** — `compute(clean, attacked) → dict`.

**Principles.**
1. **Minimal, single-purpose interfaces.** Six types; each does one thing. Everything else is an
   implementation of one of them.
2. **AV-stack agnostic.** Attacks/defenses/metrics never import an AV model — they act at named
   seams and read a `Frame`. A modular stack, an end-to-end model, and a dataset are
   interchangeable behind `Environment`/`System`.
3. **Two-part attack.** Offline optimization (data → artifact) is separate from runtime
   deployment (`apply` at seams), so static and optimized attacks share one deployment path.
4. **One bridge for dataset + simulation.** `Environment.reset/step`; simulation feeds control
   back, replay ignores it.
5. **Plugins import only the contract** (`core` + `config`), so `attacks/`/`defenses/` stay an
   extractable subtree.

**Package map**

| Package | Role | Status |
|---|---|---|
| `core` | the six interfaces + runner | [Now] |
| `envs` | `MockEnv`/`MockSystem`, `CarlaEnv`/`CarlaSystem`; `DatasetEnv` | [Now] mock; [unverified] CARLA; [Planned] dataset |
| `attacks` | attack plugins by vector (lidar-spoofing, detection-manipulation) | [Now] |
| `defenses` | defense plugins (`ScoreGateDefense`) | [Now] baseline |
| `metrics` | `ImpactMetric` | [Now] |
| `reports` | markdown report of a `Result` | [Now] |
| `config` | registries (`ENVIRONMENTS/SYSTEMS/ATTACKS/DEFENSES/METRICS`) | [Now] |
| `viz` | per-frame image/data recording (standalone) | [Now] |
| `search` / `agent` / `knowledge` | scenario search / AI harness / vuln store | [Planned] |

---

## 2. Components

### 2.1 Core interfaces (`core/`) — [Now]

**Design goal.** Carry the whole framework in the fewest, smallest contracts (§1 + INTERFACE.md).
The runner is the only orchestration: `run(env, system)` drives an environment through a system
and collects a `Trace`; `run_experiment` runs paired clean/attacked/defended passes scored by a
`Metric`, returning a `Result`.

### 2.2 Environments & systems (`envs/`)

**Design goal.** Adapt any world (simulator, dataset) and any AV pipeline to the
`Environment`/`System` pair, so attacks/metrics never change across them.

- **[Now] `MockEnv` + `MockSystem`** — simulator-free closed loop: a 1-D kinematic ego + static
  background vehicles (env) and a passthrough detector + tracker + brake reflex (system), with
  attacks fired at `perception_input`/`perception_out`. Drives the offline test suite.
- **[Now, unverified] `CarlaEnv` + `CarlaSystem`** — closed-loop CARLA via avcarla: the env owns
  world/actors/sensors/control; the system owns perception (ground-truth passthrough **or** a
  real neural detector on a CarlaLidar cloud) + tracking + route-follow PID + brake reflex.
  Ported to the new interface; **not yet re-verified against a live CARLA server.**
- **[Planned] `DatasetEnv`** — replay nuCarla/KITTI frames (ignores control); the offline source
  for `Attack.prepare`.

**Extend.** Implement `Environment` (frames) and `System` (`process` + `seams`). An end-to-end
model exposes only `{raw_*, control_out}` seams; a plugin declaring a seam the system doesn't
expose fails `attach` loudly. Normalize coordinates at the boundary (§3).

### 2.3 Attacks (`attacks/`)

**Design goal.** A faithful inventory. Each attack is a *vector* (shared mechanism) × *method*
(goal), with the two-part contract: offline `prepare` (data → artifact) + runtime `apply`.

- **[Now]** `lidar_spoofing/` — `ObjectSpoofingAttack` (false positive), `ObjectRemovalAttack`
  (false negative) at `perception_input`; `detection_manipulation/` — `PhantomDetectionAttack`,
  `DetectionRemovalAttack` at `perception_out`.
- All current attacks are static (`prepare` no-op); the offline half earns its keep once an
  optimized attack (patch/points against a victim model) is added.

**Add an attack:** subclass `Attack`, set `seams`, implement `apply(payload, ctx)` (ego from
`ctx.frame.ego`), `@ATTACKS.register_module()`. See INTERFACE.md §5.

### 2.4 Defenses (`defenses/`)

**Design goal.** Sanitize/mitigate at a seam. **[Now]** `ScoreGateDefense` gates the object-level
input by confidence (`perception_input`). **[TODO]** a neural-mode `perception_out` defense; and
defenses that require *modifying* the stack (robust fusion / retrained detector), not a hook.

### 2.5 Metrics (`metrics/`) & reports

**Design goal.** Score a run. **[Now]** `ImpactMetric` compares clean vs attacked traces (did the
attack induce braking + a stop the clean run never had); `reports.render_report` renders a
`Result`. **[Planned]** escalation/attribution analysis (the DAG was removed; its replacement is
to be designed by the project owner) and attack-attribute assessment (precision / continuity /
robustness) over augmentation sweeps.

### 2.6 Models & datasets (`models/`, `scripts/`, `third_party/nuCarla`, `data/`)

**[Now]** `scripts/fetch_models.sh` (all avstack CARLA checkpoints), `verify_models.py` (7/7
load), `eval_camera_nucarla.py` (camera models on real nuCarla images). `third_party/nuCarla` +
git-ignored `data/nuCarla` supply the nuScenes-format CARLA dataset. Weights + data are git-ignored.

### 2.7 Planned — search, AI harness, knowledge

`search/` (scenario/attack search over the Environment/System interface), `agent/` (generate
specs + interface-compliant plugins behind a CI guardrail), `knowledge/` (reusable vuln store).

---

## 3. Coordinate frames (correctness-critical)

CARLA is left-handed; mmdet3d/KITTI/nuScenes are right-handed. All conversion goes through
avstack `geometry` (reuse `CarlaReferenceFrame`, not hand-rolled), normalized at each
environment's boundary — silent frame bugs corrupt results without failing loudly.

## 4. Conventions

- **Env / tests.** conda env `avsec` (Python 3.10); `python -m pytest tests -q` (offline suite);
  `ruff check avsectester tests scripts` must pass. GPU/CARLA tests are gated.
- **Plugins import only `core` + `config`** — never envs/runner — keeping the subtree extractable.
- See `docs/SETUP.md` for installation.
