# AVSecTester — Development Guide

AVSecTester is a **security layer built on top of [avstack](https://github.com/avstack-lab)**.
avstack supplies the AV stack, geometry, sensors, CARLA bridge, dataset adapters, and an
mmengine-style registry/config system; AVSecTester adds everything security-specific.

Its central abstraction is the **attack-escalation path**: an attack signal becomes a
component error, which propagates through the pipeline, which produces a driving
consequence. Every component below exists to inject, observe, quantify, or search that path.

This document is the single source of truth for the design. Section 1 is the overall
architecture; Section 2 covers each component — its **design goal** and **how it is
implemented / extended**.

---

## 1. Overall architecture

```
   natural-language      ┌────────────────────────────────────────────────────────────┐
   goal / research  ───► │              AI agent harness  (agent/, planned)            │
   paper                 │  generate ExperimentSpec · implement attack/defense code ·  │
                         │  orchestrate runs · read traces + DAG · summarize verdict   │
                         └───────────────┬───────────────────────────┬─────────────────┘
                                         │ spec                       │ critical cases
                                         ▼                            ▼
                         ┌───────────────────────────────┐ ┌───────────────────────────┐
   ExperimentSpec ─────► │   Experiment engine (core/)    │◄│ Testing handler (search/, │
   (hand-written or      │   build plugins from registry· │ │ planned): fuzzing + sim   │
    harness-generated)   │   run clean/attacked/defended  │ │ broker → critical cases   │
                         └───────────────┬───────────────┘ └───────────────────────────┘
                                         │  resolve_binding(profile) + attach at a seam
              attacks ─┐          defenses ─┐          monitors ─┐
                       │                    │                    │
                       ▼   HookAdapter → avstack pre/post hooks  ▼
                         ┌────────────────────────────────────────────┐
        backends adapt ► │        AV pipeline (ego AV, avstack)        │
        the stack        │  sensors → perception → tracking → … → ctrl │
                         └────────────────────────────────────────────┘
                                         │ per-frame records → Trace
                                         ▼
                    monitors → escalation DAG · metrics · reports · viz
                                         │
        ┌────────────────┬───────────────┼───────────────────┬──────────────────────┐
        ▼                ▼               ▼                   ▼                      ▼
   MockBackend       CarlaBackend    DatasetBackend    (planned) AlpaSim     (planned) HIL /
   (sim-free)        (closed-loop)   (offline replay)  AI-generative sim     Block Harbor VSEC
```

**Runtime flow of one experiment** (`core/engine.py`):

1. The engine builds the backend, attack, and defense from the `ExperimentSpec` via the
   registries (`{"type": name, ...}` configs).
2. It runs a **clean** pass, then an **attacked** pass, and — if a defense is declared — an
   **attacked+defended** pass. For each plugin it calls `plugin.resolve_binding(backend.profile())`
   to pick the seam, then `backend.attach(plugin, seam)`.
3. The backend drives the AV pipeline tick-by-tick; attacks/defenses fire as avstack
   pre/post hooks at their seam; monitors record per-stage I/O into a `Trace`.
4. `EscalationMetric` diffs the clean vs attacked traces into a scalar metric dict + an
   `EscalationDAG`; `reports` renders it; `viz` optionally records images/plots.

**Four integration decisions** hold the design together:

1. **Build on avstack, don't fork it.** avstack bridges CARLA 0.9.15 + mmdetection3d +
   datasets. We add only the security layer on top.
2. **Non-invasive interception via hooks.** avstack modules run `_apply_pre_hooks` /
   `_apply_post_hooks` (the `@apply_hooks` decorator, `HOOKS` registry). Our
   attacks/defenses/monitors are hook-shaped and attach at runtime — no edits to vendored code.
3. **Shared registry/config.** We reuse avstack's `Registry` so plugins build-from-config the
   same way avstack modules do, with a local shim when avstack isn't installed.
4. **One standard, many modes.** A single `ExperimentSpec` runs unchanged across componential
   (isolated module), closed-loop simulation, and (planned) hardware-in-the-loop — all
   attaching at the same backend seam, so verdicts are cross-mode comparable.

**Plugin boundary (extraction-ready).** `attacks/` and `defenses/` import only the contract
(`core/interfaces`, `core/plugin`, `core/binding`, `core/capability`, `core/threat_model`,
`config` registries) — never the engine, backends, or search. This keeps the plugin subtree a
clean `git filter-repo` extraction if it ever moves to its own repo.

**Package map**

| Package | Role | Status |
|---|---|---|
| `core` | seams, capabilities/bindings, plugin contract, threat model, experiment spec, escalation DAG, engine, interfaces, registries | implemented |
| `hooks` | bridge plugins onto avstack's pre/post-hook calling convention | implemented |
| `backends` | stack/mode adapters: `MockBackend`, `CarlaBackend`, `DatasetBackend` (stub); planned AlpaSim, HIL/VSEC | partial |
| `attacks` | attack plugins organized by **vector × method × binding** | implemented (LiDAR-spoofing, detection-manipulation) |
| `defenses` | defense/mitigation plugins | baseline (`ScoreGateDefense`) |
| `monitors` | execution traces + clean-vs-attacked diff | implemented |
| `metrics` | escalation metric → scalar dict + DAG | implemented |
| `reports` | root-cause + audit report | implemented |
| `viz` | per-frame image/data recording + timeline/comparison plots | implemented |
| `search` | testing handler: fuzzing + simulation broker | planned |
| `agent` | AI harness: spec generation, plugin authoring, orchestration | planned |
| `knowledge` | reusable vulnerability-path store / AV vuln dataset | planned |

---

## 2. Components — design goal & how to implement

### 2.1 Registries & config (`config/`)

**Design goal.** Let every plugin type (attacks, defenses, monitors, metrics, backends,
search) be built from a `{"type": name, ...}` config dict, so an `ExperimentSpec` is fully
declarative and the same class works from YAML, code, or a harness-generated spec.

**How it's implemented.** `config/registry.py` prefers avstack's OpenMMLab-style `Registry`
and falls back to a minimal shim (`register_module`, `get`, `build`, `__contains__`) when
avstack isn't importable, so imports never hard-fail in core-only environments. Six registries
are exposed: `ATTACKS, DEFENSES, MONITORS, METRICS, BACKENDS, SEARCH`.

**To extend.** Decorate a class with `@ATTACKS.register_module()` (etc.); it becomes buildable
by name. Nothing else is needed for the engine/CLI to find it.

### 2.2 Seams (`core/seams.py`)

**Design goal.** Name the *logical* interception points of an AV pipeline independently of any
concrete avstack module, so a plugin can target "the detector's output" without knowing which
class implements it.

**How it's implemented.** A `Seam(name, phase, stage, component, arg_index)` frozen dataclass;
`Phase` is `PRE`/`POST`. `SEAMS` registers the standard points: `raw_lidar`, `perception_input`,
`perception_out`, `tracking_out`, `planning_out`, `control_out`. `SEAM_ORDER` gives their
upstream→downstream order; `resolve_seam()` maps a name to a `Seam`.

**To extend.** Add an entry to `SEAMS` (and `SEAM_ORDER`) for a new pipeline point — e.g.
`localization_out` or `prediction_out` — then teach a backend to attach there (2.8) and add a
matching `Capability` (2.3).

### 2.3 Capabilities & bindings (`core/capability.py`, `core/binding.py`)

**Design goal.** Decouple *what a plugin needs* from *what a stack offers*, so one attack works
across different AV-stack settings (ground-truth passthrough vs neural detector vs raw sensor)
instead of being hard-wired to one seam.

**How it's implemented.**
- Each backend advertises a `StackProfile(seams, capabilities)` — which seams it exposes and
  which `Capability` affordances it provides (`GT_PERCEPTION`, `NEURAL_PERCEPTION`, `RAW_LIDAR`,
  `RAW_CAMERA`, `GRADIENTS`, `TRACKER`, `PLANNER`, `CONTROLLER`, `LOCALIZATION`, `V2X`).
- A plugin declares ranked `BindingSpec(seam, payload, requires, fidelity)` options. `resolve()`
  returns the highest-fidelity binding the profile supports, or raises `IncompatiblePlugin` with
  a readable reason. `seams_downstream_of(seam)` lets a defense be checked to sit at/after the
  attack it counters.

**To extend.** Add a `Capability` value and have the relevant backend advertise it in `profile()`;
give new plugins bindings that `requires` it. Same intent + multiple bindings ⇒ automatic
cross-stack portability.

### 2.4 Plugin contract (`core/plugin.py`, `core/interfaces.py`)

**Design goal.** One uniform contract for attacks and defenses: hook-shaped, self-describing,
with a lifecycle and declared bindings — so the engine, registries, and (future) inventory/
leaderboard treat them uniformly.

**How it's implemented.** `SecurityPlugin` carries `category`, `bindings`, `resolve_binding()`,
a resolved `seam` property, default-no-op lifecycle (`setup/validate/reset/teardown`), and
`describe()` (inventory metadata). `apply(data, *, ego_state, ctx) -> data` is the hook.
- `AttackBase(SecurityPlugin)` adds a `threat_model` and folds it into `describe()`.
- `DefenseBase(SecurityPlugin)` returns the (sanitized) payload but records a
  `DefenseOutcome(seam, frame, kept, dropped, flagged, reason)` into `ctx.defense_outcomes`
  via `record_outcome()` — so mitigation can be scored while the defense stays a plain hook.

**To extend.** See 2.9 (attacks) / 2.10 (defenses) — you subclass these, not `SecurityPlugin`
directly.

### 2.5 Hook adapter (`hooks.py`)

**Design goal.** Confine *all* knowledge of avstack's hook calling convention to one place, so
plugins keep a clean, avstack-agnostic `apply(payload, ego_state=…, ctx=…)` contract and the
subtree stays extractable.

**How it's implemented.** avstack pre-hooks must return `(args, kwargs)`; post-hooks are
re-splatted each iteration and must return `(value,)`. `HookAdapter` wraps a plugin for a seam
and produces exactly those shapes; `MonitorAdapter` observes without modifying. `RunContext`
(`run_id, frame, t, ego_state, ground_truth, trace, defense_outcomes`) carries per-tick state a
module's own signature doesn't provide; the backend calls `ctx.tick(...)` each step.
`attach(module, plugin, seam, ctx)` / `attach_monitor(...)` register the adapters at runtime.

**To extend.** Rarely touched. A genuinely new *kind* of interception (not pre/post on a
`BaseModule`) would add an adapter here; everything else reuses `attach`.

### 2.6 Threat model (`core/threat_model.py`)

**Design goal.** Make an attack *security-relevant* rather than arbitrary noise: state the
adversary's goal, knowledge, access, and a checkable success criterion, and let the engine
refuse runs that violate declared constraints.

**How it's implemented.** A pydantic `ThreatModel(goal, knowledge, access[], target,
capabilities[], constraints[], timing, success_criteria)`. `Knowledge` ∈ white/gray/black-box;
`AccessLevel` ∈ physical_environment / sensor / network_v2x / software / model.

**To extend.** Every attack sets a `threat_model` in its `__init__`. When implementing a survey
attack, encode the paper's assumptions here (attacker knowledge, sensor access, feasibility
constraints) even when the delivery is simulated — this is what keeps the inventory faithful.

### 2.7 Experiment specification (`core/experiment.py`)

**Design goal.** A single declarative, validated, reproducible description of a whole
experiment that runs unchanged across backends and modes.

**How it's implemented.** Pydantic models: `ExperimentSpec(name, system, scenario, attack?,
defense?, evaluation, reproducibility)`, where `ScenarioSpec.backend` is a `{"type": ...}`
build config, `AttackConfig` pairs an attack `spec` with a `ThreatModel`, and `EvaluationConfig`
lists metrics. Sub-configs are registry-buildable dicts.

**To extend.** New scenario knobs go on `ScenarioSpec`; new metric selection on
`EvaluationConfig`. The schema is the contract the (future) agent harness generates against.

### 2.8 Backends (`backends/`)

**Design goal.** Adapt any execution environment (simulator, dataset, hardware) to one uniform
surface so attacks/defenses/monitors/metrics above it never change.

**How it's implemented.** `Backend` ABC: `build/step/run/close`, plus `profile() -> StackProfile`
and `attach(plugin, seam)`. Implemented:
- `MockBackend` — simulator-free; ground-truth passthrough detector + tracker + a forward-
  collision reflex. Profile: `{perception_input, perception_out}`, `{GT_PERCEPTION, TRACKER}`.
  Used by the offline test suite.
- `CarlaBackend` — closed-loop via avcarla. Switchable perception: ground-truth passthrough, or
  **neural** (real `CarlaLidar` → CARLA-trained PointPillars). Optional RGB camera + `RunRecorder`.
  Profile depends on mode (neural ⇒ `{perception_out}`, `{NEURAL_PERCEPTION, RAW_LIDAR, TRACKER}`).
- `DatasetBackend` — offline KITTI/nuScenes replay via avapi (stub; Phase 2).
- `backends/common.py` — shared helpers (e.g. `forward_hazard`) used by more than one backend.

**To extend (new simulator / HIL).** Subclass `Backend`; implement the five methods; advertise
an accurate `profile()`; route `attach(plugin, seam)` to either a manual pre-loop (object-level
seams) or `hooks.attach()` on the right avstack module. Normalize coordinates at the boundary
(see §3). Planned backends: `AlpaSimBackend` (AI-generative sim) and a Block Harbor **VSEC** HIL
backend — both reuse the identical seams and escalation metric.

### 2.9 Attacks (`attacks/`)

**Design goal.** Grow a large, faithful inventory of AV attacks organized so that methods
sharing a delivery mechanism share code and one consistent set of bindings.

**How it's implemented — vector × method × binding.**
- An **attack vector** (`attacks/vector.py`, `AttackVector`) is the shared *mechanism* + the
  bindings a family inherits (a stateless toolkit; per-run state lives on the method).
- A **method** is a concrete goal (false positive, false negative, …) that composes a vector,
  sets `bindings = <Vector>.bindings`, and dispatches on the resolved seam.

Implemented vectors:
- `attacks/lidar_spoofing/` — `LidarSpoofingVector` (bindings: `raw_lidar` fid-3 requires
  neural+raw cloud, `perception_input` fid-1 requires GT). Methods: `ObjectSpoofingAttack`
  (false positive; alias `LidarSpoofAttack`), `ObjectRemovalAttack` (false negative). Raw-point
  primitives are declared but raise pending the optimization track.
- `attacks/detection_manipulation/` — `DetectionManipulationVector` (binding: `perception_out`,
  no capability requirement → any detector). Methods: `PhantomDetectionAttack` (inject),
  `DetectionRemovalAttack` (suppress).

**To add an attack.**
```python
@ATTACKS.register_module()
class MyAttack(AttackBase):
    category = "<vector name>"
    bindings = MyVector.bindings          # inherit the vector's seams
    def __init__(self, ...): self.vector = MyVector(); self.threat_model = ThreatModel(...)
    def validate(self, spec): ...          # optional: enforce threat-model constraints
    def apply(self, data, ego_state=None, ctx=None, **kw):
        seam = self.bound_seam             # dispatch to the vector primitive for this seam
        return self.vector.<op>(data, ...)
```
New modality (camera-patch, mmWave, GNSS, …) ⇒ a new vector package with its own `vector.py`
toolkit + method classes. Effects that live at the same seam (FP/FN/misclassify/displace) reuse
the same seam primitives across modalities.

### 2.10 Defenses (`defenses/`)

**Design goal.** Mitigate or flag attacks at or downstream of the attack seam, and report what
they did so mitigation is measurable.

**How it's implemented.** `ScoreGateDefense` (baseline) gates the detector input by confidence
(`perception_input`, requires `GT_PERCEPTION`) and records a `DefenseOutcome`. It is honest about
its scope: the neural-mode counterpart must bind at `perception_out` (a gap tracked for the next
defense).

**To add a defense.** Subclass `DefenseBase`, declare `bindings` (at/downstream of the attacks
it counters), return the sanitized payload, and call `self.record_outcome(ctx, DefenseOutcome(...))`.
The engine refuses to attach a defense upstream of the attack.

### 2.11 Monitors & traces (`monitors/`)

**Design goal.** Observe per-stage pipeline I/O without changing it, so clean and attacked runs
can be diffed into an escalation path.

**How it's implemented.** `Trace` collects `ComponentIO(frame, stage, component, outputs)`
records (appended by `MonitorAdapter` or by the backend directly); `build_trace(records, run_id)`
assembles a `Trace` from per-frame record dicts.

**To extend.** Attach a `MonitorAdapter` at any seam via `hooks.attach_monitor`, or have a
backend push richer records (speeds, brake flags, detections) that the metric reads.

### 2.12 Escalation DAG & metric (`core/escalation.py`, `metrics/`)

**Design goal.** Turn a pair of traces into (a) a scalar verdict — did the attack activate,
propagate, and cause a driving consequence — and (b) an explainable graph of the propagation.

**How it's implemented.** `Stage` enumerates the pipeline stages (`ATTACK_SURFACE, SENSOR,
PERCEPTION, LOCALIZATION, TRACKING, FUSION, PREDICTION, PLANNING, CONTROL, SAFEGUARD,
CONSEQUENCE`). `EscalationDAG` (nodes/edges over a networkx `DiGraph`) exposes `root_cause()` and
`consequence_paths()`. `EscalationMetric.compute(clean, attacked)` finds the first per-stage
divergence, chains them into a DAG, and returns `{"metrics": {activated, propagation_depth,
reached_consequence, stopped, escalated, ...}, "dag": ...}`.

**To add a metric.** Implement `MetricBase.compute(clean, attacked, **kw) -> dict`, register it in
`METRICS`, and list it in `EvaluationConfig`.

### 2.13 Reports (`reports/`)

**Design goal.** A human-readable audit of what happened and why — root cause, propagation,
metrics, and (if run) mitigation.

**How it's implemented.** `render_report(ExperimentResult)` produces markdown from the metric
dict + DAG + traces.

### 2.14 Visualization & recording (`viz/`)

**Design goal.** Record a run as inspectable images + data, and plot the analysis, so a result
can be reviewed and shared.

**How it's implemented.** `RunRecorder` (duck-typed onto a backend via `set_recorder`) writes
per-frame BEV + RGB frames and `records.jsonl`, and a `timeline.png`; `compare_runs()` overlays
clean vs attacked. Uses matplotlib (Agg) + PIL (the `viz` extra).

### 2.15 Engine (`core/engine.py`)

**Design goal.** Orchestrate the paired clean/attacked/defended passes and produce the result —
backend-agnostic and stack-agnostic.

**How it's implemented.** `ExperimentRunner` builds plugins from registries, resolves each
plugin's binding against `backend.profile()`, attaches at the resolved seam (rejecting a defense
upstream of the attack), runs the passes, and returns an `ExperimentResult(metrics, dag,
clean/attacked/defended traces, mitigated)`.

### 2.16 CLI (`cli.py`)

**Design goal.** A thin operator entry point.

**How it's implemented.** `typer` app: `version`, `registry` (list plugins), `validate <spec>`,
`run <spec> [report]` (exit code reflects `escalated`).

### 2.17 Models & datasets (`models/`, `scripts/`, `third_party/nuCarla`, `data/`)

**Design goal.** Put genuine CARLA-trained perception in the loop, and provide real CARLA traces
to check model behavior — without committing large binaries.

**How it's implemented.**
- `scripts/fetch_models.sh` downloads all reachable avstack CARLA-trained checkpoints (2 LiDAR
  PointPillars + 5 2D-camera faster/cascade-RCNN) into `models/` and symlinks the two vendored
  mmdet roots at them; the LiDAR (`work_dirs/`) and 2D (`work_dirs_2d/`) trees are kept separate
  because avstack's loader probes the 2D root first.
- `scripts/verify_models.py` loads every model + one forward pass (7/7).
- `scripts/eval_camera_nucarla.py` runs the camera models on real nuCarla images (detection-rate
  + score stats + annotated montages) — the "good CARLA traces" check.
- `third_party/nuCarla` (submodule) + `data/nuCarla` (git-ignored symlink) supply the
  nuScenes-format CARLA dataset. All weights and datasets are git-ignored.

### 2.18 Planned components

- **Testing handler (`search/`).** Design goal: pressure-test risk instead of one hand-picked
  scenario. Implement as an evolutionary/search loop over the scenario + attack-parameter space
  with escalation as fitness, behind a simulation broker so the same loop drives any backend;
  each generated case becomes an `ExperimentSpec` variant.
- **AI agent harness (`agent/`).** Design goal: drive the framework from intent. Implement agents
  that generate/validate an `ExperimentSpec` from a goal or paper, author or adapt attack/defense
  plugins against the stable hook seam, orchestrate the runs, and summarize the verdict — all
  behind the same schema-validation and tests as hand-written artifacts.
- **Knowledge / vuln dataset (`knowledge/`).** Design goal: accumulate reusable vulnerability
  paths. Implement as a store of escalation DAGs + specs + verdicts, keyed for retrieval and
  leaderboards.

---

## 3. Coordinate frames (correctness-critical)

CARLA is **left-handed** (X-fwd, Y-right, Z-up); mmdet3d/KITTI/nuScenes are **right-handed**. All
frame conversions go through avstack `geometry` (reuse `CarlaReferenceFrame` / `refchoc` rather
than hand-rolling), with explicit round-trip tests — silent frame bugs corrupt attack results
without failing loudly. Each new simulator or HIL bridge normalizes to the same convention at its
backend boundary.

## 4. Conventions

- **Env / tests.** conda env `avsec` (Python 3.10); `python -m pytest tests -q` for the offline
  suite; `ruff check avsectester tests scripts` must pass. Slow GPU/CARLA tests are gated.
- **Plugins import only the contract**, never engine/backends/search (keeps the subtree
  extractable).
- See `docs/SETUP.md` for installation and the full env, and `dev/PLAN.md` (local, git-ignored)
  for the running task checklist.
