# AVSecTester — Development Guide

AVSecTester is a **security-testing framework for autonomous vehicles**, built as a thin
security layer on top of [avstack](https://github.com/avstack-lab). Its central abstraction is
the **attack-escalation path**: an attacker artifact becomes a component error, which
propagates through the AV stack, which produces a driving consequence. Every part of the system
exists to *inject*, *observe*, *quantify*, or *search* that path.

This guide is written **concept-first**: Section 1 gives the high-level architecture and the
principles that hold it together; Section 2 develops each component — its **design goal** first,
then **implementation details** and how to extend it. Throughout, **[Now]** marks what is
implemented today and **[Planned]** / **[TODO]** mark the intended design not yet built, so the
document doubles as a roadmap. For exact signatures of every interface named here, see
[`docs/INTERFACE.md`](INTERFACE.md).

---

## 1. High-level architecture

A **central experiment engine** sits in the middle. Around it are five subsystems it composes —
**AV stacks**, **attacks**, **defenses**, the **scenario engine**, and the **evaluation
engine** — connected by a shared **registry / config / interface** layer. Above everything is an
optional **AI harness** that drives the framework from intent while a CI/CD guardrail keeps
generated code inside the defined interfaces.

```
                         ┌──────────────────────────────────────────────────────────────┐
   goal / paper  ─────►  │   AI harness  [Planned]  — generate specs & interface-         │
                         │   compliant plugins; CI/CD guardrail on structure             │
                         └───────────────┬──────────────────────────────────────────────┘
                                         │ ExperimentSpec
                                         ▼
   ┌───────────────┐          ┌────────────────────────────┐          ┌──────────────────┐
   │   Scenario    │  cases   │   Central experiment engine │  signals │   Evaluation     │
   │   engine      │ ───────► │   build · run clean/attack/ │ ───────► │   engine         │
   │ (data + sim,  │          │   defended · orchestrate    │          │ (diagnosis →     │
   │  augmentation)│ ◄─────── │                            │ ◄─────── │  attributes →    │
   └───────────────┘  select  └───────┬────────────┬───────┘  request  │  mitigations)    │
                                       │            │                   └──────────────────┘
                       attacks ────────┤            ├──────── defenses
                                       ▼            ▼
                  read-write hooks (inject / optimize)   read hooks (diagnosis)
                                       │            │
                         ┌─────────────┴────────────┴──────────────┐
                         │   AV stacks (behind interfaces only)    │
                         │  modular (perception→…→control)  or     │
                         │  end-to-end / foundation-model driving  │
                         └─────────────────────────────────────────┘
                    backends: Mock · CARLA · Dataset · (Planned) AlpaSim · HIL
```

### Principles

1. **AV-stack agnostic — expose interfaces, hide internals.** Attacks, defenses, monitors, and
   metrics never import an AV model or a component class. They interact only through named
   **seams**. Two kinds of interface are exposed at each seam:
   - **Read hooks** — expose *diagnosis signals* (a layer's I/O) read-only, for the evaluation
     engine.
   - **Read-write hooks** — expose *attack/defense interfaces*: perturb a payload, and (for
     white-box optimization) read internal signals such as **gradients**.
   Because everything lives above these interfaces, a modular stack, an end-to-end model, and
   real hardware are interchangeable, and a verdict from one is comparable to another.
2. **One declarative standard, many modes.** A single `ExperimentSpec` runs unchanged across
   componential (isolated module), closed-loop simulation, and (planned) hardware-in-the-loop.
3. **Build on avstack, don't fork it.** avstack supplies the stack, geometry, sensors, the CARLA
   bridge, dataset adapters, and the registry/config system; we add only the security layer and
   attach non-invasively via avstack's own hook machinery.
4. **Separate generation from deployment, selection from augmentation, diagnosis from
   assessment.** An attack *artifact* is distinct from its *deployment strategy*; a *scenario
   selection* is distinct from *environmental augmentation*; *collecting* diagnosis signals is
   distinct from *assessing* attack attributes. Each split is a seam for future work.
5. **Plugins import only the contract.** `attacks/` and `defenses/` depend only on
   `core/{interfaces,plugin,binding,threat_model}` + `config` registries — never on
   the engine, backends, or scenario/evaluation engines — so the plugin subtree stays a clean,
   extractable unit.

### Package map

| Concept (Section) | Package(s) | Status |
|---|---|---|
| Central experiment engine (2.1) | `core/engine.py`, `cli.py` | [Now] |
| Registry / config / interfaces (2.2) | `config/`, `core/{interfaces,plugin,seams,binding,experiment,threat_model}` | [Now] |
| AV stacks & interfaces (2.3) | `hooks.py`, `backends/`, `models/`, `scripts/` | [Now] modular; [Planned] e2e / gradients / HIL |
| Attacks (2.4) | `attacks/` | [Now] LiDAR-spoofing, detection-manipulation |
| Defenses (2.5) | `defenses/` | [Now] baseline only |
| Scenario engine (2.6) | `core/experiment.py` (spec), `search/` | [Now] spec; [Planned] engine |
| Evaluation engine (2.7) | `monitors/`, `core/escalation.py`, `metrics/`, `reports/`, `viz/` | [Now] escalation; [Planned] attributes/augmentation/mitigation |
| AI harness (2.8) | `agent/`, CI | [Planned] |

---

## 2. Components

### 2.1 Central experiment engine

**Design goal.** Turn one declarative `ExperimentSpec` into a verdict by composing a stack, an
attack, a defense, a scenario, and metrics — agnostic to which backend or stack is underneath —
and by running the *paired* passes that make an attack's effect measurable.

**Implementation [Now].** `core/engine.py::ExperimentRunner`:
1. Builds the backend, attack, and defense from the registries (`{"type": name, ...}` configs).
2. Runs a **clean** pass, an **attacked** pass, and — if a defense is declared — an
   **attacked+defended** pass. For each plugin it checks compatibility
   (`plugin.check(backend.supported_seams())`) and attaches it at **every seam it declares**; a defense
   whose seams are all *upstream* of the attack is rejected.
3. Returns `ExperimentResult(metrics, dag, clean/attacked/defended traces, mitigated)`.

`cli.py` exposes `version`, `registry` (list plugins), `validate <spec>`, `run <spec> [report]`
(exit code reflects whether the attack escalated).

**Extend.** The engine is deliberately thin; new behavior (e.g. repetitions, augmentation
sweeps) is added by having it drive the scenario engine (2.6) rather than by growing the runner.

### 2.2 Registry, config, and interfaces

**Design goal.** Make systems, attacks, defenses, and metrics all *declarative and
interchangeable*: buildable by name from a config dict, and interacting only through stable
interfaces so implementations can be swapped without touching callers.

**Implementation [Now].**
- **Registries** (`config/registry.py`): reuse avstack's OpenMMLab-style `Registry`
  (`register_module` / `build` from `{"type": ...}`), with a local shim when avstack isn't
  importable. Six registries: `ATTACKS, DEFENSES, MONITORS, METRICS, BACKENDS, SEARCH`.
- **Experiment spec** (`core/experiment.py`): pydantic `ExperimentSpec(name, system, scenario,
  attack?, defense?, evaluation, reproducibility)`. Sub-configs are registry-buildable dicts;
  `AttackConfig` pairs an attack `spec` with a `ThreatModel`.
- **Threat model** (`core/threat_model.py`): the entity that makes an attack security-relevant —
  `goal, knowledge (white/gray/black-box), access[], target, capabilities[], constraints[],
  timing, success_criteria`. The engine can refuse runs that violate declared constraints.
- **Plugin contract** (`core/plugin.py`, `core/interfaces.py`): `SecurityPlugin` carries
  `category`, `seams` (the list of seams it hooks into), `check()`,
  `current_seam()`, lifecycle (`validate/reset`), and `describe()`. `AttackBase` adds
  `threat_model`; `DefenseBase` records a `DefenseOutcome`.

**Extend.** New plugin type ⇒ new registry + a small `*Base` interface. Everything downstream
(engine, CLI, harness) discovers it by name.

### 2.3 AV stacks (behind interfaces)

**Design goal.** Put a *real* AV stack under test while keeping the framework agnostic to its AI
model and component details. The stack exposes only **interfaces**: named seams for reading
diagnosis signals and for read-write attack/optimization access. This must cover both **modular
stacks** (perception → tracking → prediction → planning → control) and **end-to-end /
foundation-model driving** (sensors → policy → control), and eventually **hardware-in-the-loop**.

#### 2.3.1 Reusing avstack

**[Now].** avstack supplies the modular pipeline, geometry, sensors, the CARLA bridge, and
dataset adapters. We never fork it; we attach at runtime via its `@apply_hooks` machinery
(`_apply_pre_hooks` / `_apply_post_hooks`). Genuine CARLA-trained perception is in the loop:
`scripts/fetch_models.sh` pulls all reachable avstack CARLA checkpoints (2 LiDAR PointPillars +
5 2D-camera R-CNNs), `scripts/verify_models.py` confirms they load and run, and
`scripts/eval_camera_nucarla.py` validates the camera models on real nuCarla traces.

#### 2.3.2 Seams & compatibility — the interface layer

**Design goal.** Let a plugin target "the detector's output" without knowing which class
implements it, and let one plugin hook into several points at once.

**[Now].**
- **Seams** (`core/seams.py`): logical interception points — `raw_lidar`, `perception_input`,
  `perception_out`, `tracking_out`, `planning_out`, `control_out` — each a
  `Seam(name, phase, stage, component, arg_index)`; `SEAM_ORDER` fixes upstream→downstream order.
- **A backend advertises the set of seams it exposes** — just a `frozenset[str]` from
  `Backend.supported_seams()` (no separate capability type: e.g. a GT stack exposes
  `perception_input`, a neural stack doesn't — seam presence already encodes the difference).
- **A plugin declares a list of seams** (`SecurityPlugin.seams`). The engine attaches it at
  **every** declared seam the stack exposes, and `check_support()` (`core/binding.py`) fails
  loudly (`IncompatiblePlugin`) if a declared seam isn't exposed. This is what makes plugins
  **agnostic to model/component details** — they name seams, not classes — and lets one attack
  span several seams, dispatching on `ctx.seam` (via `current_seam(ctx)`). If a stack ever needs
  an affordance the seam name can't express (e.g. a *differentiable* forward), attach it as an
  attribute of the exposed seam rather than reintroducing a global capability set.

**Extend for end-to-end / foundation-model driving [Planned].** An e2e model has no internal
component seams — only input (raw sensors) and output (trajectory/control). It fits the same
abstraction: an `E2EBackend` exposes `{raw_lidar, raw_camera, control_out}` and *omits* the
intermediate seams. A plugin that declares an intermediate seam simply fails the compatibility
check (loudly) on an e2e
stack, while raw-sensor and output attacks port over unchanged. Candidate models: the nuCarla
BEV detectors and the CARLA end-to-end policies (TransFuser/InterFuser class), run in their own
env behind the backend boundary.

#### 2.3.3 Read hooks — diagnosis signals

**Design goal.** Expose each layer's I/O read-only so the evaluation engine can see *where* an
error appears and *how far* it propagates, without perturbing the run.

**[Now].** `hooks.py::MonitorAdapter` observes a seam and appends `ComponentIO(frame, stage,
component, outputs)` into the run `Trace`. `RunContext` carries per-tick `frame/t/ego_state/
ground_truth`. `attach_monitor(module, seam, ctx)` wires it.

**Extend [Planned].** Add read hooks at the new seams (localization, prediction) and richer
signals (feature maps, confidences, timing) so the evaluation engine's impacting-factor analysis
has more to observe.

#### 2.3.4 Read-write hooks — attack/optimization interfaces

**Design goal.** Let an attack both *inject* a perturbation and, for white-box optimization,
*read internal signals such as gradients* — through a defined interface, so the stack stays
agnostic and the attack never imports the model.

**[Now].** `hooks.py::HookAdapter` wraps an attack/defense at a seam, confining avstack's
calling convention (pre-hooks return `(args, kwargs)`; post-hooks return `(value,)`). It is
read-write on the payload.

**[Planned] gradient / white-box interface.** The differentiable-forward affordance is not yet
wired. The design: a backend that can expose a differentiable forward marks that seam differentiable;
the interface returns `∂(loss)/∂(input)` for an attacker-specified objective, so an optimization
attack (2.4) can do PGD/EoT **without importing the detector** — e.g. a `WhiteboxLidarDetector`
subclass that surfaces `self.model` gradients behind the interface. The stack exposes the
gradient interface; the attack owns the objective and the optimizer.

#### 2.3.5 Hardware-in-the-loop [TODO]

HIL attaches at the **same backend seam** so one standard covers real hardware. A Block Harbor
**VSEC** backend would bridge real ECUs/buses; because attacks/defenses/monitors/metrics live
above the backend interface, a simulated verdict and a hardware verdict are directly comparable.
Open questions: which seams a real ECU exposes, timing/synchronization, and safe fault
injection — deferred.

#### 2.3.6 Dataset & simulator interfaces (backends)

**Design goal.** Adapt any execution environment — simulator, dataset, hardware — to one uniform
surface (`Backend`: `build/step/run/close`, `supported_seams()`, `attach(plugin, seam)`).

**[Now].** `MockBackend` (simulator-free GT passthrough + tracker + forward-collision reflex;
used by the offline suite); `CarlaBackend` (closed-loop avcarla; switchable GT vs **neural**
perception, optional camera + `RunRecorder`). **[Stub]** `DatasetBackend` (offline KITTI/nuScenes
replay via avapi). **[Planned]** `AlpaSimBackend`. New backend ⇒ implement the five methods,
advertise an accurate `supported_seams()`, route `attach` to a manual pre-loop or `hooks.attach()`, and
normalize coordinates at the boundary (§3).

### 2.4 Attacks

**Design goal.** An attack takes **knowledge of the system and the scenario** and produces two
things: an **attack artifact** (the perturbation — spoofed points, an adversarial patch, a
fabricated detection) and a **deployment strategy** (where/when/how it is injected). It may need
to (a) read AV-stack interfaces to *optimize* the artifact (e.g. gradients), (b) access the
scenario/dataset because an attack may only work in specific scenarios, and (c) expose a
**parameter interface** for the variables the attacker controls. Physical sensor attacks
additionally require a **data-informed physical model**.

**Implementation [Now] — vector × method.**
- An **attack vector** (`attacks/vector.py`, `AttackVector`) is the shared delivery *mechanism*
  plus the `seams` a family inherits (a stateless toolkit).
- A **method** composes a vector, sets `seams = <Vector>.seams`, and dispatches on the firing
  seam. Implemented:
  - `attacks/lidar_spoofing/` — `LidarSpoofingVector` (seam `perception_input`, GT stacks only);
    `ObjectSpoofingAttack` (false positive), `ObjectRemovalAttack` (false negative).
  - `attacks/detection_manipulation/` — `DetectionManipulationVector` (seam `perception_out`,
    any detector); `PhantomDetectionAttack`, `DetectionRemovalAttack`.

Add an attack:
```python
@ATTACKS.register_module()
class MyAttack(AttackBase):
    category = "<vector>"
    seams = MyVector.seams                # the seams it hooks into (one or several)
    def __init__(self, param_a=..., param_b=...):     # (c) attacker-tunable parameters
        self.vector = MyVector(); self.threat_model = ThreatModel(...)
    def validate(self, spec): ...          # enforce threat-model constraints
    def apply(self, data, ego_state=None, ctx=None, **kw):
        return self.vector.<op>(data, ...) # act at self.current_seam(ctx)
```

**Where the richer concept maps [current vs planned].**
- **Artifact vs deployment.** Today `apply()` conflates a *fixed* artifact with its deployment
  (the hook is the deployment). The intended split adds an artifact-generation phase — `Attack.
  generate(system, scenario) -> Artifact` — that `apply()` then deploys, so optimized and static
  attacks share one deployment path. **[Planned].**
- **AV-stack access for optimization.** Generation may consume the differentiable/white-box interface (2.3.4)
  to optimize the artifact (PGD/EoT). The attack owns the objective; the stack owns the gradient
  interface. **[Planned].**
- **Scenario/dataset access.** An attack may declare *scenario preconditions* (it only works at
  certain geometries/timings); the scenario engine (2.6) filters or generates matching cases,
  and generation may read scenario/sensor data. **[Planned].**
- **Parameter interface [Now, partial].** `__init__` kwargs are the attacker-controlled
  variables (e.g. `target_xyz`, `corridor`, `score`); the search engine tunes them. A typed
  parameter schema (bounds/space) for automated search is **[Planned]**.
- **Physical-process modeling + data-informed validation [Planned].** For physical sensor
  attacks (LiDAR point injection, camera patches), a naive artifact is not faithful — e.g. raw
  point injection does not reliably fool a neural detector. Such attacks must ship a **data
  generation pipeline that models the physical process** (how the injected signal actually
  appears to the sensor) **validated against real data** (nuCarla traces, measured domain gap).
  Until validated, the attack is labeled an *effect abstraction* (it injects the downstream error
  with feasibility metadata) rather than a signal-level attack — kept honest via the threat model.

### 2.5 Defenses

**Design goal.** Detect, sanitize, or mitigate an attack at or downstream of its seam, and
report what it did so mitigation is measurable — potentially requiring **direct modification of
the AV stack** (not just a hook), which is the open design question.

**Implementation.** **[Now]** `ScoreGateDefense` — a baseline confidence gate at
`perception_input` (GT stacks only) that records a `DefenseOutcome(kept, dropped,
reason)` into `ctx.defense_outcomes` while returning the sanitized payload. A defense declares
bindings at/downstream of the attacks it counters; the engine refuses upstream placement.

**[TODO].** (1) A neural-mode `perception_out` defense (the current gate is GT-only — the main
gap). (2) Defenses that require **modifying the stack** rather than hooking it — e.g. swapping in
a robust fusion module or a retrained detector — need a stack-modification interface beyond the
read-write hook; scope and safety of that interface are open.

### 2.6 Scenario engine

**Design goal.** Provide the *cases* an experiment runs, over **both datasets and simulation**,
from a **formal definition of the applicable scenario**: either *filter* matching scenarios out
of a dataset, or *generate* simulation scenarios from the description. Then **augment** each case
with external environmental variables the attacker does *not* control (weather, lighting, traffic
noise), and let an attacker *choose* a scenario by writing its specification.

**Implementation.**
- **[Now]** `core/experiment.py::ScenarioSpec` carries `backend`, `map`, `initial_conditions`,
  `participants`, `target_objects`, `seeds`, `repetitions`. `DatasetBackend`/`CarlaBackend` are
  the data/sim sources.
- **[Planned] scenario engine (`search/`).** (1) A **formal scenario schema** (a filterable /
  generatable description). (2) A **dataset filter** that selects matching frames/scenes (e.g.
  from nuCarla by class/geometry/weather). (3) A **simulation generator** that builds CARLA
  scenarios from the description. (4) An **augmentation layer** that adds environmental noise
  (weather, lighting, non-attacker traffic) to probe robustness. (5) A **fuzzing/search loop**
  over scenario × attack-parameter space with escalation as fitness, behind a **simulation
  broker** so one loop drives any backend. The attacker's own scenario spec is just a fully
  constrained description fed to the same engine.

### 2.7 Evaluation engine

**Design goal.** Go beyond a single pass/fail: **collect diagnosis signals from every layer**,
**analyze what makes an attack effective** by observing results under data augmentation, **assess
the attack's attributes** (precision, continuity, robustness, and related), and **propose
mitigations**.

**Implementation.**
- **[Now] diagnosis + escalation.** Read hooks (2.3.3) fill a `Trace`. `core/escalation.py`
  defines the pipeline `Stage`s and an `EscalationDAG` (`root_cause`, `consequence_paths`).
  `metrics/escalation.py::EscalationMetric.compute(clean, attacked)` finds the first per-stage
  divergence, chains it into a DAG, and returns `{activated, propagation_depth,
  reached_consequence, stopped, escalated, ...}`. `reports/` renders the audit; `viz/`
  (`RunRecorder`, `compare_runs`) records per-frame images/data and timeline/comparison plots.
- **[Planned] impacting-factor analysis.** Sweep the scenario-engine augmentations (weather,
  lighting, timing, distance) and correlate them with escalation to find *why* an attack works
  and its operating envelope.
- **[Planned] attack-attribute assessment.** Quantify attributes such as **precision** (how
  targeted the induced error is), **continuity** (whether it persists frame-to-frame),
  **robustness/generalizability** (does it survive augmentation and transfer across models),
  intensity, and distance-to-impact — reusing the AV-security survey's attribute vocabulary as
  the metric set.
- **[Planned] mitigation proposal.** From the root-cause DAG + attributes, suggest candidate
  defenses (e.g. the seam to gate, the corroborating sensor to add) and, where a defense plugin
  exists, run the defended pass to confirm.

New metric: implement `MetricBase.compute(clean, attacked, **kw) -> dict`, register in `METRICS`,
list it in `EvaluationConfig`.

### 2.8 AI harness

**Design goal.** Let AI drive the framework from intent (a goal or a paper) while a guardrail
guarantees it **cannot alter the fundamental project structure** and **can only produce code that
complies with the defined interfaces**.

**Implementation [Planned].**
- **Static CI/CD guardrail.** A pipeline that runs on every AI-produced change and rejects
  anything that touches protected structure (core interfaces, seam/registry definitions, the
  plugin-boundary rule) or breaks the offline suite / ruff. The interface contracts in 2.2 and
  the import-boundary rule are what make this checkable.
- **Interface-guided generation.** The harness prompts and constrains the model to author
  `ExperimentSpec`s and new attack/defense plugins *against the stable contracts only* — subclass
  `AttackBase`/`DefenseBase`, declare `seams`, implement `apply()`/`generate()` — and validates
  every artifact through the same schema checks and tests as hand-written code before a verdict is
  trusted or indexed. A generated plugin is a normal registry entry; nothing about the runtime
  trusts it more than a human's.

---

## 3. Coordinate frames (correctness-critical)

CARLA is **left-handed** (X-fwd, Y-right, Z-up); mmdet3d/KITTI/nuScenes are **right-handed**. All
frame conversions go through avstack `geometry` (reuse `CarlaReferenceFrame` / `refchoc` rather
than hand-rolling), with explicit round-trip tests — silent frame bugs corrupt attack results
without failing loudly. Each new simulator or HIL bridge normalizes at its backend boundary.

## 4. Conventions

- **Env / tests.** conda env `avsec` (Python 3.10); `python -m pytest tests -q` for the offline
  suite; `ruff check avsectester tests scripts` must pass. Slow GPU/CARLA tests are gated.
- **Plugins import only the contract** (`core/{interfaces,plugin,binding,threat_model}`
  + `config`), never engine/backends/scenario/evaluation — keeping the subtree extractable and the
  AI-harness guardrail enforceable.
- See `docs/SETUP.md` for installation and the full environment, and `dev/PLAN.md` (local,
  git-ignored) for the running task checklist.
