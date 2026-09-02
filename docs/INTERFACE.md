# AVSecTester — Interface Reference

This is the precise contract reference for AVSecTester. It lists **every interface** a
component author must implement or call, with exact signatures, the guarantees each side owes,
and minimal examples. `docs/DEVELOPMENT.md` explains *why* the architecture is shaped this way;
this document is the *what* — the contracts.

Conventions: signatures are given as they exist in code. **[Now]** = implemented and stable;
**[Planned]** = the intended contract, not yet in code (do not import). Types are the real ones
from `avsectester/…`.

**Contents**
1. Interface map
2. Plugin contracts — `SecurityPlugin`, `AttackBase`, `DefenseBase`, `MonitorBase`, `MetricBase`, `SearchStrategy`
3. Stack interfaces — `Backend`, `StackProfile` / `Capability`, `Seam`, `BindingSpec`, hook calling convention, `RunContext`
4. Data contracts — seam payloads, the per-frame record, `Trace` / `ComponentIO`, escalation DAG, metric output, recorder
5. Config & spec contracts — `Registry`, `ExperimentSpec`, `ThreatModel`
6. Engine & result contracts — `ExperimentRunner`, `ExperimentResult`, `render_report`
7. Planned interfaces — gradient/white-box, scenario engine, evaluation attributes, AI-harness guardrail

---

## 1. Interface map

| Interface | Kind | Module | Implemented by / Called by |
|---|---|---|---|
| `SecurityPlugin` | base class | `core/plugin.py` | superclass of every attack/defense |
| `AttackBase` | base class | `core/interfaces.py` | attack plugins |
| `DefenseBase` | base class | `core/interfaces.py` | defense plugins |
| `MonitorBase` | base class | `core/interfaces.py` | trace monitors |
| `MetricBase` | base class | `core/interfaces.py` | evaluation metrics |
| `SearchStrategy` | base class | `core/interfaces.py` | scenario/attack search [Planned use] |
| `Backend` | ABC | `core/interfaces.py` | Mock / CARLA / Dataset / (HIL) backends |
| `StackProfile`, `Capability` | data | `core/capability.py` | backends advertise; plugins require |
| `Seam`, `SEAMS`, `Phase` | data | `core/seams.py` | seam vocabulary |
| `BindingSpec`, `resolve` | data/fn | `core/binding.py` | plugins declare; engine resolves |
| hook convention (`HookAdapter`, `MonitorAdapter`, `attach`) | adapter | `hooks.py` | framework only |
| `RunContext` | data | `hooks.py` | backend fills; plugins read |
| per-frame **record** dict | data | `monitors/trace.py` | backend emits; engine consumes |
| `Trace`, `ComponentIO` | data | `monitors/trace.py` | evaluation |
| `EscalationDAG`/`Node`/`Edge`, `Stage` | data | `core/escalation.py` | metrics produce; reports consume |
| metric output dict | data | `metrics/escalation.py` | metrics return |
| `Registry` | registry | `config/registry.py` | plugin discovery |
| `ExperimentSpec` (+ sub-specs) | schema | `core/experiment.py` | the declarative contract |
| `ThreatModel` | schema | `core/threat_model.py` | attacks declare |
| `ExperimentResult` | data | `core/engine.py` | engine returns |

**Golden rule.** Attacks and defenses import **only** `core/{interfaces,plugin,binding,capability,
threat_model}` and `config`. They never import a backend, an avstack model, the engine, or the
scenario/evaluation engines. The framework talks to a plugin only through the methods below.

---

## 2. Plugin contracts

### 2.1 `SecurityPlugin` (base of all attacks & defenses) — [Now]

`core/plugin.py`. A plugin is a hook-shaped, self-describing component that declares *how it can
bind* to a stack.

```python
class SecurityPlugin(ABC):
    category: str = "generic"                       # taxonomy / inventory tag
    bindings: tuple[BindingSpec, ...] = ()          # ranked ways it can attach
    _binding: BindingSpec | None = None             # set by resolve_binding()

    # identity / inventory
    @property
    def name(self) -> str: ...                       # defaults to class name
    def describe(self) -> dict[str, Any]: ...         # {name, kind, category, bindings[...]}

    # binding resolution
    def resolve_binding(self, profile: StackProfile) -> BindingSpec   # raises IncompatiblePlugin
    @property
    def binding(self) -> BindingSpec | None           # resolved binding, or None
    @property
    def bound_seam(self) -> str | None                # resolved seam name, or None
    @property
    def primary_binding(self) -> BindingSpec | None   # highest-fidelity declared binding
    @property
    def seam(self) -> str | None                      # resolved seam else primary seam

    # lifecycle (default no-ops — override what you need)
    def setup(self, spec: ExperimentSpec) -> None
    def validate(self, spec: ExperimentSpec) -> None  # raise to reject an incompatible run
    def reset(self) -> None                           # clear per-run state between passes
    def teardown(self) -> None

    # the hook
    @abstractmethod
    def apply(self, data: Any, *args: Any, **kwargs: Any) -> Any
    def __call__(self, data, *args, **kwargs) -> Any   # == apply
```

**Contract.** `apply` is called as `apply(payload, ego_state=<state|None>, ctx=<RunContext|None>)`
and MUST return a payload of the same type it received (the seam's payload type — §4.1). It may be
called many times per run; use `reset()` to clear per-run state. `bindings` must be non-empty for
a plugin the engine attaches. Reading `bound_seam` before `resolve_binding()` returns `None`.

### 2.2 `AttackBase` — [Now]

`core/interfaces.py`. Adds a threat model.

```python
class AttackBase(SecurityPlugin):
    threat_model: ThreatModel                         # set in __init__
    # describe() additionally emits {"threat_model": {goal, knowledge, access, success_criteria}}
```

**Minimal attack.**
```python
@ATTACKS.register_module()
class MyAttack(AttackBase):
    category = "my_vector"
    bindings = (BindingSpec("perception_out", "detections", fidelity=1),)
    def __init__(self, target_xyz=(12,0,-1.5)):
        self.target_xyz = target_xyz
        self.threat_model = ThreatModel(goal="…", target="…", success_criteria="…")
    def apply(self, data, ego_state=None, ctx=None, **kw):
        ...            # perturb `data` at self.bound_seam and return it
        return data
```

### 2.3 `DefenseBase` (+ `DefenseOutcome`) — [Now]

`core/interfaces.py`. A defense returns the sanitized payload **and** records what it did.

```python
class DefenseBase(SecurityPlugin):
    category: str = "defense"
    def record_outcome(self, ctx: Any, outcome: DefenseOutcome) -> None   # duck-typed; no-op if ctx is None

@dataclass
class DefenseOutcome:
    seam: str
    frame: int = 0
    kept: int = 0
    dropped: list[Any] = []          # IDs removed
    flagged: list[Any] = []          # IDs marked suspicious
    reason: str = ""
```

**Contract.** `apply` returns the (possibly sanitized) payload; per tick it SHOULD call
`self.record_outcome(ctx, DefenseOutcome(...))` so mitigation is measurable. The engine attaches a
defense only at a seam **at or downstream of** the attack (`seams_downstream_of`, §3.4); an upstream
defense is rejected.

### 2.4 `MonitorBase` — [Now]

`core/interfaces.py`. Read-only instrumentation.

```python
class MonitorBase(ABC):
    @abstractmethod
    def observe(self, stage: str, component: str, data: Any) -> None    # record; never modify
```

### 2.5 `MetricBase` — [Now]

`core/interfaces.py`. Scores one or more (clean, attacked) traces.

```python
class MetricBase(ABC):
    name: str
    @abstractmethod
    def compute(self, clean: Any, attacked: Any, **kwargs: Any) -> dict[str, Any]
```

**Contract.** `compute(clean_trace, attacked_trace)` returns a JSON-serializable dict. The built-in
`EscalationMetric` additionally returns `{"metrics": {...}, "dag": EscalationDAG}` (§4.4).

### 2.6 `SearchStrategy` — [Planned use]

`core/interfaces.py`. Closed-loop case search (scenario engine).

```python
class SearchStrategy(ABC):
    @abstractmethod
    def propose(self) -> list[dict[str, Any]]                      # next (scenario × attack) points
    @abstractmethod
    def observe(self, results: list[dict[str, Any]]) -> None       # feedback (fitness = escalation)
```

---

## 3. Stack interfaces

### 3.1 `Backend` (ABC) — [Now]

`core/interfaces.py`. Adapts any execution environment to one surface.

```python
class Backend(ABC):
    @abstractmethod
    def build(self, spec: ExperimentSpec) -> None            # instantiate AV system + scenario
    @abstractmethod
    def step(self) -> dict[str, Any]                         # advance one tick → per-frame record (§4.2)
    @abstractmethod
    def run(self) -> Iterator[dict[str, Any]]                # drive to completion, yield records
    @abstractmethod
    def close(self) -> None                                  # tear down actors/connections

    def profile(self) -> StackProfile                        # advertise seams+capabilities (default: empty)
    def attach(self, plugin: Any, seam: str = "perception_out") -> None   # default: raises
    def add_perception_hook(self, hook: Any) -> None         # legacy alias: attach(hook, "perception_input")
```

**Contract.** `profile()` must accurately list the seams `attach` supports and the capabilities the
stack provides — the engine resolves plugin bindings against it. `attach(plugin, seam)` routes to a
manual pre-loop (object-level seams like `perception_input`) or to `hooks.attach()` on the right
avstack module (post-hook seams). `step()`/`run()` emit the standardized record (§4.2). The backend
calls `ctx.tick(...)` (§3.6) at the top of each tick.

### 3.2 `StackProfile` & `Capability` — [Now]

`core/capability.py`.

```python
class Capability(str, Enum):
    GT_PERCEPTION; NEURAL_PERCEPTION; RAW_LIDAR; RAW_CAMERA; GRADIENTS
    TRACKER; PLANNER; CONTROLLER; LOCALIZATION; V2X

@dataclass(frozen=True)
class StackProfile:
    seams: frozenset[str] = frozenset()
    capabilities: frozenset[Capability] = frozenset()
    def has_seam(self, seam: str) -> bool
    def provides(self, capabilities: frozenset[Capability]) -> bool
    @classmethod
    def of(cls, seams, capabilities=()) -> "StackProfile"      # from any iterables
```

Reference profiles: `MockBackend` → `of({"perception_input","perception_out"}, {GT_PERCEPTION,
TRACKER})`; `CarlaBackend(neural)` → `of({"perception_out"}, {NEURAL_PERCEPTION, RAW_LIDAR,
TRACKER})`.

### 3.3 `Seam` / `SEAMS` / `Phase` — [Now]

`core/seams.py`. Logical interception points.

```python
class Phase(str, Enum): PRE; POST

@dataclass(frozen=True)
class Seam:
    name: str; phase: Phase; stage: Stage; component: str; arg_index: int = 0

SEAMS: dict[str, Seam]          # raw_lidar, perception_input, perception_out,
                                #   tracking_out, planning_out, control_out
SEAM_ORDER: tuple[str, ...]     # upstream → downstream
def resolve_seam(seam: str | Seam) -> Seam
```

### 3.4 `BindingSpec` & resolution — [Now]

`core/binding.py`. How a plugin's intent maps to a concrete seam on a given stack.

```python
@dataclass(frozen=True)
class BindingSpec:
    seam: str                                   # must be in SEAMS
    payload: str = ""                           # informational: "points"/"objects"/"detections"/…
    requires: frozenset[Capability] = frozenset()
    fidelity: int = 0                           # higher = preferred
    def supported_by(self, profile: StackProfile) -> bool

def resolve(bindings: tuple[BindingSpec, ...], profile: StackProfile) -> BindingSpec   # else IncompatiblePlugin
def seams_downstream_of(seam: str, *, inclusive: bool = True) -> frozenset[str]
class IncompatiblePlugin(RuntimeError): ...
```

**Contract.** `resolve` returns the highest-fidelity binding whose `seam` is exposed and whose
`requires` are all provided; on no match it raises `IncompatiblePlugin` naming the missing seam/
capability. Ties break toward the more upstream seam.

### 3.5 Hook calling convention — [Now] (framework-internal)

`hooks.py`. Plugins never see this — they keep the clean `apply(payload, ego_state=…, ctx=…)`
contract. The adapters translate to avstack's convention:

```
pre-hook  callable(*args, **kwargs) -> (args, kwargs)     # payload at args[seam.arg_index]
post-hook callable(*ret)            -> (value,)           # chain-safe single-tuple
```
```python
def attach(module, plugin, seam: Seam | str, ctx: RunContext) -> HookAdapter
def attach_monitor(module, seam, ctx, extract: Callable | None = None) -> MonitorAdapter
```

**Contract for a plugin's `apply`** (what the adapter guarantees you receive / must return):
- PRE seam: receives the payload at `args[arg_index]`; return the (possibly modified) payload.
- POST seam: receives the module's return value; return the (possibly modified) value.
Attachment order is preserved: attack attached before defense ⇒ defense sees the perturbed payload.

### 3.6 `RunContext` — [Now]

`hooks.py`. Per-run, per-tick state a plugin reads (the ego pose a module's signature doesn't
carry) and a defense writes telemetry into.

```python
@dataclass
class RunContext:
    run_id: str = "run"; frame: int = 0; t: float = 0.0
    ego_state: Any = None; ground_truth: Any = None
    trace: Trace = <auto>
    defense_outcomes: list[Any] = []
    def tick(self, frame, t, ego_state=None, ground_truth=None) -> None    # backend calls each tick
    def record(self, stage: Stage | str, component: str, outputs: Any) -> None
```

---

## 4. Data contracts

### 4.1 Seam payload types — [Now]

The object type flowing through each seam (what `apply` gets and must return):

| Seam | Phase | Payload type |
|---|---|---|
| `raw_lidar` | PRE | LiDAR point cloud (`LidarData` / `[N,4]`) |
| `perception_input` | PRE | `DataContainer[ObjectState]` (GT passthrough) |
| `perception_out` | POST | `DataContainer[BoxDetection]` |
| `tracking_out` | POST | tracks (container of track objects) |
| `planning_out` | POST | planned trajectory |
| `control_out` | POST | control command |

### 4.2 Backend per-frame **record** dict — [Now]

`monitors/trace.py::record_to_ios`. The standardized dict every `Backend.step()`/`run()` yields and
the engine/recorder consume. Required keys:

```python
{
  "frame": int, "t": float,
  "n_input": int,          # objects/points into perception
  "n_detections": int,     # detector outputs
  "n_tracks": int,         # confirmed tracks
  "braking": bool, "throttle": float, "brake": float,
  "hazard_dist": float | None,   # nearest forward hazard (m)
  "ego_speed": float,      # m/s
}
```
A backend may add extra keys (ignored by the core mapping; used by `viz`).

### 4.3 `Trace` & `ComponentIO` — [Now]

`monitors/trace.py`.

```python
@dataclass
class ComponentIO:
    frame: int; stage: str; component: str
    inputs: Any = None; outputs: Any = None; aux: dict[str, Any] = {}

@dataclass
class Trace:
    run_id: str; records: list[ComponentIO] = []
    def add(self, io) -> None
    def by_frame(self, frame) -> list[ComponentIO]
    def index(self) -> dict[tuple[int, str], ComponentIO]
    def series(self, stage: str, key: str) -> list

def build_trace(records: list[dict], run_id: str = "run") -> Trace     # from §4.2 records
```

### 4.4 Escalation DAG, `Stage`, metric output — [Now]

`core/escalation.py`, `metrics/escalation.py`.

```python
class Stage(str, Enum):
    ATTACK_SURFACE; SENSOR; PERCEPTION; LOCALIZATION; TRACKING; FUSION
    PREDICTION; PLANNING; CONTROL; SAFEGUARD; CONSEQUENCE

@dataclass
class EscalationNode: id: str; stage: Stage; component: str; description: str = ""; evidence: dict = {}
@dataclass
class EscalationEdge: src: str; dst: str; condition: str = ""; kind: str = "propagated"

class EscalationDAG:
    def add_node(self, node: EscalationNode) -> None
    def add_edge(self, edge: EscalationEdge) -> None
    @property
    def graph(self) -> nx.DiGraph
    def root_cause(self) -> EscalationNode | None
    def consequence_paths(self) -> list[list[str]]
```

`EscalationMetric.compute(clean, attacked) -> {"metrics": {...}, "dag": EscalationDAG}`, where
`metrics` has the keys:
```python
{ "activated", "activation_frame", "stages_reached", "propagation_depth",
  "reached_consequence", "persistence_frames", "brake_frames_attacked",
  "brake_frames_clean", "min_speed_attacked", "min_speed_clean",
  "final_speed_attacked", "stopped", "speed_suppression", "escalated" }
```
`escalated` (bool) is the top-line verdict.

### 4.5 Recorder interface (duck-typed) — [Now]

`viz/recorder.py`. A backend calls a recorder if one is set via `backend.set_recorder(rec)`:

```python
rec.capture(record: dict, *, points=None, detections=None, rgb=None) -> None   # per tick
rec.finalize(title: str = "run") -> None
compare_runs(clean: list[dict], attacked: list[dict], path: str | Path) -> None
```

---

## 5. Config & spec contracts

### 5.1 `Registry` — [Now]

`config/registry.py`. Registries: `ATTACKS, DEFENSES, MONITORS, METRICS, BACKENDS, SEARCH`.

```python
class Registry:
    def register_module(self, name: str | None = None, module=None)   # decorator or direct
    def get(self, key: str) -> Callable
    def build(self, cfg: dict, **kwargs) -> Any        # cfg = {"type": name, **params}
    def __contains__(self, key: str) -> bool
```

### 5.2 `ExperimentSpec` (+ sub-specs) — [Now]

`core/experiment.py` (pydantic).

```python
class SystemSpec:      pipeline: dict = {}; checkpoints: dict[str,str] = {}; access_level: str = "blackbox"
class ScenarioSpec:    backend: dict;   # {"type": "CarlaBackend", ...} (required)
                       map: str | None; initial_conditions: dict = {}; participants: list = []
                       target_objects: list = []; seeds: list[int] = [0]; repetitions: int = 1
class AttackConfig:    spec: dict;       # {"type": attack_name, **params} (required)
                       threat_model: ThreatModel
class DefenseConfig:   spec: dict        # {"type": defense_name, **params}
class EvaluationConfig: metrics: list[dict] = []; repetitions: int = 1
class ReproducibilityInfo: software_versions: dict = {}; container_image: str | None; config_hash: str | None; notes: str | None

class ExperimentSpec:
    name: str; system: SystemSpec; scenario: ScenarioSpec
    attack: AttackConfig | None; defense: DefenseConfig | None
    evaluation: EvaluationConfig; reproducibility: ReproducibilityInfo
```

### 5.3 `ThreatModel` — [Now]

`core/threat_model.py` (pydantic).

```python
class Knowledge(str, Enum): WHITEBOX; GRAYBOX; BLACKBOX
class AccessLevel(str, Enum): PHYSICAL_ENVIRONMENT; SENSOR; NETWORK_V2X; SOFTWARE; MODEL

class ThreatModel:
    goal: str                         # required
    knowledge: Knowledge = BLACKBOX
    access: list[AccessLevel] = []
    target: str                       # required
    capabilities: list[str] = []
    constraints: list[str] = []
    timing: str | None = None
    success_criteria: str             # required, checkable
```

---

## 6. Engine & result contracts

`core/engine.py`, `reports/report.py`.

```python
class ExperimentRunner:
    def __init__(self, spec: ExperimentSpec) -> None
    def run(self) -> ExperimentResult                 # clean + attacked (+ defended) passes

@dataclass
class ExperimentResult:
    name: str; metrics: dict; dag: Any                # EscalationDAG
    clean_trace: Trace; attacked_trace: Trace
    defended_trace: Trace | None = None
    defended_metrics: dict | None = None
    mitigated: bool | None = None

def render_report(result: ExperimentResult) -> str    # markdown audit
```

---

## 7. Planned interfaces — [Planned]

These contracts are **not in code yet**; they are specified here so implementations stay
consistent. Do not import them.

- **Gradient / white-box interface** (attack optimization). A backend advertising
  `Capability.GRADIENTS` exposes, at a raw-sensor seam, a differentiable objective:
  `grad(objective, payload) -> ndarray` returning `∂objective/∂payload`, so an optimization attack
  runs PGD/EoT **without importing the model**. The stack owns the interface; the attack owns the
  objective + optimizer.
- **Attack generation split.** `AttackBase.generate(system, scenario) -> Artifact` (produce the
  optimized artifact) separate from `apply` (deploy it), so static and optimized attacks share one
  deployment path. Plus a typed **parameter schema** (`parameter_space() -> dict`) exposing the
  attacker-controlled variables + bounds for automated search.
- **Scenario engine.** `ScenarioEngine.select(spec) -> Iterable[ScenarioSpec]` (filter a dataset or
  generate simulation from a formal applicable-scenario description) and `augment(scenario, env) ->
  ScenarioSpec` (add weather/lighting/traffic noise the attacker doesn't control), driven by
  `SearchStrategy` (§2.6) behind a simulation broker.
- **Evaluation attributes.** `AttributeMetric.assess(runs) -> {precision, continuity, robustness,
  intensity, distance_to_impact, ...}` over augmentation sweeps, plus `propose_mitigations(dag,
  attributes) -> list[DefenseConfig]`.
- **AI-harness guardrail.** A static CI check that rejects any change touching protected structure
  (core interfaces, `SEAMS`, registries, the plugin import boundary) or breaking the offline suite /
  ruff — the mechanism that lets AI-authored plugins be trusted only after passing the same gates as
  human code.

---

*Keep this file in sync when a signature changes.* If you change an interface, update the matching
row here and the component's section in `docs/DEVELOPMENT.md` in the same commit.
