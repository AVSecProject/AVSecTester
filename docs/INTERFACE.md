# AVSecTester — Interface Reference

The precise contract reference. Six small types in `avsectester.core` carry the whole
framework; everything else (attacks, defenses, environments, metrics) implements them.
`docs/DEVELOPMENT.md` explains *why*; this document is the *what*.

Signatures are given as they exist in code. **[Now]** = implemented; **[Planned]** = intended,
not in code yet.

**Contents**
1. Interface map
2. `Frame`, `Seam`, `Context`
3. `Environment` (dataset ⇄ simulation)
4. `System` (AV pipeline under test) + `Outcome`
5. `Attack` / `Defense`
6. `Metric` + `Trace`
7. Runner (`run`, `run_experiment`, `Result`)
8. Config & registries · `ThreatModel`
9. Planned interfaces

---

## 1. Interface map

| Type | Module | Role |
|---|---|---|
| `Frame` | `core/frame.py` | one time-step of AV data (dataset & sim share it) |
| `Seam` | `core/seam.py` | static injection-point names (enum) |
| `Context` | `core/context.py` | `(frame, seam)` handed to `apply` |
| `Environment` | `core/environment.py` | frame source: `reset`/`step`; the dataset⇄sim bridge |
| `System` | `core/system.py` | AV pipeline under test; fires attacks at seams |
| `Outcome` | `core/system.py` | `(control, record)` returned by `System.process` |
| `Attack` / `Defense` | `core/attack.py` | offline `prepare` + runtime `apply` / runtime `apply` |
| `Metric` | `core/metric.py` | `compute(clean, attacked) -> dict` |
| `Trace` | `core/trace.py` | per-frame records of one run |
| `run` / `run_experiment` / `Result` | `core/runner.py` | the runtime loop + paired passes |

**Golden rule.** Attacks/defenses import only `core` (`Attack`/`Defense`/`Context`/`Seam`/`Frame`)
+ `config` registries — never an environment, an avstack model, or the runner.

---

## 2. `Frame`, `Seam`, `Context` — [Now]

```python
@dataclass
class Frame:
    index: int = 0
    timestamp: float = 0.0
    sensors: dict[str, Any] = {}        # {"lidar": LidarData, "camera": ImageData, ...}
    ego: Any = None                     # ego ObjectState (pose + velocity)
    calibration: dict[str, Any] = {}    # per-sensor calibration
    ground_truth: Any = None            # GT objects (metrics); may be None on real data
    meta: dict[str, Any] = {}           # weather, scenario tags, route, ...

class Seam(str, Enum):                  # static injection points
    RAW_LIDAR; RAW_CAMERA; RAW_GPS
    PERCEPTION_INPUT; PERCEPTION_OUT; LOCALIZATION_OUT; TRACKING_OUT; PLANNING_OUT; CONTROL_OUT

@dataclass
class Context:
    frame: Frame                        # current frame (ego, sensors, ground_truth, meta)
    seam: Seam                          # which seam is firing now
```

---

## 3. `Environment` — [Now]

The common bridge over dataset replay and simulation.

```python
class Environment(ABC):
    @abstractmethod
    def reset(self) -> Frame: ...
    @abstractmethod
    def step(self, control: Any = None) -> tuple[Frame, bool]:   # (next_frame, done)
        ...
    def close(self) -> None: ...
```

**Contract.** `reset` returns the first frame; `step` advances one tick and returns
`(next_frame, done)`. **This is the only place dataset and simulation differ:** a dataset
ignores `control` (returns the next recorded frame); a simulator applies it and advances the
world. Reference: `MockEnv` (kinematic world), `CarlaEnv` (CARLA server) — both in
`avsectester.envs`, registered in `ENVIRONMENTS`.

---

## 4. `System` (+ `Outcome`) — [Now]

The AV pipeline under test. It processes a frame into control, and **fires attached plugins at
its seams by calling their `apply` directly** — so the plugin's `apply` *is* the hook.

```python
@dataclass
class Outcome:
    control: Any = None                 # fed back to Environment.step
    record: dict[str, Any] = {}         # per-frame metrics record

class System(ABC):
    seams: tuple[Seam, ...] = ()        # the seams this system exposes

    def attach(self, plugin, seam: Seam | str | None = None) -> None
        # attach at `seam`, or (None) at each seam in plugin.seams; raises if a seam isn't exposed
    def fire(self, seam: Seam, payload: Any, frame: Frame) -> Any
        # run every plugin attached at `seam` over `payload` (sets ctx.seam) and return it
    @abstractmethod
    def process(self, frame: Frame) -> Outcome: ...
    def close(self) -> None: ...
```

**Contract.** `process` runs the pipeline on `frame`, calling `self.fire(seam, payload, frame)`
at each interception point, and returns an `Outcome`. `record` SHOULD carry the keys metrics
read (see §6). Reference: `MockSystem` (passthrough perception + tracker + brake reflex),
`CarlaSystem` — registered in `SYSTEMS`.

---

## 5. `Attack` / `Defense` — [Now]

```python
class Attack(ABC):
    seams: tuple[Seam, ...] = ()        # where it injects
    threat_model: Any = None            # optional adversary spec (ThreatModel)
    artifact: Any = None                # prepared artifact, adopted via load()

    def prepare(self, data: Iterable[Frame]) -> Any:   # OFFLINE: data → artifact (default: None)
        return None
    def load(self, artifact: Any) -> None:             # RUNTIME: adopt a prepared artifact
        self.artifact = artifact
    def reset(self) -> None: ...                        # clear per-run state (optional)
    @abstractmethod
    def apply(self, payload: Any, ctx: Context) -> Any # RUNTIME: inject at ctx.seam

class Defense(ABC):                     # the runtime half of the attack shape (no artifact)
    seams: tuple[Seam, ...] = ()
    def reset(self) -> None: ...
    @abstractmethod
    def apply(self, payload: Any, ctx: Context) -> Any
```

**Contract.** Two parts:
- **Offline** — `prepare(frames) -> artifact` consumes a stream of `Frame`\s (a dataset) and
  returns a *serializable, attack-specific* artifact (a patch, a point set, tuned params);
  `load(artifact)` adopts it. Attacks needing no optimization skip `prepare`.
- **Runtime** — `apply(payload, ctx)` returns a payload of the same type it received. Read the
  ego from `ctx.frame.ego`; if attached at several seams, branch on `ctx.seam`.

**Minimal attack:**
```python
@ATTACKS.register_module()
class MyAttack(Attack):
    seams = (Seam.PERCEPTION_OUT,)
    def __init__(self, target_xyz=(12,0,-1.5)):
        self.target_xyz = target_xyz
        self.threat_model = ThreatModel(goal="…", target="…", success_criteria="…")
    def apply(self, payload, ctx):
        ...            # perturb `payload` using self.target_xyz / self.artifact and return it
        return payload
```

---

## 6. `Metric` (+ `Trace`) — [Now]

```python
@dataclass
class Trace:
    records: list[dict] = []; run_id: str = "run"
    def add(self, record: dict) -> None
    def series(self, key: str) -> list          # values of record[key] over the run
    def count(self, key: str) -> int            # records where record[key] is truthy
    def __len__(self) -> int

class Metric(ABC):
    @abstractmethod
    def compute(self, clean: Trace, attacked: Trace) -> dict[str, Any]: ...
```

**Standard record keys** a `System` emits (what metrics read): `frame, t, n_input,
n_detections, n_tracks, ego_speed, throttle, brake, steer, hazard_dist, braking`.
Reference: `ImpactMetric` (`metrics/impact.py`) → `{brake_frames_*, min_speed_*,
final_speed_attacked, stopped, speed_suppression, impacted}`. (Escalation/attribution analysis
is intentionally **not** here yet — to be added later.)

---

## 7. Runner — [Now]

```python
def run(env: Environment, system: System, run_id="run") -> Trace
    # frame = env.reset(); loop: out = system.process(frame); trace.add(out.record);
    #                            frame, done = env.step(out.control)

@dataclass
class Result:
    clean: Trace; attacked: Trace; metrics: dict
    defended: Trace | None = None; defended_metrics: dict | None = None

def run_experiment(make_env, make_system, metric, attack=None, defense=None) -> Result
    # fresh env+system per pass (factories); runs clean, attacked, and (if defense) defended
```

---

## 8. Config & registries · `ThreatModel` — [Now]

Registries (`config/registry.py`), built from `{"type": name, **params}`:
`ENVIRONMENTS, SYSTEMS, ATTACKS, DEFENSES, METRICS`. A YAML config for `avsectester run`:

```yaml
name: mock_object_spoof_defended
environment: {type: MockEnv, frames: 140}
system: {type: MockSystem, target_speed: 6.0}
attack:  {type: ObjectSpoofingAttack, target_xyz: [6.0, 0.0, 0.0], score: 0.3}
defense: {type: ScoreGateDefense, threshold: 0.5}
metric:  {type: ImpactMetric}
```

`ThreatModel` (`core/threat_model.py`, pydantic) — the optional adversary spec on an attack:
`goal, knowledge (white/gray/black-box), access[], target, capabilities[], constraints[],
timing, success_criteria`.

---

## 9. Planned interfaces — [Planned]

Not in code; specified so future work stays consistent.

- **Escalation / attribution analysis.** A richer metric/graph over per-seam diagnosis signals
  (the DAG was removed; its replacement will be designed by the project owner).
- **DatasetEnv.** An `Environment` that replays nuCarla/KITTI frames (ignores `control`) — the
  offline source for `Attack.prepare`.
- **Optimization attacks.** `Attack.prepare` producing an optimized artifact against a victim
  model via a differentiable/query interface exposed by the system at a raw-sensor seam.
- **New seams wired.** `raw_camera` / `localization_out` / `tracking_out` on the systems.

---

*Keep this file in sync when a signature changes — update the matching section here and in
`docs/DEVELOPMENT.md` in the same commit.*
