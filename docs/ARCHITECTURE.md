# AVSecTester Architecture

AVSecTester is a **security layer built on top of [avstack](https://github.com/avstack-lab)**.
avstack supplies the AV stack, geometry, sensors, CARLA bridge, dataset adapters, and an
mmengine-style registry/config system; AVSecTester adds everything security-specific.

```
                          ┌─────────────────────────────────────────────┐
                          │                AVSecTester                   │
                          │                                              │
  ExperimentSpec (YAML) ──►  core: spec · threat model · escalation DAG  │
                          │        · plugin interfaces · registries      │
                          │                                              │
                          │  attacks ─┐   defenses ─┐   monitors ─┐      │
                          │           │             │            │      │
                          │           ▼ pre/post hooks (HOOKS)   ▼      │
                          │  ┌───────────────────────────────────────┐  │
   backends adapt ────────►  │        avstack pipeline (ego AV)      │  │
                          │  │  perception→tracking→…→planning→ctrl  │  │
                          │  └───────────────────────────────────────┘  │
                          │        │ traces                              │
                          │        ▼                                     │
                          │  metrics · search · reports · agent          │
                          └──────────────┬───────────────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
      lib-avstack-carla           avstack-core                avstack-api
       (avcarla, CARLA)        (avstack modules,           (avapi, KITTI/
        closed-loop            geometry, mmdet3d)           nuScenes/CARLA)
```

## Key integration decisions

1. **Build on avstack** (confirmed). avstack already bridges CARLA 0.9.13 + mmdetection3d +
   datasets; its `security-sandbox` is stale, so the security layer is greenfield.
2. **Non-invasive interception via hooks.** avstack modules run `_apply_pre_hooks` /
   `_apply_post_hooks` (the `@apply_hooks` decorator, `HOOKS` registry). AVSecTester
   attacks/defenses/monitors are hook-shaped callables (`hook(*io) -> io`) attached to
   pipeline modules — **we do not fork avstack internals**. Metrics already work this way
   in avstack (`MetricsHook`).
3. **Shared registry/config.** We reuse avstack's `Registry` so plugins build-from-config
   the same way avstack modules do, with a local fallback shim when avstack isn't installed.
4. **One spec, many backends.** `ExperimentSpec` runs on `CarlaBackend` (closed-loop) or
   `DatasetBackend` (offline) unchanged.

## Package map

| Package | Role | PLAN phase |
|---|---|---|
| `core` | experiment spec, threat model, escalation DAG, interfaces, registries | 1 |
| `backends` | adapters over avcarla / avapi / mmdet3d | 2 |
| `attacks` | attack plugins + engine (hook-injected) | 3 |
| `monitors` | instrumentation, traces, clean-vs-attacked diff → escalation DAG | 4 |
| `defenses` | defense/mitigation plugins | 5 |
| `metrics` | activation/targeting/persistence/safety/detectability/mitigability/practicality | 6 |
| `search` | evolutionary fuzzing / scenario search | 7 |
| `reports` | root-cause attribution + audit reports | 8 |
| `agent` | agent-assisted integration workflows + validation gates | 9 |
| `knowledge` | reusable vulnerability-path store | 10 |

## Coordinate frames (correctness-critical)

CARLA is **left-handed** (X-fwd, Y-right, Z-up); mmdet3d/KITTI/nuScenes are **right-handed**.
All frame conversions go through avstack `geometry/refchoc`, with explicit round-trip tests —
silent frame bugs corrupt attack results without failing loudly.

See `dev/PLAN.md` (local, gitignored) for the full task checklist.
