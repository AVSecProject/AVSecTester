# Development

## Design principle: no parallel definitions

AVSecTester is a **security layer on avstack**, not a re-implementation of one. The rule that
shapes the codebase: if avstack/avcarla already defines something (the world, the ego, sensors, the
perception/tracking/planning/control modules, the config registries, the pre/post-hook mechanism),
we use it directly — we do not invent a parallel `Environment`/`System`/`Seam`/`Attack` hierarchy
in front of it. When avstack is missing a piece the closed loop needs, we add it **into the avstack
fork** where it belongs, not as glue here.

Concretely:

- The AV system is an **avcarla `CarlaMobileActor`** running an **avstack `ModularDrivingPipeline`**.
- The world/traffic/ticking is an **avcarla `CarlaClient`**; traffic is **avcarla `CarlaNpc`**.
- An attack is an **avstack `HOOKS` hook** attached with **`register_post_hook`**.
- Scenarios, egos, sensors, pipelines, and attacks are all built from config through the
  **`CARLA` / `PIPELINE` / `MODELS` / `HOOKS`** registries.

What lives in AVSecTester is only what has no avstack home: the scenario runner, the attack hooks,
and the impact metric (six files under `avsectester/`).

## Architecture

```
configs/carla_scenario.yaml ──► avsectester.scenario.run_scenario ──► Trace
                                        │
      ┌─────────────────────────────────┼───────────────────────────────┐
      ▼ (CARLA.build)                    ▼ (CARLA.build)                  ▼
  avcarla.CarlaClient           avcarla.CarlaMobileActor          avcarla.CarlaNpc ×N
   world · TM · tick        sensors=[CarlaLidar]                  autopilot traffic
                            pipeline = avstack ModularDrivingPipeline
                              perception ─► tracking ─► planning ─► control
                                   ▲
                                   └── attack: avstack HOOKS hook (register_post_hook)

  run loop:  client.tick() → ego.tick(t, frame)   # pop sensors → pipeline → apply_control
  scoring:   impact(clean_trace, attacked_trace)  # induced braking / unsafe stop?
```

The closed-loop driving pieces were contributed upstream into the forks (see
[`INTERFACE.md`](INTERFACE.md) §"contributed into avstack"): `ModularDrivingPipeline`,
`ForwardCollisionPlanner`, and the `CarlaMobileActor` control loop (`apply_control` + ego-state
feed). Those are real avstack modules, registered in avstack's registries.

## Extending

- **A new attack** — write a callable, `@HOOKS.register_module()` it (see
  `avsectester/attacks/phantom.py`), and reference it in a scenario's `attacks:` list with the
  `stage` to hook. Removal attacks, tracking-stage attacks, and pre-hook (sensor-input) attacks all
  use the same mechanism.
- **A new driving behavior** — add/replace an avstack planning or control module (in the fork) and
  name it in the pipeline config; the scenario is unchanged.
- **A new scenario** — copy `configs/carla_scenario.yaml`; change town/spawn/traffic/sensors/attack.
- **A defense** — a hook that sanitizes a stage's output (e.g. score-gating detections); attach it
  after the attack hook on the same stage and compare the impact with and without it.

## Testing

```bash
python -m pytest tests/ -q        # offline: attack hook + driving pipeline, no CARLA/GPU
avsectester run configs/carla_scenario.yaml --frames 40   # end-to-end: needs a CARLA server + weights
```

`tests/test_phantom.py` checks the attack hook (appends exactly one fabricated detection; registers
in `HOOKS`). `tests/test_pipeline.py` builds `ModularDrivingPipeline` from config and checks that
`ForwardCollisionPlanner` brakes for a forward-corridor track under translation + rotation.

## Roadmap

- More attack hooks: detection removal (false negative), tracking-stage spoofing, sensor-input
  (pre-hook) LiDAR spoofing, camera patches.
- Defense hooks + a mitigation metric (impact with vs without the defense).
- A scenario-search engine: sweep towns / traffic / attack parameters for the worst driving impact.
- Dataset-replay scenarios (avstack-api adapters) alongside the CARLA closed loop.
