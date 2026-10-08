# Development

## Design principle: no parallel definitions

AVSecTester is a **security layer** on avstack + AlpaSim, not a re-implementation of either. The rule
that shapes the codebase: if an upstream project already defines something (the world, the ego,
sensors, the perception/tracking/planning/control modules, the Alpamayo policy, the config
registries, the pre/post-hook mechanism), we use it directly — we do not invent a parallel
`Environment`/`System`/`Seam`/`Attack` hierarchy in front of it. When avstack is missing a piece the
closed loop needs, we add it **into the avstack fork** where it belongs, not as glue here.

Concretely:

- The driving loop uses the **pure data plane** (`Observation`/`Control`/`Trace` in `plane.py`)
  and **two interfaces** (`WorldBackend`/`AVStack` + `run` in `backend.py`). Scenario selection,
  insertion geometry and evaluation expose additional research-facing APIs in this repository.
  They reuse provider state and do not replace simulator actors or driving-model modules.
- The modular AV system is an **avcarla `CarlaMobileActor`** driven by an **avstack
  `ModularDrivingPipeline`**; the end-to-end one is NVIDIA's real **Alpamayo-1.5** model.
- The CARLA world/traffic/ticking is **avcarla `CarlaClient` / `CarlaNpc`**; the NuRec world is
  NVIDIA's **`nre-ga`** renderer serving a reconstructed scene.
- A universal attack is a **`perturb(Observation)`** transform; a modular white-box attack is an
  **avstack `HOOKS` hook** attached with **`register_post_hook`**.

## Architecture

The spine is one loop over a pure data plane; any world backend composes with any AV stack (the 2×2).

Before that loop, `ScenarioSource.scenarios(requirement)` prepares candidates and returns
`ScenarioInstance` objects. `case.make_backend()` reconstructs the selected origin for a paired
experiment. Selection filters stop at this boundary. Runtime insertion uses fixed actor bindings
and updates placement from the current pose. See [SCENARIOS.md](SCENARIOS.md) for the full workflow.

```
      perturb(Observation)  ── the single universal attack seam ──┐
                                                                  ▼
  WorldBackend.reset()/step(Control) ──Observation──►  run(backend, stack, frames) ──►  AVStack(obs)──►Control
       │                                                     │                                │
       ├─ CarlaBackend        (simulators/carla.py)          └──── Trace (TRUE ego state) ────┤
       │    avcarla CarlaClient · CarlaMobileActor · CarlaNpc                                  ├─ ModularAVStack (stacks/modular.py)
       │                                                                                       │    avstack ModularDrivingPipeline
       └─ NuRecBackend        (simulators/nurec.py)                                            │    perception►tracking►planning►control
            EgoPose · KinematicBicycle/TrajectoryFollower · Renderer                          │      ▲ attack: avstack HOOKS hook
              StubRenderer (CI)  |  NuRecRenderer ─gRPC─► nre-ga (NuRec scene)                 └─ AlpamayoAVStack (stacks/alpamayo.py)
                                                                                                    real Alpamayo-1.5-10B → trajectory

  scoring:  impact(clean_trace, attacked_trace)   # induced braking / unsafe stop?
  viz:      metric.plot_impact (metric)  ·  simulators/viz.record_run (per-frame scene → ./tmp/)
```

The CARLA closed-loop driving pieces were contributed upstream into the forks (see
[`INTERFACE.md`](INTERFACE.md) §"What comes from avstack / AlpaSim"): `ModularDrivingPipeline`,
`ForwardCollisionPlanner`, and the `CarlaMobileActor` control loop. The NuRec renderer and the
Alpamayo model come from NVIDIA AlpaSim (`nre-ga`, `alpasim_driver`); `NuRecRenderer` and
`AlpamayoAVStack` are thin adapters behind the `WorldBackend`/`AVStack` interfaces.

## Layout

```
avsectester/
  plane.py            Observation · Control · Trace · FrameRecord   (pure data)
  backend.py          WorldBackend · AVStack · run(...)             (interfaces + loop)
  insertion.py        Insertion · placement/orientation · asset surfaces · pose resolution
  scenario.py         run_scenario · prepare_scenario   (compose a CARLA backend + modular stack)
  simulators/                                            (WorldBackend implementations)
    carla.py          CarlaBackend + camera_view (ImageData) · lidar_bev   (CARLA + its view adapters)
    nurec.py          NuRecBackend · Renderer · StubRenderer · NuRecRenderer · dynamics
    viz.py            record_run · detections_view · filmstrip · save_gif   (generic scene pipeline)
  stacks/                                                (AVStack implementations)
    modular.py        ModularAVStack (avstack ModularDrivingPipeline)
    alpamayo.py       AlpamayoAVStack (wraps alpasim_driver Alpamayo-1.5)
  scenarios/          select/build test cases meeting an attack's preconditions   (see SCENARIOS.md)
  evaluation/         robustness harness + component logging   (see AUGMENTATION.md, COMPONENT_LOGGING.md)
  attacks/
    pipeline/         Internal pipeline attacks: PhantomInjection (avstack HOOKS hook)
    patch/            Physical patch textures, configuration and deployment
    object_insertion/ Traffic signs, pedestrian images and traffic lights
    optim/            Shared PGD/NES algorithms, perturbations, objectives and scorers
  metric.py           impact(clean, attacked) -> Impact + plot_impact  (verdict + its figure)
  cli.py              `avsectester run ...`
```

Attack modules are grouped by function, while rendering and harmonization remain in `simulators`.
For example, import `PhysicalPatch` from `avsectester.attacks.patch.physical_patch` and `RoadsideSign`
from `avsectester.attacks.object_insertion.sign_spoof`. The top-level
`from avsectester.attacks import PhantomInjection` entry point remains lazy; its implementation is
in `avsectester.attacks.pipeline.phantom`.

## Extending

- **A new world backend** — subclass `WorldBackend` (`reset`/`step`); it composes with every existing
  stack through `run(...)` unchanged.
- **A new AV stack** — subclass `AVStack` (`__call__(obs) -> Control`); keep heavy imports lazy (as
  `AlpamayoAVStack` does) so the offline suite still imports in the base env.
- **A universal attack** — write `perturb(Observation) -> Observation` and pass it to `run(..., perturb=)`;
  it works against any stack, black-box policies included.
- **A modular white-box attack** — write a callable, `@HOOKS.register_module()` it (see
  `avsectester/attacks/pipeline/phantom.py`), and reference it in a scenario's `attacks:` list with the
  `stage` to hook. Removal, tracking-stage, and pre-hook (sensor-input) attacks use the same seam.
- **A new modular driving behavior** — add/replace an avstack planning or control module (in the
  fork) and name it in the pipeline config; the scenario is unchanged.
- **A new CARLA scenario** — copy `configs/carla_scenario.yaml`; change town/spawn/traffic/sensors/attack.
- **A selection filter** — implement `Constraint.evaluate(context) -> FilterResult` and compose it
  with existing filters. Native data access and resource ownership are described in
  [SCENARIOS.md](SCENARIOS.md#custom-filters-and-native-access).
- **A dataset or candidate provider** — expose prepared initial states and a factory that recreates
  the selected origin. Preserve actor IDs across the initial window and paired experiment.
- **An inserted asset** — implement `planes()` with asset-local `PlaneSurface` rectangles and use
  `Insertion` for placement. Reuse pose resolution, projection and visibility instead of defining
  a second attachment mechanism in an attack module.
- **A defense** — a sanitizing hook (modular) or an input filter (universal); compare impact with and
  without it.

## Testing

Install the dependencies described in [tests/README.md](../tests/README.md). The `core` and
`avstack` directories run in CI; live tests are selected explicitly.

```bash
python -m pytest -q        # offline: interface + attack hook + pipeline + NuRec + viz, no CARLA/GPU
avsectester run configs/carla_scenario.yaml --frames 40   # CARLA end-to-end: needs a server + weights
python scripts/alpamayo_nurec_demo.py 8 --save-frames     # Alpamayo+NuRec (driver env; needs nre-ga + a scene)
```

`tests/core/test_interface.py` exercises the `run` loop and the `perturb` seam on an in-memory
backend/stack. `tests/avstack/attacks/pipeline/test_phantom.py` checks the attack hook (appends exactly one fabricated
detection; registers in `HOOKS`). `tests/avstack/test_pipeline.py` builds `ModularDrivingPipeline` from
config and checks `ForwardCollisionPlanner` brakes for a forward-corridor track.
`tests/core/test_nurec.py` drives the in-process NuRec backend (dynamics, closed loop, checkpoint/restore)
against `StubRenderer`. `tests/core/test_sim_viz.py` checks the per-simulation scene views. None need
CARLA, a GPU, or the NuRec renderer.

## Roadmap

- More attacks: universal camera perturbations (the AlpaSim adversarial-render seam) against the
  end-to-end stack; modular detection removal, tracking-stage spoofing, sensor-input LiDAR spoofing.
- Defense hooks + a mitigation metric (impact with vs without the defense).
- A scenario-search engine: sweep worlds / traffic / attack parameters for the worst driving impact.
- The remaining 2×2 corners end-to-end (Alpamayo in CARLA; the modular stack on NuRec sensors) and
  hardware-in-the-loop at the same `WorldBackend` seam.
