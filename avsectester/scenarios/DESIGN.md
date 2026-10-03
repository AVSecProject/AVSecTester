# Scenario layer — evaluating attacks on scenes that satisfy their assumptions

## The problem

An attack is only meaningful on a scene that satisfies its **assumptions**. A physical-patch attack
that removes a *vehicle* detection needs a target vehicle actually visible in the camera, close enough
and unoccluded enough to carry a patch. A phantom-injection attack needs the ego to be *driving toward
a clear road* so an injected obstacle can change the plan. Different attacks → different preconditions.

So before we can score an attack we must first **obtain test cases that satisfy its preconditions**:

- **Real data (Alpamayo / NuRec traces):** only a *subset* of frames/clips satisfy the requirement —
  we must **filter** the dataset to those.
- **Simulation (CARLA):** we can **construct** a scene that satisfies the requirement (spawn the target
  vehicle at a qualifying distance / angle).

The two look different (select vs. build) but must be driven by **one shared specification** of the
requirement, defined **once per attack**, so a new attack only declares *what it needs* and both
providers just work.

## The key idea: a requirement is a predicate over ground-truth scene state

Both a CARLA sim and an annotated dataset can expose the same **ground-truth scene state**
(`SceneGT`: ego state, camera calibrations, and the 3-D/2-D objects with categories, poses, distances,
image boxes, visibility). We define an attack's requirement as a **composable predicate over
`SceneGT`** that also **selects the target** the attack acts on. Then:

- **filter** = evaluate the predicate on each dataset sample's `SceneGT`, keep the ones that hold;
- **build** = construct a CARLA scene, then **validate** it by evaluating the *same* predicate on the
  built scene's `SceneGT` (and use the constraints to *parameterize* the construction).

One requirement, two `ScenarioSource`s. This is the whole design.

```
                         ScenarioRequirement (per attack)
                     target: TargetSpec  +  [Constraint, ...]
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                            ▼
      DatasetFilter(dataset)                     CarlaScenarioBuilder()
   for each clip/frame:                       parameterize a scene from the
     SceneGT from annotations                   constraints, build it, then
     req.match(scene)? ──► yield                 VALIDATE req.match(scene) ──► yield
              │                                            │
              └───────────────► ScenarioInstance ◄─────────┘
             make_backend() -> WorldBackend  +  target (ScenarioMatch)  +  provenance
                                    │
                                    ▼
                          evaluation harness runs the
                          attack on the backend + scores
                          (avsectester.metric.impact)
```

## Components (this folder)

- **`scene.py`** — `SceneGT`, `ObjectGT`, `EgoState`, `CameraCalib`: the **canonical ground-truth**
  scene representation the predicates read. A *provider adapter* maps its native GT (CARLA world state,
  or a dataset's per-frame labels) into this schema. Backend-agnostic, no sim/dataset imports.

- **`requirement.py`** — the **DSL**: a `Constraint` base + concrete constraints
  (`InView`, `ImageAreaFrac`, `DistanceRange`, `ViewpointRear`, `MinVisibility`, `EgoMoving`,
  `ClearLaneAhead`, …), `TargetSpec` (how to pick the target), and `ScenarioRequirement`
  (target + constraints) whose `match(scene) -> ScenarioMatch | None` is the predicate. Pure logic,
  fully testable offline.

- **`source.py`** — `ScenarioInstance` (a runnable case: a `WorldBackend` factory + the target +
  provenance), the `ScenarioSource` interface, and the two providers `DatasetFilter` /
  `CarlaScenarioBuilder` that consume any `ScenarioRequirement`.

- **`requirements.py`** — the **per-attack requirement definitions** (e.g. `physical_patch_hide_vehicle`).
  A new attack adds one entry here (or the attack module exports its own `ScenarioRequirement`); the
  providers and the eval harness are unchanged.

## Why a Python-object DSL (not a text/YAML DSL) — for now

Constraints are composable objects (`ScenarioRequirement(target=…, constraints=[DistanceRange(4, 25),
ImageAreaFrac(0.02, 0.5), ViewpointRear(35), MinVisibility(0.7)])`). This is an *embedded* DSL: type-
checked, trivially extensible (add a `Constraint` subclass), and directly executable. A parsed
text/YAML surface can be added later as a thin front-end that builds these objects — but the object
model is the source of truth. Keeping the object model first avoids inventing grammar before we know
the constraint vocabulary.

## Example — physical patch that hides a vehicle

```python
ScenarioRequirement(
    name="physical_patch_hide_vehicle",
    target=TargetSpec(category="vehicle", camera="front", select="nearest_ahead"),
    constraints=[
        InView("front"),            # the target projects into the front camera
        DistanceRange(4.0, 25.0),   # close enough to place + read a patch, not on top of us
        ImageAreaFrac(0.02, 0.5),   # big enough to matter, not the whole frame (false positive)
        ViewpointRear(max_deg=35),  # we see roughly its rear face (where the patch goes)
        MinVisibility(0.7),         # mostly unoccluded, so the patch is not hidden
    ],
)
```

- **DatasetFilter** keeps only Alpamayo/NuRec frames where a vehicle meets all five → those clips +
  start frames become `ScenarioInstance`s (a `NuRecBackend` seeded at that clip/frame).
- **CarlaScenarioBuilder** spawns a lead vehicle ahead of the ego, sampling distance in `[4, 25]` and a
  small lateral offset so `ViewpointRear`/`ImageAreaFrac` hold, then validates with `req.match`.

## Ground-truth sourcing (the hard part per provider)

- **CARLA:** GT is free from the sim — object 3-D boxes/poses from live actors, camera calib from the
  sensor; project boxes to 2-D with the existing `simulators.patch_insertion` projection helpers to
  fill `ObjectGT.box2d`. So the builder both *constructs* and *self-validates*.
- **NuRec / Alpamayo:** the **render API exposes no actor boxes** (only the ego trajectory), so
  `SceneGT` must come from the **dataset's own annotations** (the Alpamayo clip labels: per-frame 3-D
  boxes + calibration), read by a `Dataset` adapter — *separate from* the nre-ga renderer that produces
  pixels at run time. `DatasetFilter` reads annotations to select; the selected clip+frame then drives a
  `NuRecRenderer` at run time. This split (annotations for selection, renderer for pixels) is the main
  integration to build.

## How it plugs into the evaluation layer (context, planned next)

```python
source   = CarlaScenarioBuilder()            # or DatasetFilter(alpamayo)
req      = REQUIREMENTS["physical_patch_hide_vehicle"]
results  = []
for case in source.scenarios(req, limit=50):
    backend = case.make_backend()
    clean   = run(backend, stack, frames)                    # no attack
    attacked= run(backend, stack, frames, perturb=attack_for(case.target))
    results.append(impact(clean, attacked))                  # avsectester.metric
report = aggregate(results)   # attack success rate over the qualifying scenarios
```

The scenario layer's job ends at yielding qualifying, runnable `ScenarioInstance`s + the target; the
eval harness (separate module, next step) runs clean-vs-attacked and aggregates `impact` verdicts.

## Natural-language requirements (LLM-interpreted)

`ScenarioRequirement` carries a free-text `description`; `nl.interpret(description, llm)` turns it into
the formal target + constraints. The LLM is **injected** (`llm: str -> str`), so there is no API
dependency: `nl.build_prompt` grounds the model in the exact constraint vocabulary
(`serialize.constraint_vocabulary`, auto-generated from the dataclasses), `nl.parse_response` extracts
its JSON, and `serialize.requirement_from_dict` deserializes it through the constraint registry — an
unknown kind or bad field raises rather than silently mis-specifying the scenario. So a human writes
"a car directly ahead, close and unoccluded, rear facing us" and gets a runnable predicate. The
object model (`serialize`) is the source of truth; NL is a front-end onto it.

## Phased implementation — status

1. **Interface + DSL — DONE.** `scene.py`, `requirement.py` (fully implemented + tested), `source.py`,
   `requirements.py`. Offline, ruff-clean, unit-tested.
2. **CARLA builder — DONE (validated live).** `CarlaScenarioBuilder` enumerates lead placements
   *analytically* (`carla_gt.predict_scene_gt` — no CARLA; needs a physically-spawnable `min_gap`),
   keeps those where `req.match` holds, and builds the real `CarlaBackend` lazily in `make_backend`.
   Confirmed end-to-end on a live CARLA server: a built scene's live GT satisfies all five constraints.
3. **CARLA GT adapter — DONE (validated live).** `carla_gt.carla_scene_gt(backend)` (live actors →
   `ObjectGT` in the ego frame, 3-D boxes projected via `simulators.patch_insertion`). Note: derive the
   box centre from the world vertices — never `carla.Transform.transform(bb.location)`, which mutates
   `bb.location` in place and corrupts the next projection.
4. **Serialization + NL — DONE.** `serialize.py` (dict <-> requirement) + `nl.py` (LLM interpreter),
   tested with a stub LLM.
5. **Dataset adapter + filter — DONE (real data, both datasets labeled).** A `Dataset` exposes a GT
   `SceneGT` per frame; both real sources ship 3-D labels, read directly:
   - **nuScenes:** `datasets/nuscenes.py:NuScenesDataset` reads GT boxes via the devkit -> `SceneGT`,
     paired with `RecordedFrameBackend` (replay the recorded frame). Validated on **real nuScenes
     v1.0-trainval GT** (permission fixed): 400 keyframes / 861 GT vehicles -> **80 qualify** for
     `physical_patch_hide_vehicle`, with real distance + orientation + visibility (so `ViewpointRear`/
     `MinVisibility` are meaningful). Target in `tmp/compare/nuscenes_gt_filter.png` (truck 25 m ahead).
   - **nuRec:** `datasets/nurec.py:NuRecDataset` reads each `.usdz` (a ZIP) — the render RPC returns only
     pixels, but the artifact itself ships GT: `sequence_tracks.json` (actor cuboid tracks: id,
     `label_class`, per-ts pose `[x,y,z,qx,qy,qz,qw]`, dims) + `rig_trajectories.json` (per-camera-frame
     rig poses `cameras_frame_T_rig_worlds` aligned 1:1 with the rendered `.mp4`, and the f-theta camera
     calibration). At a chosen camera frame each present actor is expressed in the **rig frame** (= our ego
     frame; AlpaSim CONTRIBUTING.md) via `inv(T_rig_world) @ actor_pose` — the tracks and the rig share one
     frame, so this inverse is the whole transform, **matching AlpaSim** (it uses the tracks directly with
     `pose_local_to_rig`; `world_to_nre` is renderer-only and is NOT applied to the tracks). This gives
     exact center/yaw/extent/distance; `box2d` is projected with the scene's **real f-theta model**
     (`FThetaCamera`, `T_sensor_rig` extrinsic + `angle_to_pixeldist` poly); `visibility=1.0` (tracks carry
     no occlusion fraction). nuRec is a **peer of `NuScenesDataset`** — no renderer/GPU needed to filter.
     **Validated on real `PhysicalAI-Autonomous-Vehicles-NuRec` 26.01 `.usdz`** by overlaying the projected
     boxes on the rendered `.mp4` (they land on the real vehicles across early and late frames); ~7/21 and
     8/21 sampled frames of two clips qualify for `physical_patch_hide_vehicle`. Tests read the real
     artifacts and **skip** if absent (no synthetic stand-in).

   The earlier detector-labeling fallback (`DetectorLabeler`/`ImageFolder`/`FrameSource`) is **removed** —
   both datasets now have real 3-D labels, so deriving `SceneGT` from a 2-D detector had no remaining use.
6. **Eval harness — DONE.** `avsectester/evaluation/robustness.py:evaluate_robustness` runs a
   `ScenarioSource` × an attack × a **corruption suite** (`simulators.augment`): for each scenario and
   each condition (clean baseline + each `AugmentationPipeline`) it drives clean-vs-attacked under the
   *same* corruption and scores with `metric.impact`. `RobustnessReport` aggregates
   `impact.attack_succeeded` into an **attack success rate (ASR)** per condition and a **resilience**
   (`ASR(corruption)/ASR(clean)`, a Robo3D-mRR-style retention rate) + `mean_resilience`. The attack is
   injected (`attack_for(match, backend) -> perturb`), so the harness is attack-agnostic. Tested with a
   closed-loop stub (`tests/test_robustness.py`).
