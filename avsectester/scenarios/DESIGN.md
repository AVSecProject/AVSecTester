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
5. **Dataset adapter + filter — PARTIAL.** `DatasetFilter` implemented + tested over a stub `Dataset`;
   the concrete Alpamayo/NuRec `Dataset` (annotations -> `SceneGT`, clip -> `NuRecBackend`) is the
   remaining integration.
6. **Eval harness — TODO.** `avsectester/evaluation/` — run a `ScenarioSource` × an attack, score with
   `metric.impact`, aggregate to an attack success rate + report.
