# First-party functional tests

Run from the repository root in a Python 3.10 environment with the project's development
dependencies and the avstack packages installed:

```bash
python -m pytest tests/ -q
```

The default suite does not connect to CARLA or load neural checkpoints. It does import the
real avstack/avcarla packages. Their import dependencies must therefore be installed, including
the CARLA Python client, pygame, and ipywidgets (used by avstack-api visualization imports).
See [the setup guide](../docs/SETUP.md) for the dependency layout. Matplotlib is needed for the
plot tests; those tests skip if it is absent.

## Coverage

| File | Behavior exercised |
|---|---|
| `test_interface.py` | The `run(backend, stack, frames, perturb)` loop and the universal `perturb(Observation)` attack seam on an in-memory backend/stack (no avstack/CARLA) |
| `test_nurec.py` | The in-process `NuRecBackend`: `KinematicBicycle`/`TrajectoryFollower` dynamics, the closed loop on `StubRenderer`, and `checkpoint`/`restore` round-trip |
| `test_sim_viz.py` | Per-simulation scene views: `camera_view` (frame vs non-image), `lidar_bev` defensiveness, and `record_run` saving a frame per step |
| `test_phantom.py` | Hook registration and execution, detection preservation, phantom geometry, empty outputs, moving sensor frames, source-reference metadata |
| `test_pipeline.py` | Pipeline construction, forward-corridor planning, clean/attacked sequences through real perception, tracking, planning, and PID control |
| `test_patch_optim.py` | PGD/NES adversarial optimization: objectives, placement homography, and that PGD/NES reduce the objective on a synthetic scorer (no CARLA/mmdet) |
| `test_physical_patch.py` | Physical-patch texture helpers (checkerboard/image RGBA) and config parsing (no CARLA) |
| `test_metric.py` | Driving verdicts, unchanged/natural stops, threshold boundaries, custom thresholds, reported measurements, empty runs |
| `test_scenario.py` | Runner frame records, actor initialization, hook ordering, NPC configuration, delayed sensor data, cleanup on success and failure, GPU override |
| `test_reproducibility.py` | `prepare_scenario`: seeded scene selection, resolved settings, client release on failure, and the clean-spawn-retry flag |
| `test_cli.py` | Configuration and overrides, clean/attacked passes, verdict exit codes, plotting |
| `test_carla_integration.py` | Explicitly enabled neural perception and real closed-loop driving |

The propagation tests supply a moving ego's state and synthetic detection sequences to the real
avstack modules. They establish that a phantom becomes a confirmed track and changes the plan
and control command; they do not simulate vehicle dynamics. The runner tests replace only the
simulator-facing objects and registry builders, exercising the actual `run_scenario` function.
The CLI tests replace the scenario runner while exercising the real CLI and impact metric.

The interface / NuRec / viz / patch-optimization tests (`test_interface`, `test_nurec`,
`test_sim_viz`, `test_patch_optim`, `test_physical_patch`) import without avstack/avcarla/CARLA;
their torch/matplotlib/PIL-dependent cases skip when those packages are absent.

## Live CARLA test

The integration test is marked `carla` and skipped unless `--run-carla` is supplied. Opting in
uses a live server, reloads its world, spawns actors, and runs clean/clean/attacked drives.
Use a dedicated CARLA 0.9.15 server with the desired map already loaded, the full GPU perception
environment, and downloaded model checkpoints. Missing prerequisites fail an opted-in run;
they do not silently skip it.

```bash
python -m pytest tests/test_carla_integration.py --run-carla -q
# Use another scenario or a different perception GPU:
python -m pytest tests/test_carla_integration.py --run-carla \
  --carla-config configs/carla_scenario.yaml --carla-gpu 1 -q
```

The spawn-only test puts an NPC at the ego's requested position to force relocation, then checks
that a reset run accepts the recorded poses without applying the configured offset twice. It runs
no driving steps and uses passthrough perception, so it does not need neural inference.

The driving test compares initial vehicle poses and velocities across all three runs, and requires repeated
clean speeds to agree within 0.05 m/s with equal braking-frame counts. It also checks the attack's
driving impact. It writes the resolved configuration, all three traces, initial states and impact
measurements to `carla-result.json` under pytest's temporary
test directory before checking the verdict. Simulation physics and model inference are only
covered when this test is actually executed.
