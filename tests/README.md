# Tests and CI

Offline tests use **Python 3.11**. They are selected by directory, before pytest imports the
test modules, so the core suite does not need avstack installed.

| Directory | What runs | Requirements |
|---|---|---|
| `core/` | Run loop, metrics, CPU attack optimization, images, scenario selection, NuRec dynamics, Alpamayo adapter | CPU test dependencies below |
| `avstack/` | Phantom geometry, real tracking/planning/control, modular adapter, scenario orchestration, CLI | Core dependencies plus the three avstack packages and CARLA Python client |
| `live/` | Real CARLA driving and real NuRec/nuScenes datasets | Explicit selection and the corresponding server, models or datasets |

`python -m pytest` selects `core/` and `avstack/`. A core-only installation should explicitly run
`python -m pytest tests/core`. Missing offline test dependencies **fail**, rather than silently skip
the affected functionality. Live tests retain their prerequisite checks and CARLA opt-in.

## Core tests

From the repository root, in a Python 3.11 environment:

```bash
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]' -c constraints.txt
NO_ALBUMENTATIONS_UPDATE=1 python -m pytest tests/core -q
```

The `test` extra includes pytest, coverage, Pillow, matplotlib, OpenCV and Albumentations.
CPU torch is installed separately to avoid downloading CUDA dependencies. In an existing full
environment, keep its compatible torch installation and install only the test extra.

## Offline avstack tests

These tests use the real avstack implementation. They replace the CARLA server-facing objects
and use synthetic detections instead of neural inference; no server, CUDA or checkpoints are needed.

After installing the core test dependencies:

```bash
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --depth 1 -- \
  third_party/avstack-core third_party/avstack-api third_party/lib-avstack-carla
python -m pip install -e third_party/avstack-core \
  carla==0.9.16 pygame ipywidgets 'opencv-python<4.12' -c constraints.txt
python -m pip install --no-deps -e third_party/avstack-api -e third_party/lib-avstack-carla
NO_ALBUMENTATIONS_UPDATE=1 python -m pytest tests/avstack -q
# Both offline groups:
NO_ALBUMENTATIONS_UPDATE=1 python -m pytest -q
```

The adapters are installed with `--no-deps` because their full dataset/Open3D dependencies and
legacy sibling-path declarations are unnecessary for this suite. The commands above install
the imports exercised by these tests. This is an offline test environment; use
[SETUP.md](../docs/SETUP.md) for neural perception or real dataset workflows.

## Functional coverage

| Tests | Behavior exercised |
|---|---|
| `core/test_interface.py` | Per-step controls affect subsequent true state; replacement attack observations reach the stack without falsifying the recorded state |
| `core/test_metric.py` | Induced stops, suppressed safe stops, duration/speed thresholds, late stops, inconclusive baselines |
| `core/test_alpamayo.py` | Camera history padding/order, ego history, model vs simulation clocks, candidate selection, plan chaining, experiment reset |
| `core/test_nurec.py` | Dynamics, stub-rendered closed loop, reset, checkpoint replay, trajectory interpolation and rig-to-world conversion |
| `core/test_patch_optim.py` | PGD/NES optimization direction, patch footprint, valid pixels and L-infinity budget |
| `core/test_patch_insertion.py`, `core/test_physical_patch.py` | Projection, compositing, texture construction and patch configuration |
| `core/test_augment.py` | Image corruptions, seeded paired experiments, sensor write-back, composition, real Albumentations operators |
| `core/test_scenarios.py`, `core/test_recorded_backend.py` | Scene constraints/filtering and recorded-image playback without full datasets |
| `core/test_sim_viz.py` | Camera/LiDAR views, projected labels/boxes and saved frames |
| `avstack/test_phantom.py` | Attack registration, detection preservation, moving/empty source frames and phantom geometry |
| `avstack/test_pipeline.py` | Real tracking/planning/PID propagation, hook output replacement and the `ModularAVStack` adapter |
| `avstack/test_scenario.py`, `avstack/test_reproducibility.py` | Actor setup/cleanup, hook ordering, scene resolution and actual spawn replay |
| `avstack/test_cli.py` | YAML/overrides, clean/attacked calls, verdict exit codes and plotting |

Alpamayo tests use small external-schema stand-ins and a model with prescribed predictions.
They validate our adapter, not AlpaSim API compatibility, checkpoint loading or real inference.
NuRec core tests use `StubRenderer`, not the rendering service. The avstack propagation tests
exercise real tracking/planning/control against constructed observations, not CARLA physics.

## GitHub Actions

[ci.yml](../.github/workflows/ci.yml) runs on pull requests, pushes to `main`, and manual dispatch.
It uses Ubuntu 22.04 / Python 3.11 with three parallel checks:

- `core-tests`: core suite, then wheel build and CLI startup from a fresh installation outside
  the checkout, without the source tree on `PYTHONPATH`.
- `avstack-tests`: avstack suite at the main repository's recorded submodule revisions.
- `lint`: Ruff correctness checks on `avsectester/` and `tests/`; no third-party lint or formatting gate.

The workflow caches downloaded dependencies and cancels superseded runs on the same PR/branch.
JUnit and coverage reports remain available as artifacts for seven days, including after a test
failure. Coverage is informational, with no percentage gate. Test dependencies come from the
`test` extra and workflow install steps, not the historical CUDA `requirements.lock`.

To prevent merging a failing PR, a repository administrator can make `core-tests`, `avstack-tests`
and `lint` required status checks in the GitHub branch rules. Workflow files alone do not enable
that policy. The workflow does not publish packages or images.

## Live tests

Select these explicitly; they are outside the default offline test paths.

For CARLA, use a dedicated **0.9.16** server with the full perception environment and downloaded
model checkpoints. The test reloads the world and spawns actors:

```bash
python -m pytest tests/live/test_carla_integration.py --run-carla -q
python -m pytest tests/live/test_carla_integration.py --run-carla \
  --carla-config configs/carla_scenario.yaml --carla-gpu 1 -q
```

The spawn-only case needs no neural inference. The driving case compares clean/clean/attacked
runs, including initial actor state and driving impact. It writes `carla-result.json` under
pytest's temporary directory. Opting into CARLA makes missing prerequisites an error.

Real dataset tests can use configurable roots (the existing workstation paths remain defaults):

```bash
AVSECTESTER_NUSCENES_ROOT=/path/to/nuscenes \
  python -m pytest tests/live/test_nuscenes_dataset.py -q
AVSECTESTER_NUREC_ROOT=/path/to/sample_set/26.01_release \
  python -m pytest tests/live/test_nurec_dataset.py -q
```

The nuScenes test needs `v1.0-trainval` and the devkit. The NuRec tests use the two scene UUIDs
listed in `live/test_nurec_dataset.py`. Dataset tests skip when their required artifacts are absent;
a skipped dataset test does not establish dataset-reader correctness.
