# Tests

> **Audience:** Framework and component contributors.
>
> **Outcome:** Run the appropriate test layer and extend public-behavior coverage.
>
> **Prerequisites:** [Installation](../docs/getting-started/installation.md).

Tests follow three layers. Each behavior belongs to its responsible layer. Higher layers check
the public workflow rather than repeating lower-level numerical assertions.

| Layer | Organization | Inputs |
|---|---|---|
| Framework core | One public method or feature per test | Real framework objects and shared memory components |
| Components | One contract template per family and 1–3 unique behaviors per component | Real components and synthetic observations |
| Numerical records and geometry | Public numerical functions and transport contracts | Tiny arrays, poses, rays and logs |

## Layout

```text
tests/
  avstack/
    fixtures/
      curved_lanes.xodr
      discontinuous_lanes.xodr
      referenced_controls.xodr
      straight_stop.xodr
    conftest.py
    test_carla_observation.py
    test_detector_configs.py
    test_modular.py
    test_native_records.py
    test_opendrive.py
    test_phantom_injection_attack.py
  core/
    attack/
      insertion/
        test_action.py
        test_assets.py
        test_compositor.py
        test_placement.py
        test_visibility.py
      optimization/
        test_algorithms.py
        test_perturbations.py
        test_scorers.py
      test_attack.py
      test_offline_report.py
      test_runtime.py
      test_segments.py
    closed_loop/
      avstacks/
        implemented/
          modules/
            test_detector.py
            test_planner.py
            test_ranging.py
            test_stack.py
            test_tracker.py
          __init__.py
          test_alpamayo.py
          test_camera_forward_collision.py
          test_scorers.py
        __init__.py
        conftest.py
        test_avstacks.py
        test_base.py
        test_driving_adapter.py
      simulators/
        carla/
          __init__.py
          test_conversions.py
          test_sensors.py
          test_simulator.py
        nurec/
          __init__.py
          test_renderer.py
          test_server.py
          test_simulator.py
          test_traffic.py
        __init__.py
        conftest.py
        test_base.py
        test_ego_dynamics.py
        test_ground_contact.py
        test_navigation.py
        test_recorded.py
        test_simulators.py
      test_environment.py
      test_hooks.py
      test_sensor_corruptions.py
    datasets/
      scenario_map_contract.py
      test_datasets.py
    evaluation/
      test_evaluator.py
      test_metric.py
      test_report.py
      test_visualization.py
    execution/
      test_frame_data_cache.py
      test_runner.py
      test_services.py
    experiment/
      test_configuration.py
      test_experiment.py
      test_html_report.py
      test_metadata_file.py
      test_models.py
      test_output_directory.py
      test_result_evaluation.py
      test_type_registry.py
    geometry/
      test_boxes.py
      test_cameras.py
      test_mesh.py
      test_poses.py
      test_projection.py
    implemented_attacks/
      conftest.py
      test_implemented_attacks.py
      test_object_insertion_attack.py
      test_patch_attack.py
      test_replica_visibility.py
      test_speed_feedback_attack.py
      test_stop_sign_patch_attack.py
    maps/
      test_clipgt.py
      test_scenario_map.py
    rendering/
      test_compositing.py
      test_harmonizers.py
      test_objects.py
      test_surfaces.py
      test_textures.py
    runlog/
      test_runlog.py
    scripts/
      test_scripts.py
      test_write_compose_env.py
    conftest.py
    test_architecture.py
    test_cli.py
    test_data_types.py
    test_docs.py
    test_public_api.py
    test_serialization.py
    test_type_paths.py
    test_validation.py
  live/
    test_camera_forward_collision_gpu.py
    test_carla_integration.py
    test_nurec_dataset.py
    test_nuscenes_dataset.py
    test_yolox_detector_gpu.py
  __init__.py
  conftest.py
  fakes.py
```

`tests/fakes.py` supplies shared memory components and external SDK, model, RPC and Docker
boundaries. The root `tests/conftest.py` provides native `AVStack` fixtures and live-test opt-in
settings. `tests/core/conftest.py` contains shared core records, camera fixtures and synthetic
NuRec recordings. Component-specific fixtures stay beside their owning tests.

## Writing tests

Write as many tests as you need while developing to check that a change is correct: edge cases,
error paths, coverage probes or temporary mutations. Keep these verification tests outside the
repository. Commit only tests that follow the layers above:

- A framework-core test covers one public method or feature. Its normal behavior and a few
  representative errors stay in that test, not in separate tests or per-parameter tables.
- A new component joins its family's template as one more case. Separate tests cover only
  behavior unique to that component.
- A behavior is tested once, in the layer that owns it. Higher layers do not repeat it.
- A parameterized test has one body for every case and never branches on its parameter. Two
  code paths are two tests.
- Tests use public APIs. They do not inspect private state, count implementation calls, guard
  removed names, test the shared fakes or mock framework objects.
- Only external boundaries are replaced: Docker, CARLA/avcarla, NuRec RPC, learned model weights,
  AlpaSim model dependencies and GPU execution. Shared alternative `Dataset`, `Simulator` and
  `AVStack` implementations live in `fakes.py` rather than being copied into individual
  test files.
- Coverage, test counts and line counts are not targets.

## Prerequisites

CPU builds of `torch` and `torchvision` satisfy the offline test prerequisites. CI installs
both from the CPU index. Use the test and nurec extras for the core suite, including Embree for
synthetic mesh depth and protobuf for rendered-probe requests. The supplied images provide their
corresponding native dependencies.

nuScenes tests require nuscenes-devkit and skip when it is absent, and avsectester:carla includes
it. Native `AVStack` tests require the avstack checkouts and CARLA Python client. Detector
configuration tests construct models without checkpoint weights and skip configurations whose
optional checkout or model package is absent. `avsectester:nurec` provides AlpaSim for the
AlpaSim-dependent tests.

Offline tests do not start simulator services or require model weights or a GPU.

## Offline checks

Run these commands from the checkout root inside the chosen test environment:

```bash
python -m pytest tests/core -q
python -m pytest tests/avstack tests/core -q
ruff check .
mypy
```

Compose mounts the checkout, so the image supplies dependencies and tests use the current
source. Complete the
[Compose setup](../docs/getting-started/installation.md#repository-and-compose-settings) first.
These commands disable dependencies and therefore do not start a CARLA server:

```bash
docker compose run --rm --no-deps --entrypoint python \
  -e NVIDIA_VISIBLE_DEVICES=void carla-experiment \
  -m pytest tests/core tests/avstack -q

docker compose run --rm --no-deps --entrypoint python \
  -e NVIDIA_VISIBLE_DEVICES=void nurec-experiment \
  -m pytest tests/core tests/avstack -q

docker compose run --rm --no-deps --entrypoint bash \
  -e NVIDIA_VISIBLE_DEVICES=void carla-experiment \
  -lc 'ruff check . && mypy'
```

The [CI workflow](../.github/workflows/ci.yml) separates four jobs:

| Job | Scope |
|---|---|
| core-tests | Core tests with CPU dependencies |
| avstack-tests | All native avstack and core tests: `pytest tests/avstack tests/core` |
| runlog-python312 | `RunLog` tests in a Python 3.12 environment with pytest, numpy, pyarrow and pytest's transitive dependencies |
| lint | Ruff and the configured mypy file set |

Coverage reports are informational. Recording-only tests keep heavy framework and native imports
out of collection so the Python 3.12 job needs no driving models or simulator packages.

CI skips these tests because their optional components are not installed:

- AlpaSim: traffic, controller and Alpamayo policy.
- nuScenes: dataset, recorded playback and nuScenes insertion.
- OpenMMLab: detector configuration, scorer and camera forward-collision.

**Not yet implemented.** See
[optional-component CI coverage](../docs/development/TODO.md#optional-component-tests-in-ci).

## Optional external checks

The live suite is selected explicitly:

```bash
python -m pytest tests/live -q
```

Tests skip when their stated prerequisites are absent. Dataset-only checks require no
simulator server. Environment paths must be visible inside the container. See
[external data mounts](../docs/getting-started/installation.md).

### Real NuRec dataset

Set `AVSECTESTER_NUREC_ROOT` to a licensed sample-set directory containing:

```text
023b7fcc-671c-40e3-9bd2-c66b0b073fbc/
  023b7fcc-671c-40e3-9bd2-c66b0b073fbc.usdz
```

For a checkout with that release linked under data/nurec:

```bash
docker compose run --rm --no-deps --entrypoint python \
  -e NVIDIA_VISIBLE_DEVICES=void \
  -e AVSECTESTER_NUREC_ROOT="$PWD/data/nurec/26.01_release" \
  nurec-experiment -m pytest tests/live/test_nurec_dataset.py -q
```

### Real nuScenes dataset

`AVSECTESTER_NUSCENES_ROOT` points to a nuScenes root containing v1.0-trainval metadata and its
referenced sensor files. The GPU camera-policy test uses `CAM_FRONT` images from this dataset:

```bash
docker compose run --rm --no-deps --entrypoint python \
  -e NVIDIA_VISIBLE_DEVICES=void \
  -e AVSECTESTER_NUSCENES_ROOT="$PWD/data/nuscenes" \
  carla-experiment -m pytest tests/live/test_nuscenes_dataset.py -q
```

### YOLOX GPU inference

The test requires CUDA, YOLOX and a local checkpoint. `AVSECTESTER_YOLOX_WEIGHTS` selects the
checkpoint and defaults to `models/yolox_m.pth`. Missing prerequisites skip the test with a reason.
The nurec image provides YOLOX. See
[YOLOX weights](../docs/getting-started/installation.md#yolox-detector-weights).

```bash
docker compose run --rm --no-deps --entrypoint python \
  -e AVSECTESTER_YOLOX_WEIGHTS="$PWD/models/yolox_m.pth" \
  nurec-experiment -m pytest tests/live/test_yolox_detector_gpu.py -q
```

### Camera-policy GPU inference

The test requires CUDA, nuscenes-devkit, mmdet, the native `AVStack` detector configuration and
this local checkpoint relative to the installed avstack package root:

```text
<installed-avstack-root>/third_party/mmdetection/work_dirs/carla/
  faster_rcnn_r50_fpn_1x_carla_vehicle.pth
```

Choose an available experiment GPU in `.env`. `AVSECTESTER_TEST_GPU` selects its container-local
device index:

```bash
docker compose run --rm --no-deps --entrypoint python \
  -e AVSECTESTER_NUSCENES_ROOT="$PWD/data/nuscenes" \
  -e AVSECTESTER_TEST_GPU=0 carla-experiment \
  -m pytest tests/live/test_camera_forward_collision_gpu.py -q
```

The [model setup](../docs/getting-started/installation.md) defines weight acquisition and
mounts. This test does not download weights.

### Live CARLA integration

The live integration test needs a running CARLA server, installed native models and a matching
configuration. It is disabled unless `--run-carla` is supplied. `--carla-config` selects the
configuration and defaults to `configs/carla_phantom.yaml`:

```bash
docker compose run --rm --no-deps --entrypoint python carla-experiment \
  -m pytest tests/live/test_carla_integration.py --run-carla \
  --carla-config configs/carla_phantom.yaml -q
```

Server setup is described in [Installation](../docs/getting-started/installation.md). The live test
executes the public preparation, execution and evaluation workflow.

## Next steps

Read [Contributing](../docs/development/contributing.md)
and [Architecture](../docs/concepts/architecture.md).
