# Tests

Tests follow three layers. Each behavior belongs to its responsible layer; higher layers check the
public workflow rather than repeating lower-level numerical assertions.

| Layer | Organization | Inputs |
|---|---|---|
| Framework core | One file per public class/module; one test per public method or feature | Real framework objects with shared in-memory implementations |
| Components | One parameterized contract template per component family; 1–3 unique behaviors per component | Real components and small synthetic observations |
| Core data structures | Numerical tests for geometry, image transport, insertion and serialization | Tiny arrays, poses, rays and logs |

## Layout

```text
tests/
  fakes.py                         shared memory world and native dependency boundaries
  conftest.py                      live opt-in, native observations and shared NuRec recordings
  core/
    experiment/                    configuration, type discovery, output, execution and evaluation
    execution/
      test_runner.py               decision lifecycle and actions through run()
      test_services.py             frame-data cache, isolated sessions and candidate operations
      test_frame_data_cache.py     generated-frame cache identity and publication
    closed_loop/
      test_hooks.py                HookSet, Hook and Plugin
      test_environment.py          environment validation and deterministic sensor corruptions
      simulators/
        test_base.py               shared public simulator lifecycle
        test_ego_dynamics.py       actuator integration, ideal following and optional real MPC
        test_navigation.py         map/recorded route generation
        test_ground_contact.py     mesh-constrained ego height and orientation
        test_simulators.py         shared simulator contract
        carla/                     native lifecycle, sensor capture and conversions
        nurec/                     runtime and renderer behavior
        test_recorded.py           replay aliases and terminal support
      avstacks/
        test_base.py               driving interface and observation pipeline
        test_avstacks.py           AVStack/DrivingAdapter component template
        test_scorers.py            model scorer bindings
        test_alpamayo.py           actual input history, routes, trajectories and model ownership
        test_camera_forward_collision.py   image detection policy
    datasets/                      dataset contract template and interpolation
    maps/                          semantic records and clipgt map reader
    attack/
      test_attack.py               Attack and AttackSegment contracts
      test_segments.py             eligible decision-window splitting
      optimization/                PGD/NES objective-decrease template
      insertion/                   geometry, alpha/depth composition, visibility and textures
    implemented_attacks/           shared attack lifecycle and concrete attack behaviors
    geometry/                      pose, camera, sampling-grid and visibility numerical tests
    rendering/                     object assets, textures, surfaces and compositing
    runlog/                        Recorder/RunLog recording and reading APIs
    evaluation/                    metric template, report formatting and visualization
    test_data_types.py                shared records, detection and sensor-data transport
    test_validation.py             integer and finite real-number validation
    test_type_paths.py             callable imports by explicit module paths
    test_serialization.py          JSON serialization and independent snapshots
    test_cli.py                    run, evaluate and version
    test_docs.py                   links, parsed blocks and executable examples
  avstack/                         real native pipeline, detector forward, native records and phantom
  live/                            optional real-server, real-data and GPU checks
```

The root conftest.py provides nurec_usdz_path and nurec_longitudinal_recording to core tests.

## Writing tests

Write as many tests as you need while developing to check that a change is correct: edge cases,
error paths, coverage probes or temporary mutations. Keep these verification tests outside the
repository. Commit only tests that follow the layers above:

- A framework-core test covers one public method or feature. Its normal behavior and a few
  representative errors stay in that test, not in separate tests or per-parameter tables.
- A new component joins its family's template as one more case. Separate tests cover only
  behavior unique to that component.
- A behavior is tested once, in the layer that owns it. Higher layers do not repeat it.
- A parameterized test has one body for every case and never branches on its parameter. Two code
  paths are two tests.
- Tests use public APIs. They do not inspect private state, count implementation calls, guard
  removed names, test the shared fakes or mock framework objects.
- Only external boundaries are replaced: CARLA/avcarla, NuRec RPC, learned model weights,
  AlpaSim model dependencies and GPU execution. Shared alternative Dataset, Simulator and AVStack
  implementations live in fakes.py rather than being copied into individual test files.
- Coverage, test counts and line counts are not targets.

## Offline checks

Run from the repository root:

```bash
python -m pytest tests/core -q
python -m pytest tests/avstack tests/core/closed_loop/avstacks \
  tests/core/implemented_attacks/test_implemented_attacks.py -q
ruff check .
mypy
```

The full offline suite needs both test and nurec extras, including Embree for synthetic NuRec depth.
Native tests additionally need avstack and the native
SDK dependencies. The detector template builds each configured model and forwards synthetic
inputs without checkpoint weights. Cases explicitly skip when an optional detector package or its
external model configuration checkout is unavailable. Core memory tests require no model weights,
simulator server or GPU.

For the existing Docker environment, run the complete offline suite:

```bash
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -e NVIDIA_VISIBLE_DEVICES=void -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/app \
  -v "$PWD":/app -w /app --entrypoint bash avsectester:latest \
  -lc 'python -m pytest -p no:cacheprovider -q && ruff check . && mypy'
```

core-tests runs core tests on Python 3.11; avstack-tests exercises the real native pipeline and
the attack lifecycle template, including phantom injection; runlog-python312 runs
core/runlog/test_runlog.py with the recording-only Python 3.12 dependencies; lint runs Ruff and
mypy. Coverage reports are informational.

## Optional external checks

The only live simulator integration test is live/test_carla_integration.py. Enable it explicitly
with --run-carla; --carla-config defaults to configs/carla_phantom.yaml:

```bash
python -m pytest tests/live/test_carla_integration.py --run-carla \
  --carla-config configs/carla_phantom.yaml -q
```

There is no NuRec server integration test. Real dataset checks use AVSECTESTER_NUREC_ROOT and
AVSECTESTER_NUSCENES_ROOT. The camera-policy GPU test additionally needs CUDA, detector weights and
nuScenes v1.0-trainval images; AVSECTESTER_TEST_GPU selects its device. These checks skip when their
external prerequisites are absent. Offline checks do not launch simulator servers.

AlpaSim-dependent policy and controller tests use pytest.importorskip outside the Python 3.12
NuRec image. Inside avsectester:nurec, they use the installed AlpaSim protobufs and navigation,
replace only learned model weights, and run without a renderer server or GPU.
