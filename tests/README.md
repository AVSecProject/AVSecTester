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
  conftest.py                      live opt-in and native observation fixtures
  core/
    experiment/                    configuration, prepare/execute, persistence
    closed_loop/
      test_runner.py               decision lifecycle and effects through run()
      test_services.py             frame-data cache, isolated sessions and candidate operations
      test_simulator.py            shared public simulator lifecycle
      test_hooks.py                HookSet, Hook and Plugin
      test_environment.py          environment validation and deterministic transforms
      test_datatypes.py            detection and camera RGB numerics
      simulators/                  CARLA, NuRec and recorded replay contract template
      avstacks/                    AVStack/DrivingAdapter template and model scorer bindings
    datasets/                      dataset contract template and interpolation
    attack/
      test_attack.py               Attack, segments and placement specifications
      implemented_attacks/         one shared attack lifecycle template and unique behaviors
      optimization/                PGD/NES objective-decrease template
      insertion/                   geometry, alpha/depth composition, visibility and textures
    geometry/                      pose and camera numerical tests
    runlog/                        Recorder/RunLog and codec registry public APIs
    evaluation/                    metric template, evaluation reports and plots
    test_cli.py                    run, evaluate and version
    test_docs.py                   links, parsed blocks and executable examples
  avstack/                         real native pipeline, detector forward, codecs and phantom
  live/                            optional real-server, real-data and GPU checks
```

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
  tests/core/attack/implemented_attacks/test_implemented_attacks.py -q
ruff check .
mypy
```

The core environment needs the test extra. Native tests additionally need avstack and the native
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
