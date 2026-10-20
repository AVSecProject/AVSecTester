# Tests and CI

Offline tests use Python 3.11. Select the core suite explicitly in an installation without avstack.
Native AVStack implementation tests use importorskip when the optional native library is absent; a
skipped test does not validate that implementation. Required CPU dependencies are not optional.

| Directory | Coverage | Prerequisites |
|---|---|---|
| core/ | Contracts, datasets, attacks, experiment orchestration, logs, evaluation, CLI and docs | CPU test dependencies |
| avstack/ | Native detection geometry, pipeline propagation, ModularAVStack and native codecs | Core dependencies, avstack packages and CARLA Python client |
| live/ | CARLA integration, dataset-reader checks and a GPU detector-capture check | --run-carla, dataset roots, or a GPU with detector weights and nuScenes data |

## Core tests

```bash
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]' -c constraints.txt
NO_ALBUMENTATIONS_UPDATE=1 python -m pytest tests/core -q
```

## Offline avstack tests

```bash
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --depth 1 -- \
  third_party/avstack-core third_party/avstack-api third_party/lib-avstack-carla
python -m pip install -e third_party/avstack-core \
  carla==0.9.16 pygame ipywidgets 'opencv-python<4.12' -c constraints.txt
python -m pip install --no-deps -e third_party/avstack-api -e third_party/lib-avstack-carla
NO_ALBUMENTATIONS_UPDATE=1 python -m pytest tests/avstack -q
```

The suites include an in-memory CliRunner run-to-evaluate integration, fake native worlds,
calibrated insertion fixtures and log persistence. The packages use synthetic observations and
server stand-ins. They do not require CUDA or model checkpoints. A full native experiment
environment has additional dependencies in [Installation](../docs/getting-started/installation.md).

## Functional coverage

Tests mirror datasets/, attack/, closed_loop/, experiment/, runlog/ and evaluation/. Concrete
attacks mirror core/attack/implemented_attacks/; insertion and optimization tools have
core/attack/insertion/ and core/attack/optimization/. Shared geometry lives in core/geometry/.
AVStack implementations live in core/closed_loop/avstacks/, including native-library-dependent
checks. tests/avstack/ also contains AVStack checks in test_scenario.py, test_runlog_codecs.py and
test_detector_configs.py, alongside native simulator/codec checks. Geometry and sensor-transform
tests cover projection, visibility and deterministic pixel transforms.

Important behavioral checks include:

- Metadata decision windows, optional role/placement fields and per-run attack state.
- One offline preparation per attack, explicit artifacts and validated save/load.
- CARLA source initialization, shared deployment/capture ticks and forward-scan equivalence.
- Required role actor existence under each seed and shared execution caches.
- Sensor noise consistency across offline requests and online absolute timeline positions.
- Canonical ego/actor boxes, detection recording and Parquet schema round trips.
- Log-only metrics, missing-evidence reasons, summaries and independent evaluation loading.
- Supplied configuration loading, CLI exit codes and prevention of output overwrites.
- Documentation links, Python/YAML syntax and examples marked runnable.

Alpamayo core tests use prescribed model outputs and schema stand-ins; NuRec core tests use
StubRenderer. They validate adapter behavior without establishing real service compatibility or
attack strength. Native simulator timing and GPU memory behavior need live checks.

## GitHub Actions

[ci.yml](../.github/workflows/ci.yml) runs on pull requests, pushes to main and manual dispatch. It
covers core-tests, avstack-tests, lint and runlog-python312. Only the latter uses Python 3.12 and
checks Parquet recording with the driver interpreter version. The workflow does not publish packages
or images. JUnit/coverage artifacts are retained by CI; missing native prerequisites are not counted
as successful live validation.

## Live tests

Use a dedicated CARLA 0.9.16 server and matching Python client. Neural runs additionally need model
checkpoints and an available GPU. The only simulator integration test is
tests/live/test_carla_integration.py. Enable it with --run-carla. --carla-config PATH is optional
and defaults to configs/carla_phantom.yaml. There is no NuRec integration test.

```bash
python -m pytest tests/live/test_carla_integration.py --run-carla \
  --carla-config configs/carla_phantom.yaml -q
```

tests/live/test_component_log_gpu.py checks that stage capture records a real detector output. It
needs CUDA, the CARLA 2D Faster R-CNN weights fetched by scripts/preparation/fetch_models.sh, and
nuScenes v1.0-trainval data (AVSECTESTER_NUSCENES_ROOT); otherwise it is skipped.

Real dataset tests accept configured roots:

```bash
AVSECTESTER_NUSCENES_ROOT=/path/to/nuscenes \
  python -m pytest tests/live/test_nuscenes_dataset.py -q
AVSECTESTER_NUREC_ROOT=/path/to/sample_set/26.01_release \
  python -m pytest tests/live/test_nurec_dataset.py -q
```

A skipped dataset test does not establish reader correctness. Dataset-reader tests do not run full
experiments. Keep separately performed simulator validation results distinct from offline counts.
Delete temporary validation images and logs after recording the outcome when they are no longer
needed.
