# Docker — reproducible environment + example run

The `Dockerfile` builds a **minimal, torch-free / CUDA-free** image that runs the CARLA-free
end-to-end example (an attack against the mock AV pipeline). It builds in ~30s on any machine
because the mock uses avstack's passthrough perception + tracker, which are pure Python — none
of the heavy GPU stack (torch / mmcv / mmdet3d / CARLA) is needed.

## Build

```bash
git clone --recurse-submodules <this-repo> && cd AVSecTester
git submodule update --init third_party/avstack-core     # the mock needs only avstack-core
docker build -t avsectester:mock .
```

> The image installs `avstack-core` with its **base** dependencies only (the `percep` extra —
> torch/mmcv/mmdet3d — is skipped). `.dockerignore` excludes the GPU-only vendored code
> (`third_party/avstack-core/third_party`, `third_party/nuCarla`) and all data/model/output trees.

## Run the example attack (end-to-end)

```bash
docker run --rm avsectester:mock
# or: docker compose run --rm avsectester
```

This runs `avsectester run configs/mock_experiment.yaml`, which executes three passes of the
runtime loop and prints an impact report:

- **clean** — the ego cruises to ~5.7 m/s, never brakes.
- **attacked** — `ObjectSpoofingAttack` injects a phantom obstacle at the `perception_input`
  seam → the ego brakes and stops (`impacted: True`, `stopped: True`).
- **attacked + defended** — `ScoreGateDefense` gates out the low-confidence phantom → the ego
  cruises again (`Mitigated: True`).

Other commands:

```bash
docker run --rm avsectester:mock pytest -q                    # offline test suite (12 pass, 1 skip)
docker run --rm avsectester:mock avsectester registry        # list registered plugins
docker run --rm -v "$PWD/configs:/app/configs" avsectester:mock \
    avsectester run configs/mock_experiment.yaml             # edit configs on the host, run in the image
```

## Full end-to-end: neural perception + closed-loop CARLA (GPU)

`Dockerfile.gpu` reproduces the full stack (torch 1.13+cu117, compiled mmdet3d ops, avstack +
CARLA client) and `docker-compose.gpu.yml` wires it to a `carlasim/carla:0.9.15` server. This
runs the **real** end-to-end path: a CARLA-trained PointPillars detector on a live CarlaLidar in
a closed-loop drive, attacked by a phantom detection.

Prerequisites: an NVIDIA GPU with the nvidia container runtime, and the nested mm* submodules on
the host (needed to compile mmdet3d):

```bash
cd third_party/avstack-core && \
  git submodule update --init --depth 1 third_party/mmdetection third_party/mmdetection3d third_party/mmsegmentation && cd -
./scripts/fetch_models.sh          # pull the carla-vehicle weights → ./models (mounted into the image)
```

Then bring the whole thing up:

```bash
docker compose -f docker-compose.gpu.yml up --build
```

It builds the GPU image (the mmdet3d CUDA compile takes ~10–20 min the first time), starts the
CARLA server, waits for it, and runs `scripts/smoke_carla_neural.py` — a clean pass then a
phantom-attacked pass. Expected: the clean run cruises while the detector reports real NPC
detections; the attacked run brakes and stops (`SMOKE: PASS`).

Notes:
- `nvcc` 11.7 can't target Ada (sm_89) directly, so the image emits `sm_86` cubin + PTX that JITs
  on the L40S (`TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"`).
- `Dockerfile.gpu.dockerignore` keeps `third_party/avstack-core/third_party` (the mm* sources)
  that the light `.dockerignore` excludes.
- `./models` is a bind mount, so weights are shared with the host rather than baked into the image;
  `fetch_models.sh` runs at container start to (re)create the mmdet3d symlinks.
- The manual (conda) install is documented in [`SETUP.md`](SETUP.md).
