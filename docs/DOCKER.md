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

## Full stack (neural perception + CARLA)

The neural-perception path (real CARLA-trained detector) and closed-loop CARLA need the full
GPU stack (torch 1.13+cu117, compiled mmdet3d ops) and a CARLA 0.9.15 server — that install is
documented in [`SETUP.md`](SETUP.md) and is intentionally *not* in this lightweight image.
