# Setup

AVSecTester has three install tiers:

- **Core only** (no GPU, no simulator): the interface (`plane`/`backend`), the in-process NuRec
  backend on `StubRenderer`, the attack/metric seams, and the offline tests.
- **Full avstack stack** (GPU + CARLA): the `CarlaBackend` + `ModularAVStack` closed loop and neural
  perception (§2).
- **NuRec + Alpamayo** (GPU + the AlpaSim driver env): the `NuRecBackend` + `AlpamayoAVStack` closed
  loop — the real Alpamayo policy on photoreal NuRec imagery (§4).

## 0. Clone with submodules

The avstack projects are vendored as git submodules under `third_party/`:

```bash
git clone --recurse-submodules <this-repo>
# or, after a plain clone:
git submodule update --init            # top-level avstack repos (NOT their nested mmdet submodules)
```

## 1. Core-only install (fast, CI-friendly)

```bash
conda create -y -n avsec311 python=3.11 && conda activate avsec311
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,test]" -c constraints.txt
python -m pytest tests/core -q    # no avstack, CARLA server or checkpoints
```

See [tests/README.md](../tests/README.md) for the offline avstack suite and CI environment.

## 2. Full stack (GPU + CARLA)

Modern **OpenMMLab-official** stack (every version cap satisfied → no patching, no source builds):
**Python 3.11, torch 2.1.0+cu121, torchvision 0.16.0, numpy 1.26.4, mmcv 2.1.0, mmdet 3.2.0,
mmdet3d 1.4.0, albumentations 1.4 (MIT), CARLA 0.9.16**.

> **Why torch 2.1.0, not newer?** mmdet3d 1.4.0 caps `mmcv<2.2.0`, and mmcv 2.1.0 (the newest allowed)
> only ships prebuilt wheels up to **torch 2.1.0**. Anything newer forces mmcv 2.2.0, which breaks the cap
> (needs patching) and/or a source build. torch 2.1.0+cu121 still fully supports the L40S (Ada sm_89) and
> gives torch 2.x + CUDA 12.1. It is numpy<2, so the augmentation backend is classic **albumentations 1.4**
> (MIT) rather than AlbumentationsX/2.x (which require numpy≥2, hence torch≥2.4 → the cap conflict).

> **CARLA version note.** CARLA **0.9.15** has no Python-3.11 wheel; PyPI ships **0.9.16** for cp311,
> which imports fine (client only — pair with a matching 0.9.16 server for a live run). `avcarla` uses
> only stable API, so this is transparent to it.

Install torch/mmcv/mm-detectors explicitly **before** the avstack packages, in this order:

```bash
conda activate avsec311
# 1. torch + torchvision (cu121 index) — torch 2.x + CUDA 12.1 on the L40S (Ada, sm_89)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
# 2. mmcv 2.1.0 prebuilt wheel for torch2.1.0/cu121 (no source build) + mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
pip install "mmengine>=0.10"
# 3. mmdet 3.2.0 + mmdet3d 1.4.0 from PyPI — all version caps are satisfied by mmcv 2.1.0, so NO patching
pip install --no-deps mmdet==3.2.0 mmdet3d==1.4.0
pip install "numpy==1.26.4" "scikit-image>=0.21,<0.24" "pandas>=2,<3" \
            pycocotools terminaltables plyfile trimesh tensorboard networkx   # deps, kept numpy<2
# 4. avstack packages (base avstack-core, NOT [percep])
pip install -e third_party/avstack-core -c constraints.txt   # constraints.txt = "numpy>=1.26,<2"
pip install -e third_party/avstack-api  --no-deps
pip install -e third_party/lib-avstack-carla --no-deps
pip install carla==0.9.16 pygame ipywidgets nuscenes-devkit -c constraints.txt
# 5. augmentation backend (classic albumentations 1.4, MIT)
pip install -e ".[augment]"
```

> **Known submodule-name shim.** `lib-avstack-carla`'s own metadata references sibling path
> deps `../lib-avstack-core` and `../lib-avstack-api`, but our submodules are named
> `avstack-core` / `avstack-api`. Create compatibility symlinks before installing avcarla:
>
> ```bash
> ln -s avstack-core third_party/lib-avstack-core
> ln -s avstack-api  third_party/lib-avstack-api
> ```

Capture a working lockfile once it succeeds (`pip freeze > requirements.lock`) — reproducing this
install is the #1 adoption risk (PLAN.md).

CARLA **server** (0.9.16) runs separately via docker (the box's default docker runtime is nvidia):

```bash
docker pull carlasim/carla:0.9.16
docker run -d --name carla-avsec --gpus 'device=2' --net=host \
  carlasim/carla:0.9.16 ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -quality-level=Epic
```

Run the simulator in **synchronous mode** for reproducible perception.

## 3. Verify

```bash
avsectester version                                          # prints the version
python -m pytest -q                                         # offline groups; install .[test] first
avsectester run configs/carla_scenario.yaml --frames 40 --gpu 1   # end-to-end (needs a CARLA server)
```

`avsectester run` runs the scenario clean then phantom-attacked and asserts the attack forced an unsafe
stop. On a single host, `--gpu 1` keeps neural inference off GPU 2 (which CARLA is rendering on).

## 4. NuRec + Alpamayo (the end-to-end path)

This tier drives the real **Alpamayo-1.5-10B** policy on photoreal **NuRec** imagery. It reuses two
NVIDIA [AlpaSim](https://github.com/NVlabs/alpasim) pieces: the `nre-ga` renderer (serves a
reconstructed scene over gRPC) and `alpasim_driver` (provides the Alpamayo model). Alpamayo pins a
**Python-3.12** environment, separate from avstack's 3.11 — heavy imports in `stacks/alpamayo.py` are
lazy, so the base install still imports and tests AVSecTester without it.

**a. The Alpamayo driver env.** Clone AlpaSim and build its driver workspace (`uv sync --package
alpasim_driver`); the local checkpoint lives at
`/workspace/hdd/models/huggingface/nvidia/Alpamayo-1.5-10B`. Run AVSecTester scripts from that env
with AVSecTester on `PYTHONPATH`:

```bash
cd /workspace/nvme/qzzhang/alpasim
HF_HOME=/workspace/hdd/models/huggingface PYTHONPATH=/workspace/nvme/qzzhang/AVSecProject/AVSecTester \
  uv run python /workspace/nvme/qzzhang/AVSecProject/AVSecTester/scripts/alpamayo_nurec_demo.py 8 --gpu 1
```

Add `--stub` to swap `NuRecRenderer` for `StubRenderer` (black frames — no renderer needed) to sanity
-check the loop, and `--save-frames` to dump each rendered frame under `./tmp/alpamayo_nurec/`.

**b. The NuRec renderer (`nre-ga`) + a scene.** Pull a NuRec scene (license-gated dataset; approve it
once on your HF account, and pass `HF_TOKEN` explicitly when `HF_HOME` is overridden):

```bash
HF_TOKEN=$(cat ~/.cache/huggingface/token) hf download nvidia/PhysicalAI-Autonomous-Vehicles-NuRec \
  --repo-type dataset --revision 26.01 "sample_set/26.01_release/<uuid>/<uuid>.usdz" \
  --local-dir /workspace/hdd/datasets/huggingface/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec
```

Serve it on `:50051` (GPU 2, alongside CARLA). The `--entrypoint` is required — the image's default
entrypoint swallows the command:

```bash
docker run -d --name nre --net=host --gpus '"device=2"' -e HOME=/tmp \
  -v <scenes-dir>/sample_set/26.01_release:/mnt/nre-data \
  --entrypoint /app/internal/scripts/pycena/runtime/pycena_nrm_full \
  nvcr.io/nvidia/nre/nre-ga:26.04 serve-grpc --host=0.0.0.0 --port=50051 \
  '--artifact-glob=/mnt/nre-data/**/*.usdz' --cache-size=2 --enable-editing-actors
```

The scene loads under an id like `clipgt-<uuid>`; `NuRecRenderer(scene_id="<substring>")` resolves it
via `get_available_scenes()`. **GPU layout:** the CARLA server and the NRE renderer both run on
**GPU 2**; perception / AV models (Alpamayo, PointPillars) run on GPU 1 (`--gpu 1`).
