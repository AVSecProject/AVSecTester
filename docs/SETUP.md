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
pip install -e ".[dev]"
pytest            # scaffold tests pass without the heavy stack
```

## 2. Full stack (Python 3.11 — upgraded, `dependencies` branch)

Modern stack: **Python 3.11, torch 2.5.1+cu124, torchvision 0.20.1, numpy 2.2.6, mmcv 2.2.0,
mmdet 3.3.0, mmdet3d 1.4.0, AlbumentationsX 2.x, CARLA 0.9.16**. (The old numpy<1.26 / torch1.13+cu117 /
mmdet3d 1.1.0 stack lives in git history before the `dependencies` branch.)

> **CARLA version note.** CARLA **0.9.15** has no Python-3.11 wheel; PyPI ships **0.9.16** for cp311,
> which imports fine (client only — pair with a matching 0.9.16 server for a live run). `avcarla` uses
> only stable API, so this is transparent to it.

Install torch/mmcv/mm-detectors explicitly **before** the avstack packages, in this order:

```bash
conda activate avsec311
# 1. torch + torchvision (cu124 index) — L40S (Ada, sm_89) wants torch 2.x + CUDA 12
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
# 2. mmcv 2.2.0: the torch2.4/cu121 prebuilt wheel is ABI-compatible with torch2.5+cu124 (no source
#    build needed — verified: mmcv.ops CUDA kernels load) + mmengine
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html
pip install "mmengine>=0.10"
# 3. mmdet 3.3.0 + mmdet3d 1.4.0 from PyPI. Their mmcv<2.2.0 assert is conservative; mmcv 2.2.0 works,
#    so relax the version cap in the two installed __init__.py (documented workaround):
pip install --no-deps mmdet==3.3.0 mmdet3d==1.4.0
pip install scikit-image terminaltables plyfile trimesh tensorboard networkx   # their runtime deps
SP=$(python -c "import site; print(site.getsitepackages()[0])")
sed -i "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" "$SP/mmdet/__init__.py" "$SP/mmdet3d/__init__.py"
sed -i "s/mmdet_maximum_version = '3.3.0'/mmdet_maximum_version = '3.4.0'/" "$SP/mmdet3d/__init__.py"
# 4. avstack packages (base avstack-core, NOT [percep]) — pins now allow numpy2/torch2.5
pip install -e third_party/avstack-core -c constraints.txt   # constraints.txt = "numpy>=2,<3"
pip install -e third_party/avstack-api  -c constraints.txt
pip install -e third_party/lib-avstack-carla --no-deps
pip install carla==0.9.16 pygame ipywidgets nuscenes-devkit   # nuscenes-devkit numpy<2 pin is conservative (works on numpy 2)
# 5. AlbumentationsX augmentation backend
pip install -e ".[augment]"
```

> **No CUDA compilation needed.** In OpenMMLab 2.0 the CUDA ops live in **mmcv** (prebuilt wheel above);
> mmdet/mmdet3d are pure-Python, so there is no mmdet3d source build (unlike the old stack).

> **CUDA build note (mmdet3d ops).** This box has system nvcc **11.5** (`/usr/bin/nvcc`, full
> toolkit under `/usr/include`) and a too-new `/usr/local/cuda` → 13.3. torch is built for 11.7, so
> build mmdet3d's CUDA ops with `CUDA_HOME=/usr` (the 11.5 toolkit; 11.5-vs-11.7 minor mismatch is
> tolerated). The GPUs are **L40S = sm_89 (Ada)**, which nvcc 11.5 can't target directly — pin
> `TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"` so it emits sm_86 cubin (forward-compatible on sm_89) plus
> PTX JIT fallback.

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

CARLA **server** (0.9.15) runs separately via docker (the box's default docker runtime is nvidia):

```bash
docker pull carlasim/carla:0.9.15
docker run -d --name carla-avsec --gpus 'device=2' --net=host \
  carlasim/carla:0.9.15 ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -quality-level=Epic
```

Run the simulator in **synchronous mode** for reproducible perception.

## 3. Verify

```bash
avsectester version                                          # prints the version
python -m pytest tests/ -q                                   # offline: attack hook + pipeline
avsectester run configs/carla_scenario.yaml --frames 40 --gpu 1   # end-to-end (needs a CARLA server)
```

`avsectester run` runs the scenario clean then phantom-attacked and asserts the attack forced an unsafe
stop. On a single host, `--gpu 1` keeps neural inference off GPU 2 (which CARLA is rendering on).

## 4. NuRec + Alpamayo (the end-to-end path)

This tier drives the real **Alpamayo-1.5-10B** policy on photoreal **NuRec** imagery. It reuses two
NVIDIA [AlpaSim](https://github.com/NVlabs/alpasim) pieces: the `nre-ga` renderer (serves a
reconstructed scene over gRPC) and `alpasim_driver` (provides the Alpamayo model). Alpamayo pins a
**Python-3.12** environment, separate from avstack's 3.10 — heavy imports in `stacks/alpamayo.py` are
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
