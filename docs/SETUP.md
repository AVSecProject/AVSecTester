# Setup

AVSecTester has two install tiers:

- **Core only** (no GPU, no simulator): the schema/registry/DAG layer + tests.
- **Full avstack stack** (GPU + CARLA): closed-loop execution and perception.

## 0. Clone with submodules

The avstack projects are vendored as git submodules under `third_party/`:

```bash
git clone --recurse-submodules <this-repo>
# or, after a plain clone:
git submodule update --init            # top-level avstack repos (NOT their nested mmdet submodules)
```

## 1. Core-only install (fast, CI-friendly)

```bash
conda create -y -n avsec python=3.10 && conda activate avsec
pip install -e ".[dev]"
pytest            # scaffold tests pass without the heavy stack
```

## 2. Full stack (Python 3.10 required)

avstack pins **Python 3.10, torch 1.13.1+cu117, torchvision 0.14.1, mmcv 2.0.1, mmdet 3.0.0,
mmdet3d 1.1.0**. Do **not** chase newer torch/mmdet for the closed-loop path — match avstack's stack.

> **CARLA version note.** avstack's docs say CARLA **0.9.13**, but 0.9.13 has **no Python-3.10
> client** (only cp27/cp37 wheels ship in the release and docker image). Since the whole stack is
> Python 3.10, we use the **CARLA 0.9.15** client + `carlasim/carla:0.9.15` server instead — the
> closest cp310-capable release. `avcarla` pins no CARLA version and uses only stable API, so this
> is transparent to it.

Because avstack-core's exact pins live in its `[tool.uv.sources]` (which only `uv` honors), a plain
`pip` install must reproduce them by hand — install torch/mmcv/mm-detectors explicitly **before**
the avstack packages, in this order:

```bash
conda activate avsec
# 1. torch + torchvision (cu117 index)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --index-url https://download.pytorch.org/whl/cu117
# 2. mmcv prebuilt wheel (built for torch1.13.1/cu11.7 — no compilation) + mmengine
pip install "https://g-b0ef78.1d0d8d.03c0.data.globus.org/packages/mmcv/torch1.13.1_cu11.7/mmcv-2.0.1-cp310-cp310-linux_x86_64.whl" "mmengine>=0.7.3,<0.8"
pip install "numpy==1.24.4"   # mmengine pulls numpy 2.x; pin back under avstack's <1.26
# 3. editable mm-detectors (init nested submodules first: cd third_party/avstack-core && git submodule update --init --depth 1 third_party/mmdetection third_party/mmdetection3d third_party/mmsegmentation)
pip install -e third_party/avstack-core/third_party/mmdetection    -c constraints.txt   # constraints.txt = "numpy<1.26"
pip install -e third_party/avstack-core/third_party/mmsegmentation -c constraints.txt
# mmdet3d compiles CUDA ops — see the CUDA note below
CUDA_HOME=/usr FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX" MAX_JOBS=16 \
  pip install -e third_party/avstack-core/third_party/mmdetection3d -c constraints.txt
# 4. avstack packages (base avstack-core, NOT [percep] — that would re-pull mmcv/mmdet3d from PyPI)
pip install -e third_party/avstack-core -c constraints.txt
pip install -e third_party/avstack-api  -c constraints.txt
pip install -e third_party/lib-avstack-carla --no-deps    # name-shim; deps already present
pip install carla==0.9.15 pygame -c constraints.txt        # pygame is an undeclared avcarla dep
```

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
docker run -d --name carla-avsec --gpus 'device=0' --net=host \
  carlasim/carla:0.9.15 ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -quality-level=Epic
```

Run the simulator in **synchronous mode** for reproducible perception.

## 3. Verify

```bash
avsectester version     # prints version + whether the avstack stack imported
avsectester registry    # lists registered attacks/defenses/backends/metrics
```
