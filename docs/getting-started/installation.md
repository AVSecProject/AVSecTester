# Installation

> **Audience:** Anyone setting up AVSecTester for the first time.  
> **Outcome:** A working environment for the experiments you plan to run, with the required services and model weights.  
> **Prerequisites:** A Linux host. NVIDIA GPUs for simulator and neural-model runs.


## Choose an installation

Install only what your experiments need. Each option supports a different set of
[simulators and driving systems](../concepts/architecture.md#supported-combinations). Model weights
and NuRec scene archives are never part of an environment. Download them separately, as described
in [Model weights and assets](#model-weights-and-assets).

| Installation | Supports | Requires |
|---|---|---|
| [Docker](#docker) | CARLA experiments with the modular driving system and neural perception, through the CLI or Python. The offline test suites against a mounted checkout | NVIDIA GPUs, Docker Compose with the NVIDIA runtime, CARLA perception checkpoints |
| [Local core](#local-core-installation) | The driving loop and interfaces, runtime handlers, preparation, metrics, NuRec with the offline `StubRenderer`, and the `core` offline tests. With the `rendering` extra, insertion rendering and geometric visibility | Python 3.11. No GPU, simulator or submodules |
| [Local full](#local-full-installation) | The same as Docker, installed on the host, including both offline test groups | Python 3.11, NVIDIA GPUs with CUDA 12.1 wheels, a CARLA 0.9.16 server, CARLA perception checkpoints |
| [NuRec and Alpamayo](#nurec-and-alpamayo-environment) | NuRec rendering with the Alpamayo driving system, through the NuRec demo scripts and Python. No CLI | The AlpaSim driver environment (Python 3.12), the `nre-ga` rendering service, a NuRec scene, the Alpamayo checkpoint, NVIDIA GPUs |

For CARLA, Docker is the shortest path. The image installs every Python dependency, and Compose
starts the CARLA server next to it. The local full installation installs the same dependency set
on the host.

## Clone the repository

```bash
git clone https://github.com/AVSecProject/AVSecTester.git
cd AVSecTester
```

External projects are Git submodules under `third_party/`. Initialize only the ones you need:

| Submodules | Needed for |
|---|---|
| `third_party/avstack-core`, `third_party/avstack-api`, `third_party/lib-avstack-carla` | Docker, the local full installation and the `avstack` offline tests |
| `mmdetection` and `mmdetection3d`, nested in `third_party/avstack-core` | CARLA perception checkpoints, in Docker and the local full installation |
| `third_party/libcom` | The [PCTNet harmonizer](#pctnet-harmonizer) |

The local core installation needs no submodules. For Docker or the local full installation,
initialize the avstack projects and the nested detector checkouts:

```bash
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --depth 1 -- \
  third_party/avstack-core third_party/avstack-api third_party/lib-avstack-carla
git -C third_party/avstack-core -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init --depth 1 -- third_party/mmdetection third_party/mmdetection3d
```

`.gitmodules` records SSH URLs for some submodules. The `-c url...insteadOf` option fetches them
over HTTPS for this command only, as CI does, without SSH keys or changes to your global Git
settings. The commands do not fetch the other nested dependencies of the avstack projects.

## Docker

The `Dockerfile` builds an image with the CARLA path's dependencies. `docker-compose.yml` runs that
image next to a `carlasim/carla:0.9.16` server container.

The image contains a Python 3.11.13 environment installed with uv, PyTorch 2.1.0 for CUDA 12.1,
torchvision 0.16.0, MMCV 2.1.0 with prebuilt CUDA operators, MMDetection 3.2.0, MMDetection3D
1.4.0, the three avstack packages, the CARLA 0.9.16 Python client, `nuscenes-devkit`,
`lyft-dataset-sdk`, the pinned PyArrow dependency for Parquet logs, and AVSecTester with its
`dev`, `viz` and `augment` extras. No host Python
environment is needed. The image does not contain the AlpaSim driver environment, Alpamayo or the
NuRec renderer client. Those use the [NuRec and Alpamayo environment](#nurec-and-alpamayo-environment).

### Prerequisites

- NVIDIA GPUs, Docker with Compose, and the NVIDIA container runtime configured as Docker's
  default runtime. The Compose file selects GPUs only through `NVIDIA_VISIBLE_DEVICES`. It uses
  physical GPU 2 for CARLA and physical GPU 1 for AVSecTester. Edit both values in
  `docker-compose.yml` to match your machine before starting it.
- Free host ports. Both services use `network_mode: host`. The CARLA server listens on port 2000,
  and `configs/carla_scenario.yaml` uses traffic manager port 8000.
- The avstack submodules and the nested detector checkouts from
  [Clone the repository](#clone-the-repository). The build copies the checkout into the image.
- The CARLA perception checkpoints on the host, fetched once from the repository root:

  ```bash
  ./scripts/preparation/fetch_models.sh
  ```

  See [CARLA perception checkpoints](#carla-perception-checkpoints).

### Build and start

From the repository root:

```bash
docker compose up -d --build
docker compose exec avsectester avsectester version
```

The first build downloads prebuilt wheels and compiles no CUDA code. Compose tags the image as
`avsectester:latest`. The `avsectester` service starts an idle shell. Run commands in it with
`docker compose exec`:

```bash
docker compose exec avsectester bash
```

Stop both containers with `docker compose down`. To build the image without Compose, run
`docker build -t avsectester .`.

Continue with [Run a CARLA comparison](quickstart.md#run-a-carla-comparison).

### Devices, mounts and rebuilds

- **GPU numbering.** The AVSecTester container sees only physical GPU 1, as CUDA device 0.
  `configs/carla_scenario.yaml` sets perception `gpu: 0`, which therefore selects physical GPU 1
  in the container. On a host without Docker, `--gpu 1` selects the same physical device.
- **Weights.** `./models` is bind-mounted at `/app/models`, so the image shares the host's
  weights. At container start, the entrypoint reruns `fetch_models.sh` silently to recreate the
  detector links inside the image. The script skips files that already exist.
- **Code and outputs.** No other directory is mounted. The container runs the copy of the
  checkout made at build time. Files written in the container stay there until you copy them out,
  as in [Save a plot](quickstart.md#save-a-plot).
- **Rebuilds.** Pulling Git changes or editing code does not update an existing image. Rebuilding
  the image alone does not update a running container. After changing code, the Dockerfile or
  dependencies, rebuild the image and recreate the AVSecTester service. Recreating the container
  removes files stored only in its writable layer. First copy any experiment outputs you want to
  keep to the host, as in [Save a plot](quickstart.md#save-a-plot).

  With CARLA already running, update the AVSecTester service:

  ```bash
  docker compose up -d --build --no-deps avsectester
  ```

  The `--no-deps` option prevents Compose from starting dependency services, leaving the existing
  CARLA service running.

  To test local changes without rebuilding, mount the checkout as in
  [Tests and CI](../development/contributing.md#tests-and-ci).

## Local core installation

The package requires Python 3.11 (`requires-python = ">=3.11,<3.12"`). Use a stable 3.11
release. Avoid prerelease interpreters, such as the `python3.11` package in Ubuntu 22.04, which is
a release candidate that can fail during TorchScript initialization. The Docker image uses 3.11.13.
To install the same interpreter with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv python install 3.11.13
uv venv --python 3.11.13 --seed .venv
source .venv/bin/activate
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,test]" -c constraints.txt
```

Without uv, create the environment with Conda (`conda create -n avsec311 python=3.11.13 pip`,
then `conda activate avsec311`) or with an installed stable interpreter
(`python3.11 -m venv .venv`). Then run the same `pip` commands. Check `python --version` in the
active environment.

The core dependencies are PyYAML, NumPy, Typer and PyArrow. The exact PyArrow version is pinned in
`pyproject.toml` and installed by the package commands above; it is imported when opening a
persistent run log, so importing the driving interfaces does not load Arrow.
`constraints.txt` keeps NumPy below 2. The CPU
build of PyTorch is installed separately for the offline tests, so that no CUDA packages are
downloaded. Optional extras add the following:

| Extra | Adds | Used by |
|---|---|---|
| `rendering` | OpenCV (headless), Pillow | Insertion rendering, compositing and geometric visibility |
| `viz` | matplotlib | The driving-impact plot (`plot_impact`, `avsectester run --plot`) |
| `augment` | Albumentations 1.x | The Albumentations backend for [image corruptions](../guides/conditions.md#built-in-image-corruptions) |
| `test` | pytest, pytest-cov, Pillow, matplotlib, OpenCV, Albumentations 1.4.24 | The offline tests |
| `dev` | pytest, pytest-cov, Ruff, mypy | Development tools |

For example, to add insertion rendering:

```bash
pip install -e ".[rendering]" -c constraints.txt
```

Simulator connections, neural perception and learned harmonizers need the dependencies of the
other installations.

## Local full installation

The local full installation reproduces the Docker image on the host. It uses this dependency set:
Python 3.11, PyTorch 2.1.0 for CUDA 12.1, torchvision 0.16.0, NumPy 1.26.4, MMCV 2.1.0,
MMDetection 3.2.0, MMDetection3D 1.4.0, Albumentations 1.4 and the CARLA 0.9.16 client. The MMCV
wheel matches PyTorch 2.1 and CUDA 12.1 and satisfies the detector packages' version caps. Other
combinations may need different package versions or rebuilt CUDA operators.

Initialize the submodules as in [Clone the repository](#clone-the-repository). Create a fresh
Python 3.11 environment as in [Local core installation](#local-core-installation), but do not
install the CPU build of PyTorch. pip treats an installed `torch 2.1.0+cpu` as satisfying
`torch==2.1.0` and would keep it. Install the packages in this order:

```bash
# 1. PyTorch and torchvision for CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
# 2. Prebuilt MMCV for PyTorch 2.1.0 and CUDA 12.1, and MMEngine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
pip install "mmengine>=0.10"
# 3. Detector packages and their dependencies, keeping NumPy below 2
pip install --no-deps mmdet==3.2.0 mmdet3d==1.4.0
pip install "numpy==1.26.4" "scikit-image>=0.21,<0.24" "pandas>=2,<3" \
  pycocotools terminaltables plyfile trimesh tensorboard networkx
# 4. avstack packages and the CARLA client
pip install -e third_party/avstack-core -c constraints.txt
pip install -e third_party/avstack-api --no-deps
pip install -e third_party/lib-avstack-carla --no-deps
pip install carla==0.9.16 pygame ipywidgets nuscenes-devkit lyft-dataset-sdk==0.0.8 -c constraints.txt
# 5. AVSecTester
pip install -e ".[dev,rendering,viz,augment]" "setuptools==69.5.1" -c constraints.txt
```

- The avstack adapters are installed with `--no-deps` because their metadata declares
  sibling-path dependencies and dataset packages that this path does not need. The earlier steps
  install the dependencies it does need.
- MMDetection3D's dataset-module registration imports `lyft-dataset-sdk`, even for CARLA-only
  inference. The package downloads no data.
- PyTorch 2.1 and MMEngine use `pkg_resources`, which newer setuptools releases remove. Step 5
  pins setuptools 69.5.1, as the Dockerfile does.
- `requirements.lock` is not the installation specification for this environment. Use the
  commands above, `pyproject.toml` and `constraints.txt`.

### Start a CARLA server

The CARLA client and server must both be version 0.9.16. To run the server in Docker on physical
GPU 0:

```bash
docker pull carlasim/carla:0.9.16
docker run -d --name carla-avsec --gpus 'device=0' --net=host \
  carlasim/carla:0.9.16 ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -quality-level=Epic
```

Each run reloads the server's world, so use a server dedicated to your experiments. Run neural
inference on another device with `avsectester run ... --gpu 1`. The supplied configuration runs
CARLA in synchronous mode. See [Scene configuration](../guides/simulators.md#scene-configuration).
Stop the server with `docker rm -f carla-avsec`.

## NuRec and Alpamayo environment

The NuRec path with the Alpamayo driving system uses two parts of NVIDIA
[AlpaSim](https://github.com/NVlabs/alpasim): `alpasim_driver`, which provides the Alpamayo model in
its own Python environment, and the `nre-ga` rendering service, which serves reconstructed scenes
over gRPC from its own container. AVSecTester runs from its checkout with the driver
environment's interpreter.

### Driver environment

1. Clone AlpaSim and build its driver environment with uv, following AlpaSim's instructions
   (`uv sync --package alpasim_driver` in the AlpaSim checkout). The environment uses Python 3.12
   and provides `alpasim_driver`, the Alpamayo model packages and the `alpasim_grpc` client
   stubs that `NuRecRenderer` imports.

   Install AVSecTester's Parquet dependency into that interpreter. Run this from the AVSecTester
   checkout; the command reads the same exact pin used by Docker and the Python 3.11 installations:

   ```bash
   cd /path/to/AVSecTester
   arrow_requirement=$(/path/to/alpasim/.venv/bin/python -c \
     'import tomllib; print(next(d for d in tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"] if d.startswith("pyarrow==")))')
   uv pip install --python /path/to/alpasim/.venv/bin/python "$arrow_requirement"
   /path/to/alpasim/.venv/bin/python -c 'import pyarrow; import pyarrow.parquet; print(pyarrow.__version__)'
   ```

   Repeat this step after `uv sync`, which may remove dependencies outside AlpaSim's lockfile.
   This installs PyArrow without installing AVSecTester's Python 3.11 package or changing the
   driver's model dependencies.

2. Download the Alpamayo weights into a directory of your choice:

   ```bash
   hf download nvidia/Alpamayo-1.5-10B --local-dir /path/to/Alpamayo-1.5-10B
   ```

3. Run the NuRec demo from the repository root with the driver environment's interpreter.
   `--stub` replaces NuRec rendering with black frames, so no rendering service is needed. It
   still loads the real Alpamayo model:

   ```bash
   cd /path/to/AVSecTester
   /path/to/alpasim/.venv/bin/python -m scripts.demos.nurec.alpamayo_nurec_demo 8 --stub --gpu 1 \
     --checkpoint /path/to/Alpamayo-1.5-10B
   ```

   Without `--stub`, the demo renders through the service at `--endpoint` (default
   `127.0.0.1:50051`) and loads the scene named by `--scene` (default `01d503d4`). Add
   `--save-frames` to save each rendered frame under `tmp/alpamayo_nurec/` in the repository.

The checkpoint location is set as described in [Alpamayo checkpoint](#alpamayo-checkpoint).

### NuRec renderer and scenes

Obtain access to the NuRec dataset on Hugging Face and authenticate before downloading a scene.
Replace `<uuid>` and the local directory:

```bash
hf download nvidia/PhysicalAI-Autonomous-Vehicles-NuRec \
  --repo-type dataset --revision 26.01 "sample_set/26.01_release/<uuid>/<uuid>.usdz" \
  --local-dir /path/to/nurec-scenes
```

Serve the scenes on port 50051. This command uses physical GPU 0 and the server entrypoint:

```bash
docker run -d --name nre --net=host --gpus 'device=0' -e HOME=/tmp \
  -v /path/to/nurec-scenes/sample_set/26.01_release:/mnt/nre-data \
  --entrypoint /app/internal/scripts/pycena/runtime/pycena_nrm_full \
  nvcr.io/nvidia/nre/nre-ga:26.04 serve-grpc --host=0.0.0.0 --port=50051 \
  '--artifact-glob=/mnt/nre-data/**/*.usdz' --cache-size=2 --enable-editing-actors
```

`NuRecRenderer(scene_id=...)` accepts the service's full scene identifier, or a substring that
matches exactly one served scene, such as part of the scene UUID. These examples use GPU 0 for the
rendering service and GPU 1 for Alpamayo. CARLA and NuRec experiments do not need both servers
running at the same time. Renderer and backend options are in [NuRec](../guides/simulators.md#nurec).

### Why NuRec has no CLI path

The `avsectester` command is a console script that pip installs with the package. The package
declares `requires-python = ">=3.11,<3.12"`, matching avstack-core. `alpasim_driver` requires
Python 3.12. No interpreter satisfies both, so pip refuses to install the package, and with it the
`avsectester` command, in the driver environment. The avstack environment cannot load Alpamayo for
the same reason.

The driver environment therefore runs AVSecTester from the repository checkout. `python -m`
started from the repository root puts the checkout on the import path, so `avsectester` and
`scripts` import without installation. The supplied entry points are the NuRec demo scripts,
run as `python -m scripts.demos.nurec.<module>`. See
[Demo scripts and outputs](../examples/image-insertion.md#demo-scripts-and-outputs). Your own Python code can call
the same APIs from the checkout, for example `run_configured` with a NuRec backend mapping, as shown
in [Backend configuration](../guides/simulators.md#backend-configuration).

## Model weights and assets

Install only the models your experiment uses. Their roles are described in
[Models used in this repository](../guides/driving-systems.md#models-used-in-this-repository). No
weights are bundled with the Python package.

| Model or asset | Python dependencies or service | How it is loaded |
|---|---|---|
| CARLA-trained PointPillars, Faster R-CNN and Cascade R-CNN | Docker or the local full installation | `scripts/preparation/fetch_models.sh` downloads them and links them where avstack looks |
| COCO Faster R-CNN | `torch` and `torchvision` | torchvision downloads the default weights on first use |
| PCTNet | `torch`, `torchvision`, `numpy`, `einops`, OpenCV, the `libcom` submodule and `huggingface_hub` | A local `PCTNet.pth`, an explicit `weights` path, or a Hugging Face download |
| SAM | `torch`, `transformers` with `SamModel` and `SamProcessor`, OpenCV | `from_pretrained` with a model ID or a local directory |
| Alpamayo-1.5-10B | The AlpaSim driver environment | A checkpoint directory passed to `AlpamayoAVStack` |
| NuRec scenes | The `nre-ga` service | The service loads USDZ scene archives. There is no driving-model checkpoint |

`fetch_models.sh` downloads only the CARLA perception checkpoints and a stock KITTI PointPillars
checkpoint. Keep `torch` and `torchvision` compatible within each environment. The versions in the
local full installation apply to the avstack environment, not to the AlpaSim driver environment.

### CARLA perception checkpoints

Initialize the nested detector checkouts as in [Clone the repository](#clone-the-repository). Then,
from the repository root:

```bash
./scripts/preparation/fetch_models.sh
```

The script uses `curl`. It stores CARLA-trained PointPillars files under `models/work_dirs/`,
CARLA-trained 2D detector files under `models/work_dirs_2d/` and the stock checkpoint under
`models/checkpoints/`. It links these directories into the vendored MMDetection3D and MMDetection
roots in `third_party/avstack-core/third_party/`. The 2D and 3D directories stay separate because
avstack resolves configuration paths through them. The script skips files that already exist, so
it is safe to run again.

The supplied CARLA configuration uses PointPillars with `dataset: carla-vehicle`. The camera patch
scripts request Faster R-CNN with the same dataset.

The CARLA-trained PointPillars configuration needs the custom `CarlaDataset` registry entry for
inference metadata. No CARLA training data is needed. `ModularAVStack` registers the entry before
it builds a non-deployed `MMDetObjectDetector3D` with a `carla-*` dataset. When you build these
models outside `ModularAVStack`, register it first:

```python
from avsectester.stacks.mmdet3d import register_carla_dataset

register_carla_dataset()
```

Registration applies to the current Python process and keeps an existing registration. If the
installed MMDetection3D lacks the declaration, the helper loads it from the nested
`mmdetection3d` checkout and raises an error when that checkout is missing.

### COCO detector

The NuRec and nuScenes object demos use this detector with `--eval` to score inserted objects.
`build_coco_detector` in `scripts/common/demo_common.py` constructs torchvision's
`fasterrcnn_resnet50_fpn` with `FasterRCNN_ResNet50_FPN_Weights.DEFAULT`. torchvision downloads
missing weights into its Torch Hub cache on first use. Set `TORCH_HOME` before starting the script
to choose the cache directory. Offline runs need a populated cache. The scripts have no
checkpoint-path option for this detector. No COCO images are needed. The detector runs on the
images that the calling script supplies.

### PCTNet harmonizer

Initialize the source submodule and install the extra dependencies:

```bash
git -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init --depth 1 -- third_party/libcom
pip install einops huggingface-hub -c constraints.txt
```

`PCTNetHarmonizer` loads the harmonization modules directly from the checkout. Installing the
`libcom` package is not needed. Without an explicit weight path, it uses
`third_party/libcom/libcom/image_harmonization/pretrained_models/PCTNet.pth`. If that file is
missing, it downloads `PCTNet.pth` from the `BCMIZB/Libcom_pretrained_models` Hugging Face
repository into that directory. To use an existing checkpoint:

<!-- runnable -->
```python
from avsectester.rendering.compositor import PatchCompositor
from avsectester.rendering.harmonizers import PCTNetHarmonizer

harmonizer = PCTNetHarmonizer(device=0, weights="/path/to/PCTNet.pth", strict=True)
compositor = PatchCompositor(harmonizer)
```

The model loads at the first harmonization call. With `strict=True`, loading and inference errors
are raised. With the default `strict=False`, the harmonizer falls back to classic harmonization on
errors. Harmonization usage is described in [Harmonization](../examples/image-insertion.md#harmonization).

### SAM person cutouts

`scripts/preparation/extract_person_cutouts.py` segments annotated nuScenes pedestrians into
RGBA assets with SAM. Its `--sam` option defaults to `facebook/sam-vit-huge`, which
`from_pretrained` loads through the Hugging Face cache. Set `HF_HOME` to choose the cache
directory, or pass `--sam /path/to/sam-model` for a local directory that contains both the model
and processor files. SAM is needed only to produce the assets. The extraction command is in
[Demo scripts and outputs](../examples/image-insertion.md#demo-scripts-and-outputs).

### Alpamayo checkpoint

Download the weights as in [Driver environment](#driver-environment), then pass the directory in
one of three ways:

- `AlpamayoAVStack(checkpoint_path="/path/to/Alpamayo-1.5-10B")` in Python.
- `--checkpoint /path/to/Alpamayo-1.5-10B` for the Alpamayo demo scripts.
- The `AVSECTESTER_ALPAMAYO_CHECKPOINT` environment variable.

An explicit `checkpoint_path` takes precedence over the environment variable. The model loads
lazily. A missing location raises an error before the driver package is imported. For example:

<!-- runnable -->
```python
# Run in the AlpaSim driver environment from the repository root.
from avsectester.stacks.alpamayo import AlpamayoAVStack

stack = AlpamayoAVStack(checkpoint_path="/path/to/Alpamayo-1.5-10B", device="cuda:1")
```

Model loading and sharing are described in
[Loading and sharing the model](../guides/driving-systems.md#loading-and-sharing-the-model).

## Verify the installation

| Installation | Command | Expected result |
|---|---|---|
| Docker | `docker compose exec avsectester avsectester version` | `avsectester <version>` |
| Local core | `NO_ALBUMENTATIONS_UPDATE=1 python -m pytest tests/core -q` | All tests pass |
| Local full | `NO_ALBUMENTATIONS_UPDATE=1 python -m pytest -q` | The `core` and `avstack` tests pass |
| NuRec and Alpamayo | The `alpamayo_nurec_demo` command in [Driver environment](#driver-environment) | One speed line per driving step, then `END-TO-END OK: ...` |

`NO_ALBUMENTATIONS_UPDATE=1` stops Albumentations from checking for updates. Missing offline test
dependencies make the tests fail rather than skip. The test groups are described in
[tests/README.md](../../tests/README.md). In Docker, run the offline tests as in
[Tests and CI](../development/contributing.md#tests-and-ci).

To verify a complete CARLA run, follow [Run a CARLA comparison](quickstart.md#run-a-carla-comparison).

## Next steps

- [Quickstart](quickstart.md): run a clean and attacked CARLA comparison, and an offline drive.
- [Architecture](../concepts/architecture.md): the components and terms used in every guide.
- [Run experiments](../guides/experiments.md): choose an entry point for your own experiments.
