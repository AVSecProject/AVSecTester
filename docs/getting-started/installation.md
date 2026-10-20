# Installation

> **Audience:** Users setting up an experiment environment.
>
> **Outcome:** Install matching dependencies, services and model weights.
>
> **Prerequisites:** Linux; NVIDIA GPUs for native simulation and neural inference.

## Choose an installation

| Environment | Use |
|---|---|
| Docker | CARLA + ModularAVStack, CLI and offline checks |
| Local core | Interfaces, stub simulation, CPU tests and log analysis |
| Local full | Native CARLA + modular perception on the host |
| NuRec and Alpamayo | AlpaSim driver interpreter with rendering/model dependencies |

Weights and scene archives are separate downloads. Installing a package does not supply them.

## Clone the repository

```bash
git clone https://github.com/AVSecProject/AVSecTester.git
cd AVSecTester
```

For Docker or local full, initialize the required avstack and detector submodules:

```bash
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --depth 1 -- \
  third_party/avstack-core third_party/avstack-api third_party/lib-avstack-carla
git -C third_party/avstack-core -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init --depth 1 -- third_party/mmdetection third_party/mmdetection3d
```

The per-command HTTPS rewrite avoids needing GitHub SSH credentials and changes no global
configuration. The local core installation needs no submodules.

## Docker

Dockerfile builds the CARLA experiment environment. Compose runs it alongside CARLA 0.9.16. The
image includes Python 3.11.13, PyTorch 2.1.0/CUDA 12.1, torchvision 0.16.0, MMCV 2.1.0, MMDetection
3.2.0, MMDetection3D 1.4.0, the avstack packages, CARLA client and pinned PyArrow. It does not
contain the AlpaSim driver or Alpamayo model environment.

### Prerequisites

- NVIDIA GPUs, Docker Compose and NVIDIA runtime configured as Docker's default runtime.
- Adjust both NVIDIA_VISIBLE_DEVICES entries in docker-compose.yml for your physical devices.
  The supplied values select GPU 2 for CARLA and GPU 1 for AVSecTester.
- Free host ports: Compose uses host networking, CARLA RPC 2000 and traffic manager 8000.
- Initialized avstack/detector submodules and downloaded perception checkpoints.

```bash
./scripts/preparation/fetch_models.sh
```

### Build and start

```bash
docker compose up -d --build
docker compose exec avsectester avsectester version
docker compose exec avsectester avsectester run configs/carla_phantom.yaml
```

The service image is avsectester:latest. Stop both services with docker compose down. Model
inference uses the GPU number visible inside the container, configured in avstack options. Compose
uses Epic rendering quality and 2 GB of shared memory for the CARLA server. Epic is the quality
setting used for the verified Town03 experiment runs.

### Devices, mounts and rebuilds

Only models/ is bind-mounted by default. Code is copied at build time, and experiment outputs remain
in the container's writable layer. Copy needed results to the host before recreation:

```bash
docker compose cp avsectester:/app/results/carla_phantom ./results/carla_phantom
```

After editing code or dependencies, rebuild and recreate the service. Building the image alone does
not update the running container:

```bash
docker compose up -d --build --no-deps avsectester
```

The no-deps option leaves the existing CARLA service running. For development checks, mount the
checkout as shown in [Contributing](../development/contributing.md#tests-and-ci).

Compose's entrypoint runs fetch_models.sh to expose the model directories at the paths used by
avstack. A standalone container that mounts a checkout and overrides the entrypoint bypasses that
step. For PointPillars inference, explicitly mount the existing LiDAR configuration and checkpoint
tree read-only at its detector lookup path:

```bash
docker run --rm --gpus 'device=1' --net=host \
  -v "$PWD":/app -w /app \
  -v "$PWD/models/work_dirs":/app/third_party/avstack-core/third_party/mmdetection3d/work_dirs:ro \
  --entrypoint python avsectester:latest \
  -m avsectester.cli run configs/carla_phantom.yaml
```

This uses the files already downloaded under models/work_dirs; it does not download weights. Keep
the 2D model tree models/work_dirs_2d separate: the detector loader probes MMDetection before
MMDetection3D, so sharing one work_dirs tree can select the wrong loader. fetch_models.sh leaves an
existing non-symlink directory unchanged; a read-only mount also handles that case without changing
the checkout's directories or model links.

## Local core installation

Package installation targets Python 3.11 (`requires-python = ">=3.11,<3.12"`). Use a stable
interpreter, such as the Docker image's 3.11.13:

```bash
uv python install 3.11.13
uv venv --python 3.11.13 --seed .venv
source .venv/bin/activate
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,test]" -c constraints.txt
```

The package dependencies include PyYAML, NumPy, Typer, matplotlib, Pillow and pinned PyArrow.
PyArrow loads when persistent logs are opened. CPU torch is installed separately for tests; it is
not required just to import the plain interfaces. Constraints keep NumPy below 2.

| Extra | Purpose |
|---|---|
| avstack | Native avstack packages from the configured source submodules |
| rendering | OpenCV and Pillow for attack rendering |
| augment | Albumentations sensor transforms |
| test | CPU test image, plot and augmentation dependencies |
| dev | pytest, Ruff and mypy |

## Local full installation

Initialize the submodules and create Python 3.11 without installing CPU torch. Install the same
compatible versions as the Docker environment:

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
pip install "mmengine>=0.10"
pip install --no-deps mmdet==3.2.0 mmdet3d==1.4.0
pip install "numpy==1.26.4" "scikit-image>=0.21,<0.24" "pandas>=2,<3" \
  pycocotools terminaltables plyfile trimesh tensorboard networkx
pip install -e third_party/avstack-core -c constraints.txt
pip install -e third_party/avstack-api --no-deps
pip install -e third_party/lib-avstack-carla --no-deps
pip install carla==0.9.16 pygame ipywidgets nuscenes-devkit lyft-dataset-sdk==0.0.8 -c constraints.txt
pip install -e ".[dev,rendering,augment]" "setuptools==69.5.1" -c constraints.txt
```

Adapter metadata has sibling dependencies; no-deps avoids resolving them after the necessary imports
are installed. MMDetection3D imports Lyft dataset registration even for CARLA inference. The
setuptools pin retains pkg_resources for this native dependency set.

### Start a CARLA server

Use matching CARLA 0.9.16 client and server versions. This example dedicates physical GPU 0:

```bash
docker run -d --name carla-avsec --gpus 'device=0' --net=host \
  carlasim/carla:0.9.16 ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -quality-level=Epic
```

Every formal run can reload the world. Use a dedicated server and configure perception on another
available device in avstack options. Stop this server with docker rm -f carla-avsec.

## NuRec and Alpamayo environment

NuRec experiments combine the AlpaSim alpasim_driver Python 3.12 environment and the nre-ga
rendering service. They do not use the CARLA Python 3.11 inference environment.

### Driver environment

Build AlpaSim's driver environment following its instructions (`uv sync --package alpasim_driver`).
It must provide Alpamayo, alpasim_driver and alpasim_grpc. Install AVSecTester's exact PyArrow pin
into that interpreter:

```bash
cd /path/to/AVSecTester
arrow_requirement=$(/path/to/alpasim/.venv/bin/python -c \
  'import tomllib; print(next(d for d in tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"] if d.startswith("pyarrow==")))')
uv pip install --python /path/to/alpasim/.venv/bin/python "$arrow_requirement"
```

Repeat after uv sync if it removes dependencies outside AlpaSim's lockfile. The package metadata
targets Python 3.11, so use the checkout from the driver's interpreter rather than installing the
full CARLA package into Python 3.12. Install the CLI, report and insertion-image dependencies into
that interpreter, retaining its existing model and numerical-library versions:

```bash
uv pip install --python /path/to/alpasim/.venv/bin/python \
  "pyyaml>=6,<7" "typer>=0.15,<0.17" "matplotlib>=3.6" "Pillow>=10" \
  "opencv-python-headless>=4.9,<4.12" "$arrow_requirement"
```

Do not apply the CARLA Python 3.11 constraints or replace the driver's Torch installation. Then run
the same CLI module:

```bash
cd /path/to/AVSecTester
/path/to/alpasim/.venv/bin/python -m avsectester.cli run configs/nurec_object_insertion.yaml
```

Set dataset paths, renderer endpoint, checkpoint_path and device in the configuration first. The
avsectester:nurec image can supply an already prepared driver environment where available; use its
driver interpreter with the checkout mounted.

The repository's Dockerfile supplies this environment as its nurec target:

```bash
docker build --target nurec -t avsectester:nurec .
```

That target installs the framework's CLI, report and Parquet dependencies into AlpaSim's Python 3.12
environment while preserving AlpaSim's NumPy version. It imports AVSecTester from the mounted
checkout rather than installing the Python 3.11 package. An older image built with only the
additional PyArrow dependency needs rebuilding or the dependency installation above before it can
run the CLI.

### NuRec renderer and scenes

Authenticate and obtain dataset access before downloading the selected USDZ scene:

```bash
hf download nvidia/PhysicalAI-Autonomous-Vehicles-NuRec \
  --repo-type dataset --revision 26.01 "sample_set/26.01_release/<uuid>/<uuid>.usdz" \
  --local-dir /path/to/nurec-scenes
```

Serve it with nre-ga on a separate available GPU:

```bash
docker run -d --name nre --net=host --gpus 'device=0' -e HOME=/tmp \
  -v /path/to/nurec-scenes/sample_set/26.01_release:/mnt/nre-data \
  --entrypoint /app/internal/scripts/pycena/runtime/pycena_nrm_full \
  nvcr.io/nvidia/nre/nre-ga:26.04 serve-grpc --host=0.0.0.0 --port=50051 \
  '--artifact-glob=/mnt/nre-data/**/*.usdz' --cache-size=2 --enable-editing-actors
```

NuRecRenderer accepts a served scene id or an unambiguous substring. NuRecDataset.scene_id(path)
returns its native clip id. Set dataset time_step and simulator dt consistently.

## Model weights and assets

### CARLA perception checkpoints

```bash
./scripts/preparation/fetch_models.sh
```

The script stores checkpoints under models/work_dirs/, models/work_dirs_2d/ and models/checkpoints/,
and links them into the vendored detector checkouts. Existing files are reused. No training data is
required for model metadata registration. ModularAVStack registers CARLA metadata; external model
construction can call closed_loop.avstacks.mmdet3d.register_carla_dataset explicitly.

### Alpamayo checkpoint

```bash
hf download nvidia/Alpamayo-1.5-10B --local-dir /path/to/Alpamayo-1.5-10B
```

Pass checkpoint_path in avstack options or set AVSECTESTER_ALPAMAYO_CHECKPOINT. An explicit path
wins. Model weights load lazily. Set device to the GPU visible in the driver environment.

### PCTNet harmonizer

```bash
git -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init --depth 1 -- third_party/libcom
pip install einops huggingface-hub -c constraints.txt
```

PCTNetHarmonizer loads from the libcom checkout. Supply weights explicitly for reproducible loading.
Without an explicit path, it uses libcom's pretrained_models/PCTNet.pth and can download missing
weights from BCMIZB/Libcom_pretrained_models.

<!-- runnable -->
```python
from avsectester.attack.insertion.compositor import PatchCompositor
from avsectester.attack.insertion.harmonizers import PCTNetHarmonizer

harmonizer = PCTNetHarmonizer(device=0, weights="/path/to/PCTNet.pth", strict=True)
compositor = PatchCompositor(harmonizer)
```

Loading occurs at the first harmonization call. strict=True raises load/inference errors; non-strict
mode may fall back to classic harmonization.

### SAM person cutouts

scripts/preparation/extract_person_cutouts.py uses SAM to produce RGBA assets. Its sam option
accepts a model id or local model/processor directory. Inspect its help for dataset and output
arguments. SAM is needed for asset preparation, not for deploying existing cutouts.

## Verify the installation

```bash
avsectester version
NO_ALBUMENTATIONS_UPDATE=1 python -m pytest tests/core -q
```

The core suite needs CPU test dependencies. Full offline checks add tests/avstack. Native
integration checks require the configured services and weights; a stub run does not verify them. See
[Tests](../../tests/README.md).

## Next steps

Follow [Quickstart](quickstart.md), [Experiments](../guides/experiments.md) and
[Configuration](../reference/configuration.md).
