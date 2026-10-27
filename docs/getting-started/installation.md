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
| NuRec and Alpamayo | Compose experiment service with Python 3.12 and automatic local services |

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
- Run scripts/preparation/write_compose_env.sh once, then set CARLA_SERVER_GPU,
  CARLA_EXPERIMENT_GPU and NUREC_GPU in .env. All default to GPU 0.
- Free host ports: Compose uses host networking, CARLA RPC 2000 and traffic manager 8000.
- Initialized avstack/detector submodules and downloaded perception checkpoints.

```bash
./scripts/preparation/fetch_models.sh
```

### Build and start

```bash
scripts/preparation/write_compose_env.sh
docker compose up -d --build
docker compose exec carla-experiment avsectester version
docker compose exec carla-experiment avsectester run configs/carla_phantom.yaml
```

The carla-server service uses carlasim/carla:0.9.16; carla-experiment uses avsectester:carla.
A standalone build is `docker build --target carla -t avsectester:carla .`. Stop both services with docker compose down. Model
inference uses the GPU number visible inside the container, configured in avstack options. Compose
uses Low rendering quality and 2 GB of shared memory for the CARLA server.

### Export CARLA maps

Semantic-map reads use local OpenDRIVE files. With CARLA running, export the towns needed by your
recipes from an environment containing the carla client package:

```bash
docker compose exec carla-experiment python -m scripts.preparation.export_carla_maps \
  --host 127.0.0.1 --port 2000 --town Town03 Town05 --output data/carla_maps
```

The script connects once and loads each requested town to write `<town>.xodr`. Set
dataset.options.map_directory to ../data/carla_maps in a configuration under configs/; paths remain
relative to that configuration file. Copy exports to the host if they must survive container
recreation. Later map reads use carla.Map locally and require no server connection. Missing files,
map_directory or the optional client package produce Unavailable with an export instruction rather
than starting a simulator. See [Semantic maps](../guides/scenarios.md#semantic-maps).

### Devices, mounts and rebuilds

Only models/ is bind-mounted by default. Code is copied at build time, and experiment outputs remain
in the container's writable layer. Copy needed results to the host before recreation:

```bash
docker compose cp carla-experiment:/app/results/carla_phantom ./results/carla_phantom
```

After editing code or dependencies, rebuild and recreate the service. Building the image alone does
not update the running container:

```bash
docker compose up -d --build --no-deps carla-experiment
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
  --entrypoint python avsectester:carla \
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
pip install -e ".[dev,test,nurec]" -c constraints.txt
```

The package dependencies include PyYAML, NumPy, Typer, matplotlib, Pillow and pinned PyArrow.
PyArrow loads when persistent logs are opened. CPU torch is installed separately for tests; it is
not required just to import the plain interfaces. Constraints keep NumPy below 2.

| Extra | Purpose |
|---|---|
| avstack | Native avstack packages from the configured source submodules |
| rendering | OpenCV and Pillow for attack rendering |
| nurec | Pinned trimesh/Embree dependencies for NuRec mesh depth and its offline tests |
| augment | Albumentations sensor corruptions |
| test | CPU test image, plot and augmentation dependencies |
| dev | pytest, Ruff and mypy |

NuRec defaults to mesh depth and ego ground contact. Install the optional nurec extra when using
either feature:

```bash
pip install -e ".[nurec]" -c constraints.txt
```

It pins trimesh==5.1.1 and embreex==4.4.0.post1. trimesh loads the archived PLY meshes; embreex
provides CPU Embree ray queries and supplies wheels for Python 3.11 and 3.12.
No GPU is required for mesh intersection. Without this extra, explicitly set both depth="none"
and physics_update_mode="none"; either mesh feature rejects missing intersection dependencies at
construction. The AlpaSim controller has separate workspace dependencies described below.

## Local full installation

Initialize the submodules and create Python 3.11 without installing CPU torch. Install the same
compatible versions as the Docker environment. The full offline test suite requires both test and
nurec extras; dev supplies pytest, Ruff and mypy:

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
pip install "mmengine>=0.10"
pip install --no-deps mmdet==3.2.0 mmdet3d==1.4.0
pip install "numpy==1.26.4" "scikit-image>=0.21,<0.24" "pandas>=2,<3" \
  pycocotools terminaltables plyfile tensorboard networkx
pip install -e third_party/avstack-core -c constraints.txt
pip install -e third_party/avstack-api --no-deps
pip install -e third_party/lib-avstack-carla --no-deps
pip install carla==0.9.16 pygame ipywidgets nuscenes-devkit lyft-dataset-sdk==0.0.8 -c constraints.txt
pip install -e ".[dev,test,rendering,augment,nurec]" "setuptools==69.5.1" -c constraints.txt
```

Adapter metadata has sibling dependencies; no-deps avoids resolving them after the necessary imports
are installed. MMDetection3D imports Lyft dataset registration even for CARLA inference. The
setuptools pin retains pkg_resources for this native dependency set.

### Start a CARLA server

Use matching CARLA 0.9.16 client and server versions. This example dedicates physical GPU 0:

```bash
docker run -d --name carla-avsec --gpus 'device=0' --net=host \
  carlasim/carla:0.9.16 ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -quality-level=Low
```

Every formal run can reload the world. Use a dedicated server and configure perception on another
available device in avstack options. Stop this server with docker rm -f carla-avsec.

## NuRec and Alpamayo environment

The nurec-experiment Compose service uses avsectester:nurec, with AlpaSim's Python 3.12 driver,
controller, rendering and report dependencies. It imports the bind-mounted checkout through
PYTHONPATH instead of installing the Python 3.11 CARLA package. The nurec profile keeps it out
of ordinary `docker compose up`; an explicit `docker compose run nurec-experiment` enables it.

### One-time preparation

```bash
scripts/preparation/write_compose_env.sh
docker login nvcr.io
docker compose build nurec-experiment
```

The script writes the repository's absolute path, user/group IDs, Docker socket group, host Docker
executable and default GPU selections to .env. It refuses to overwrite existing settings unless
--force is supplied. Set NUREC_GPU in .env or override it per command. The experiment runs as that
user, with the Docker socket group added.

For data and models outside the repository, create docker-compose.override.yml with read-only
mounts at the same absolute host paths:

```yaml
services:
  nurec-experiment:
    volumes:
      - /srv/datasets:/srv/datasets:ro
      - /srv/models:/srv/models:ro
```

Then link the existing files into the repository paths used by configurations, for example:

```bash
mkdir -p data models
ln -s /srv/datasets/nurec/sample_set data/nurec
ln -s /srv/models/Alpamayo-1.5-10B models/Alpamayo-1.5-10B
ln -s /srv/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
```

Compose automatically merges this ignored override file. The repository, archives and probe files
must have identical absolute paths inside the experiment container and on the host: its Docker
commands launch sibling service containers using host paths. It mounts the Docker socket and host
Docker CLI for this purpose.

### Run

Set dataset paths, renderer endpoint/probe paths, weights and model device in the configuration:

```bash
NUREC_GPU=0 docker compose run --rm nurec-experiment run configs/nurec_stop_sign_patch_modular_2601.yaml
```

The run command checks the configured services, reuses compatible running instances, and starts
missing local NRE/trafficsim services before preparation. It stops and removes only containers it
started, including on errors and interruption. Started services use the experiment's GPU (NUREC_GPU)
unless --nre-gpu or --trafficsim-gpu selects another host GPU. YAML device indexes address devices
visible inside the experiment container.

### NuRec renderer and scenes

Obtain dataset access and place the selected USDZ archives under data/nurec or configure dataroot /
usdz_paths explicitly. NuRecDataset returns native clip identifiers. NuRec requires ScenarioStart;
the experiment supplies its recorded clock, initial ego pose, map and actors. Decision timing comes
from dataset.time_step.

NRE receives --enable-editing-actors. The simulator moves filtered controllable tracks below the
scene; inserted world objects are composited onto rendered images. See
[NuRec actor movement](../guides/simulators.md#move-recorded-actors).
The renderer's probe_directory is created before launch and mounted at server_probe_directory,
or at its same absolute path if no server path is set. This supports rendered-probe occlusion;
see [Rendered-probe occlusion](../guides/simulators.md#rendered-probe-occlusion).

For optional manual service reuse:

```bash
NUREC_GPU=0 docker compose run --rm nurec-experiment nre serve CONFIG.yaml --gpu 0
NUREC_GPU=0 docker compose run --rm nurec-experiment run CONFIG.yaml
NUREC_GPU=0 docker compose run --rm nurec-experiment nre stop CONFIG.yaml
```

A compatible existing service is left running after run. --dry-run on nre serve prints its command
without Docker or filesystem changes. See [NRE commands](../reference/cli.md#nre).

### Reactive traffic service

Build nurec before the optional CATK service, which extends that image:

```bash
docker compose build nurec-experiment
DOCKER_BUILDKIT=1 docker build --target trafficsim -t avsectester:trafficsim .
```

Download the four files from the pinned AlpaSim commit, preserving their subdirectories:

```bash
traffic_models_dir=/srv/models/alpasim-trafficsim
traffic_models_url=https://media.githubusercontent.com/media/NVlabs/alpasim/affc2eab209fa43bdfa2f26c0f8d437922d78a68/data/trafficsim-models
mkdir -p "$traffic_models_dir/catk_v120" "$traffic_models_dir/tokens"
for file in catk_v120/config.yaml catk_v120/latest.ckpt \
  tokens/cabs_tc_map_vocab_c2048_t1000.pkl tokens/cabs_tc_base_agent_vocab_c1024.pkl; do
  curl --fail --location "$traffic_models_url/$file" --output "$traffic_models_dir/$file"
done
```

Check SHA256 and byte count against the Git LFS pointers bundled in the image:

```bash
docker compose run --rm -T --entrypoint python nurec-experiment - <<'PYTHON'
from pathlib import Path
import hashlib

source = Path("/opt/alpasim/data/trafficsim-models")
for relative in (
    "catk_v120/config.yaml", "catk_v120/latest.ckpt",
    "tokens/cabs_tc_map_vocab_c2048_t1000.pkl",
    "tokens/cabs_tc_base_agent_vocab_c1024.pkl",
):
    fields = dict(line.split(" ", 1) for line in (source / relative).read_text().splitlines())
    data = (Path("/srv/models/alpasim-trafficsim") / relative).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert len(data) == int(fields["size"]), relative
    assert digest == fields["oid"].removeprefix("sha256:"), relative
    print(relative, len(data), digest)
PYTHON
```

Set NuRecSimulator.traffic in the configuration and supply the existing model directory:

```bash
NUREC_GPU=0 docker compose run --rm nurec-experiment run CONFIG.yaml \
  --trafficsim-models /srv/models/alpasim-trafficsim
```

Run automatically manages both services. --trafficsim-gpu can select another physical GPU.
Manual trafficsim serve/stop commands remain available through the same Compose service.


## Model weights and assets

### CARLA perception checkpoints

```bash
./scripts/preparation/fetch_models.sh
```

The script stores checkpoints under models/work_dirs/, models/work_dirs_2d/ and models/checkpoints/,
and links them into the vendored detector checkouts. Existing files are reused. No training data is
required for model metadata registration. ModularAVStack registers CARLA metadata; external model
construction can call closed_loop.avstacks.implemented.modular.register_carla_dataset explicitly.

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

PCTNetHarmonizer requires libcom_directory pointing to the libcom repository root. Supply weights
explicitly for reproducible loading; the framework does not infer the checkout path.
Without an explicit path, it first uses libcom's pretrained_models/PCTNet.pth. If missing, the
first harmonization call imports huggingface_hub lazily and downloads PCTNet.pth from
BCMIZB/Libcom_pretrained_models with hf_hub_download into that directory. Construction alone does
not load the model or fetch weights.

<!-- runnable -->
```python
from avsectester.attack.insertion.compositor import PatchCompositor
from avsectester.rendering.harmonizers import PCTNetHarmonizer

harmonizer = PCTNetHarmonizer(
    libcom_directory="/path/to/libcom", device="cuda:0", weights="/path/to/PCTNet.pth", strict=True,
)
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
