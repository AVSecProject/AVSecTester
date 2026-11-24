# Installation

> **Audience:** Users setting up an experiment environment.
>
> **Outcome:** Prepare Compose services, data and model weights.
>
> **Prerequisites:** Linux, NVIDIA GPUs, Docker Compose and NVIDIA Container Toolkit.

Configure NVIDIA Container Toolkit as Docker's default runtime.

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
```

## Repository and Compose settings

Clone the checkout and initialize the native driving-system submodules:

```bash
git clone https://github.com/AVSecProject/AVSecTester.git
cd AVSecTester
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --depth 1 -- \
  third_party/avstack-core third_party/avstack-api third_party/lib-avstack-carla
git -C third_party/avstack-core -c url.https://github.com/.insteadOf=git@github.com: \
  submodule update --init --depth 1 -- third_party/mmdetection third_party/mmdetection3d
scripts/preparation/write_compose_env.sh
```

The script writes `.env` with the repository path, user/group IDs, Docker socket group, Docker
executable and GPU selections. It refuses to overwrite an existing file unless --force is
supplied. Choose available host GPUs in `.env`: `CARLA_SERVER_GPU`, `CARLA_EXPERIMENT_GPU` and
`NUREC_GPU` each default to 0. `CARLA_SERVER_ARGS` defaults to an empty string and appends
arguments to the CARLA server command.

| Compose service | Image | Purpose |
|---|---|---|
| `carla-server` | carlasim/carla:1.0, pinned by digest in Compose | CARLA server |
| `carla-experiment` | avsectester:carla | CARLA driving-system environment |
| `nurec-experiment` | avsectester:nurec | NuRec, AlpaSim controller and Alpamayo environment |

The checkout is bind-mounted at its host absolute path. Experiments run as the configured user.
Files written below the checkout are available on the host. Rebuild images after changing
dependencies. Checkout code is read through `PYTHONPATH`.

### External data and models

For files outside the checkout, create an ignored `docker-compose.override.yml` that mounts the
same absolute directories into every service that reads those external data or model files. For
example, mount them into `nurec-experiment`:

```yaml
services:
  nurec-experiment:
    volumes:
      - /srv/av-data:/srv/av-data:ro
      - /srv/models:/srv/models:ro
```

Link data and weights into the repository, or configure their paths directly:

```bash
mkdir -p data models
ln -s /srv/av-data/nurec data/nurec
ln -s /srv/models/Alpamayo-1.5-10B models/Alpamayo-1.5-10B
```

NuRec service containers mount the resolved archive and probe paths from the host. These paths
must have the same absolute names on the host and in `nurec-experiment`. The service mounts the
Docker socket and CLI to start sibling containers.

## CARLA environment

### CARLA map packs

`Town10HD_Opt` is included in the server image. `Town03_Opt` requires the matching
[AdditionalMaps package](https://downloads.carlasim.com/Linux/AdditionalMaps_1.0.tar.gz).
Extract the package into a directory outside the checkout, then link it for Compose:

```bash
mkdir -p /srv/carla/maps
tar xzf AdditionalMaps_1.0.tar.gz -C /srv/carla/maps
mkdir -p data
ln -s /srv/carla/maps data/carla_content_packs
```

Validate the extracted pack against the pinned server image:

```bash
docker run --rm -e NVIDIA_VISIBLE_DEVICES=void \
  -v /srv/carla/maps:/packs:ro --entrypoint python3 \
  carlasim/carla:1.0@sha256:39681ed1eff895b6a813587285c51b687d7739ef8eeabcf51e38771d7b5b0263 \
  /workspace/Tools/carla_pack.py inspect /packs/AdditionalMaps --server /workspace
```

Compose mounts this directory read-only at `/packs` and passes `-carla-packs=/packs` to CARLA.
Content packs and exported semantic maps are separate: `data/carla_content_packs` contains
the mounted packs, while `data/carla_maps` contains exported OpenDRIVE files. See
[CARLA content packs](https://carla-ue5.readthedocs.io/en/latest/map_content_packs/).

### Build and start CARLA

Fetch the avstack CARLA and nuScenes perception checkpoints, then build the experiment image.

```bash
scripts/preparation/fetch_models.sh
docker compose build carla-experiment
docker compose up -d carla-server carla-experiment
docker compose exec carla-experiment avsectester version
```

The image provides the native avstack and detector dependencies. Model files live under
`models/checkpoints`, `models/work_dirs` and `models/work_dirs_2d`.
Dockerfile creates the corresponding links inside the image.
The Python client wheel comes from the same pinned CARLA image as the server.

CARLA uses host networking with RPC port 2000 and traffic-manager port 8000. Select the server
and experiment host GPUs through `.env`. Model device indexes address GPUs visible in the
experiment container. Stop the services with `docker compose down`.

### Native ROS 2

Choose an unused ROS domain ID. From the checkout, create the DDS network and profiles.

```bash
scripts/preparation/setup_carla_ros2_network.sh 94
```

Add the printed values to `.env`. They include `CARLA_SERVER_ARGS`, with
`-ros2 -rmw=fastdds -ros-domain-id=N`, the Fast DDS profile path and its mount setting.
Use the same domain in the external driving system. Start the ROS 2 services with the DDS overlay.

```bash
docker compose -f docker-compose.yml -f docker-compose.ros2.yml up -d carla-experiment
```

The network script starts an idle container to keep the user's bridge active. After stopping
the Compose services with the same overlays, check that your runs have ended. Verify that both
of the following resources have the `avsectester.ros2-user` label with your numeric user ID.

```bash
docker inspect --format '{{json .Config.Labels}}' "avsectester-ros2-keepalive-$(id -u)"
docker network inspect --format '{{json .Labels}}' "avsectester-ros2-$(id -u)"
```

Remove your idle container and network when you no longer need them.

```bash
docker rm -f "avsectester-ros2-keepalive-$(id -u)"
docker network rm "avsectester-ros2-$(id -u)"
```

Configure `ExternalAVStack` in the experiment. See
[native ROS 2 control](../guides/simulators.md#native-ros-2-control) for runtime behavior.

#### Hosted Autoware

Pull the pinned Autoware 1.9.0 image on the Docker host.

```bash
docker pull ghcr.io/autowarefoundation/autoware@sha256:1db2988caa5dc5bae41548a8d556d0b57f55eb4dd63685426d94fd9c7a786034
```

The official model installer is in the CARLA 1.0 integration. The CARLA image pinned in Compose
contains the package at `/workspace`. To obtain the installer on the host, copy the Autoware
integration directory from that image.

```bash
carla_image=carlasim/carla:1.0@sha256:39681ed1eff895b6a813587285c51b687d7739ef8eeabcf51e38771d7b5b0263
carla_container=$(docker create "$carla_image")
docker cp "$carla_container:/workspace/PythonAPI/examples/av_stacks/autoware" ./carla-autoware
docker rm "$carla_container"
(cd carla-autoware && ./install/install_autoware.sh --docker --distro jazzy --version 1.9.0)
```

Point `autoware_data_directory` at the host directory containing `ml_models`. Autoware writes
TensorRT caches there as root. The framework uses the locally available image.

Follow [Native ROS 2](#native-ros-2) to set up the network and domain. Extract the official
Autoware assets and ROS camera profile from the checkout.

```bash
scripts/preparation/extract_carla_autoware_assets.sh
```

Keep the assets, DDS profiles and per-run work directory under the checkout. The experiment
process and Docker host need matching paths. When starting the hosted services, add the Autoware
overlay, which mounts the official camera profile into the CARLA server.

```bash
docker compose -f docker-compose.yml -f docker-compose.ros2.yml -f docker-compose.autoware.yml \
  up -d carla-experiment
```

The supplied configuration uses single-step physics at the configured tick rate. Its framework
RGB camera captures and records once per second. Autoware uses the separate ROS sensor rig.
Follow [Hosted Autoware](../examples/hosted-autoware.md) to copy and run the configuration.

### CARLA semantic maps

Map export is optional. With `carla-server` running, export both configured maps and set
`dataset.options.map_directory: ../data/carla_maps`:

```bash
docker compose exec carla-experiment python -m scripts.preparation.export_carla_maps \
  --host 127.0.0.1 --port 2000 --town Town03_Opt Town10HD_Opt --output data/carla_maps
```

The files are written into the mounted checkout. Map use and unavailable-map behavior are
described in [Datasets](../guides/datasets.md).

## NuRec environment

Use an NGC account for NRE images. Obtain Hugging Face access to the selected NuRec dataset
and Alpamayo checkpoint. Build the NuRec image and authenticate to the NRE image registry.

```bash
docker login nvcr.io
docker pull nvcr.io/nvidia/nre/nre-ga:26.04
docker compose build nurec-experiment
docker compose run --rm nurec-experiment version
```

`nurec-experiment` uses the nurec profile, so ordinary docker compose up does not start it. An
explicit docker compose run enables it. The image provides the Python 3.12 AlpaSim workspace and
imports this checkout without installing the `avsectester` package, which requires Python 3.11.

Obtain access to the
[NuRec dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NuRec) and
place its 26.01 and 26.04 sample sets at `data/nurec/26.01_release` and
`data/nurec/26.04_release`. To download selected archives with the included Hugging Face CLI,
set `HF_TOKEN` and replace `ARCHIVE_PATH` with the archive's path in the dataset repository.

```bash
docker compose run --rm -e HF_TOKEN --entrypoint hf nurec-experiment \
  download nvidia/PhysicalAI-Autonomous-Vehicles-NuRec ARCHIVE_PATH --repo-type dataset \
  --local-dir data/nurec/26.01_release
```

Use the corresponding local release directory for each archive. Stop-sign configurations use
either release. The modular vehicle insertion configuration uses 26.04. The physical patch,
NuRec image insertion and speed-feedback configurations use scene
`023b7fcc-671c-40e3-9bd2-c66b0b073fbc` from the 26.01 release.
Place the nuScenes v1.0-trainval release under `data/nuscenes`, containing `samples/`,
`sweeps/`, `maps/` and `v1.0-trainval/`. The
[configuration data column](../examples/README.md#configurations) lists data and weights for
each workflow. `dataroot` or `usdz_paths` can select other locations.

The run command manages compatible NRE and trafficsim services automatically. See
[Service lifecycle](../reference/cli.md#automatic-local-services). For manual service reuse, see
[NRE commands](../reference/cli.md#nre).

### Alpamayo weights

Set `HF_TOKEN` in your shell to a Hugging Face token with checkpoint access, then download:

```bash
export HF_TOKEN=YOUR_HUGGING_FACE_TOKEN
docker compose run --rm -e HF_TOKEN --entrypoint hf nurec-experiment \
  download nvidia/Alpamayo-1.5-10B --local-dir models/Alpamayo-1.5-10B
```

The supplied configurations set `avstack.options.checkpoint_path`. For a configuration that
relies on `AVSECTESTER_ALPAMAYO_CHECKPOINT`, explicitly forward the variable to the Compose run:

```bash
export AVSECTESTER_ALPAMAYO_CHECKPOINT=/srv/models/Alpamayo-1.5-10B
docker compose run --rm -e AVSECTESTER_ALPAMAYO_CHECKPOINT nurec-experiment run CONFIG.yaml
```

The path must exist inside the experiment container.
[Driving systems](../guides/driving-systems.md) describes model loading.

### YOLOX detector weights

The nurec image includes YOLOX. The rule-based driving system uses the COCO YOLOX-m checkpoint:

```bash
mkdir -p models
curl --fail --location \
  https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth \
  --output models/yolox_m.pth
```

This is the local `weights_path` in the supplied modular stop-sign and vehicle insertion
configurations. The framework does not fetch this checkpoint during a run.

### Reactive traffic weights and image

Build `nurec-experiment` first, then the CATK image that extends it:

```bash
docker build --target trafficsim -t avsectester:trafficsim .
```

Download the four weights files into models/alpasim-trafficsim:

```bash
traffic_models_dir=models/alpasim-trafficsim
traffic_models_url=https://media.githubusercontent.com/media/NVlabs/alpasim/affc2eab209fa43bdfa2f26c0f8d437922d78a68/data/trafficsim-models
mkdir -p "$traffic_models_dir/catk_v120" "$traffic_models_dir/tokens"
for file in catk_v120/config.yaml catk_v120/latest.ckpt \
  tokens/cabs_tc_map_vocab_c2048_t1000.pkl tokens/cabs_tc_base_agent_vocab_c1024.pkl; do
  curl --fail --location "$traffic_models_url/$file" --output "$traffic_models_dir/$file"
done
```

Validate their byte count and SHA256 against the Git LFS pointers bundled in the NuRec image:

```bash
docker compose run --rm -T --entrypoint python nurec-experiment - <<'PYTHON'
from pathlib import Path
import hashlib

source = Path("/opt/alpasim/data/trafficsim-models")
weights = Path("models/alpasim-trafficsim")
for relative in (
    "catk_v120/config.yaml", "catk_v120/latest.ckpt",
    "tokens/cabs_tc_map_vocab_c2048_t1000.pkl",
    "tokens/cabs_tc_base_agent_vocab_c1024.pkl",
):
    pointer = dict(line.split(" ", 1) for line in (source / relative).read_text().splitlines())
    data = (weights / relative).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert len(data) == int(pointer["size"]), relative
    assert digest == pointer["oid"].removeprefix("sha256:"), relative
    print(relative, len(data), digest)
PYTHON
```

With traffic configured, supply the weights directory to run:

```bash
docker compose run --rm nurec-experiment run CONFIG.yaml \
  --trafficsim-models models/alpasim-trafficsim
```

See [Reactive traffic](../guides/simulators.md#reactive-traffic) for configuration and behavior.

## Optional assets

PCTNet harmonization requires `third_party/libcom`, `einops` and `huggingface-hub`. Only
`avsectester:nurec` includes the latter two dependencies. Initialize libcom with the same HTTPS
rewrite used for the other submodules:

```bash
git -c url.https://github.com/.insteadOf=git@github.com: submodule update --init --depth 1 -- third_party/libcom
```

Provide local `PCTNet.pth` weights from
[Libcom pretrained models](https://huggingface.co/BCMIZB/Libcom_pretrained_models) when
possible. Without a local checkpoint, the harmonizer downloads weights on first use. See the
[harmonizer API](../reference/python-api.md#avsectesterrenderingharmonizers) for
loading options.

SAM person-cutout preparation accepts a model ID or local model/processor directory. Run it in
`nurec-experiment`:

```bash
docker compose run --rm --entrypoint python nurec-experiment \
  -m scripts.preparation.extract_person_cutouts --help
```

Existing cutouts can be deployed without SAM.

## Verify the installation

Use the corresponding Compose service's version command, then follow the offline checks in
[Tests](../../tests/README.md). Simulator, renderer and model prerequisites are listed there.

## Next steps

Run [Quickstart](quickstart.md), then see [Experiments](../guides/experiments.md)
and [Configuration](../reference/configuration.md).
