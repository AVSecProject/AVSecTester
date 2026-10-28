# Installation

> **Audience:** Users setting up an experiment environment.
>
> **Outcome:** Prepare Compose services, data and model weights.
>
> **Prerequisites:** Linux, NVIDIA GPUs, Docker Compose and NVIDIA Container Toolkit.

Use an NGC account for NRE images and obtain Hugging Face access to the selected NuRec dataset
and Alpamayo checkpoint. Configure NVIDIA Container Toolkit as Docker's default runtime:

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
`NUREC_GPU` each default to 0.

| Compose service | Image | Purpose |
|---|---|---|
| `carla-server` | carlasim/carla:0.9.16 | CARLA server |
| `carla-experiment` | avsectester:carla | CARLA driving-system environment |
| `nurec-experiment` | avsectester:nurec | NuRec, AlpaSim controller and Alpamayo environment |

The checkout is bind-mounted at its host absolute path. Experiments run as the configured user;
files written below the checkout are available on the host. Rebuild images after changing
dependencies; checkout code is read through `PYTHONPATH`.

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

Fetch the avstack CARLA and nuScenes perception checkpoints and build the experiment image:

```bash
scripts/preparation/fetch_models.sh
docker compose build carla-experiment
docker compose up -d carla-server carla-experiment
docker compose exec carla-experiment avsectester version
```

The image provides the native avstack and detector dependencies. Model files live under
`models/checkpoints`, `models/work_dirs` and `models/work_dirs_2d`.
Dockerfile creates the corresponding links inside the image.

CARLA uses host networking with RPC port 2000 and traffic-manager port 8000. Select the server
and experiment host GPUs through `.env`; model device indexes address GPUs visible in the
experiment container. Stop the services with docker compose down.

### CARLA semantic maps

Map export is optional. With `carla-server` running, export Town03 and set the dataset's
`dataset.options.map_directory: ../data/carla_maps`:

```bash
docker compose exec carla-experiment python -m scripts.preparation.export_carla_maps \
  --host 127.0.0.1 --port 2000 --town Town03 --output data/carla_maps
```

The files are written into the mounted checkout. Map use and unavailable-map behavior are
described in [Datasets](../guides/datasets.md).

## NuRec environment

Build the NuRec image and authenticate to the NRE image registry:

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
place the single-scene example at
`data/nurec/023b7fcc-671c-40e3-9bd2-c66b0b073fbc/023b7fcc-671c-40e3-9bd2-c66b0b073fbc.usdz`.
Stop-sign configurations use `data/nurec/26.01_release` or `data/nurec/26.04_release`.
Place the nuScenes v1.0-trainval release under `data/nuscenes`, containing `samples/`,
`sweeps/`, `maps/` and `v1.0-trainval/`. The
[configuration data column](../examples/README.md#configurations) lists data and weights for
each workflow; `dataroot` or `usdz_paths` can select other locations.

The run command manages compatible NRE and trafficsim services automatically; see
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

This is the local `weights_path` in the supplied modular stop-sign configurations. The framework
does not fetch this checkpoint during a run.

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
possible; without a local checkpoint, the harmonizer downloads weights on first use. See the
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
