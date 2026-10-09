# Docker — reproducible GPU + CARLA end-to-end

`Dockerfile` reproduces the full stack (torch 2.1.0+cu121, mmcv 2.1.0 / mmdet 3.2.0 / mmdet3d 1.4.0 —
with prebuilt MMCV CUDA operators, avstack and the CARLA client) and `docker-compose.yml` wires it to a
`carlasim/carla:0.9.16` server. Together they run the **real** end-to-end path: a CARLA-trained
PointPillars detector on a live CarlaLidar in a closed-loop drive, attacked by a phantom detection.

The AVSecTester image includes the Python environment and CARLA/avstack dependencies, including
`lyft-dataset-sdk`. No host Conda environment or separate Python-package installation is required
for this path. Model weights remain separate, and CARLA runs in its own server container.
Alpamayo and the NuRec renderer client use the separate AlpaSim driver environment described in
[SETUP.md §4](SETUP.md#4-nurec--alpamayo-the-end-to-end-path). They are not installed by this
Dockerfile.

## Prerequisites

- NVIDIA GPUs and Docker with the NVIDIA runtime configured as its default runtime.
  The supplied Compose file selects physical GPU 2 for CARLA and GPU 1 for AVSecTester.
  Adjust `NVIDIA_VISIBLE_DEVICES` in that file to match your machine before starting it.
- The nested mmdet/mmdet3d submodules on the host (checkpoint-symlink roots, not a build step) and the
  CARLA-trained weights:

```bash
git clone --recurse-submodules https://github.com/AVSecProject/AVSecTester.git && cd AVSecTester
git submodule update --init third_party/avstack-core
cd third_party/avstack-core && \
  git submodule update --init --depth 1 third_party/mmdetection third_party/mmdetection3d && cd -
./scripts/preparation/fetch_models.sh          # pull carla-vehicle weights → ./models (bind-mounted into the image)
```

## Run the end-to-end attack

Start the stack (a CARLA server + the AVSecTester container, which defaults to an idle shell),
then run the attack in it:

```bash
docker compose up -d --build       # first build downloads prebuilt wheels (no CUDA compile)
docker compose exec avsectester avsectester run configs/carla_scenario.yaml --frames 40
```

`avsectester run` **is** the demo: it builds `configs/carla_scenario.yaml`, runs a clean pass then a
phantom-attacked pass, and compares their driving traces. It prints detection counts, peak and
final speeds, braking-frame counts and a verdict. Results depend on the configured scene and
whether the baseline establishes driving.

| Exit code | Meaning |
|---|---|
| `0` | Attack succeeded under the current driving-impact criterion |
| `1` | A valid baseline was established, but the attack did not meet that criterion |
| `2` | Inconclusive because the clean run did not establish driving |

See [INTERFACE.md](INTERFACE.md#3-metric-clean-vs-attacked--verdict) for the criterion.

Add `--plot results/impact.png` to also save a **driving-impact figure** (ego speed + brake over
time, clean vs attacked). The default Compose file mounts `models/`, not the output directory.
Copy the generated image from the container to retain it on the host:

```bash
docker compose exec avsectester avsectester run configs/carla_scenario.yaml --frames 40 --plot results/impact.png
mkdir -p results
docker compose cp avsectester:/app/results/impact.png results/impact.png
```

The container defaults to a **shell** — open one, or run anything else, with `exec`:

```bash
docker compose exec avsectester bash                                       # a shell in the environment
```

When you're done: `docker compose down`.

## Notes

- MMCV's CUDA operators use a prebuilt wheel. The version pins and the NumPy <2
  constraint are explained in [`SETUP.md`](SETUP.md).
- `.dockerignore` drops VCS/data/model/output trees.
- `./models` is a bind mount, so weights are shared with the host rather than baked into the image.
  `fetch_models.sh` runs at container start to (re)create the mmdet/mmdet3d checkpoint symlinks.
- Both services use `network_mode: host`, so the client reaches the server at `127.0.0.1:2000`. The
  default docker runtime is nvidia, so both containers get GPUs (CARLA on GPU 2, AVSecTester on 1).
  This is why the scenario config targets `gpu: 0` (the inference container's dedicated GPU).
  When running directly on the host, pass `--gpu 1` to select that same physical inference device
  (`avsectester run configs/carla_scenario.yaml --gpu 1`) to run neural inference on a free device.
- The alternative manual Python installation is documented in [`SETUP.md`](SETUP.md).
- After changing the Dockerfile, rebuild with `docker compose build avsectester`. Pulling Git
  changes does not update an existing image.
