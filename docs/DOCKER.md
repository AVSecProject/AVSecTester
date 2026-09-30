# Docker — reproducible GPU + CARLA end-to-end

`Dockerfile` reproduces the full stack (torch 2.1.0+cu121, mmcv 2.1.0 / mmdet 3.2.0 / mmdet3d 1.4.0 —
prebuilt, no ops compile, avstack + CARLA client) and `docker-compose.yml` wires it to a
`carlasim/carla:0.9.16` server. Together they run the **real** end-to-end path: a CARLA-trained
PointPillars detector on a live CarlaLidar in a closed-loop drive, attacked by a phantom detection.

## Prerequisites

- An NVIDIA GPU with the nvidia container runtime.
- The nested mm* submodules on the host (needed to compile mmdet3d) and the CARLA-trained weights:

```bash
git clone --recurse-submodules <this-repo> && cd AVSecTester
git submodule update --init third_party/avstack-core
cd third_party/avstack-core && \
  git submodule update --init --depth 1 third_party/mmdetection third_party/mmdetection3d third_party/mmsegmentation && cd -
./scripts/fetch_models.sh          # pull carla-vehicle weights → ./models (bind-mounted into the image)
```

## Run the end-to-end attack

Start the stack (a CARLA server + the AVSecTester container, which defaults to an idle shell),
then run the attack in it:

```bash
docker compose up -d --build       # first build: the mmdet3d CUDA compile takes ~10–20 min
docker compose exec avsectester avsectester run configs/carla_scenario.yaml --frames 40
```

`avsectester run` **is** the demo: it builds `configs/carla_scenario.yaml`, runs a clean pass then a
phantom-attacked pass, and diffs them. Expected output:

```
[clean]    mean_detections=6.8 peak_speed=5.19 final_speed=5.17 brake_frames=0
[attacked] mean_detections=8.1 final_speed=0.00 brake_frames=38
clean:    peak_speed= 5.19  final_speed= 5.17  brake_frames=0
attacked: final_speed= 0.00  brake_frames=38
=> ATTACK SUCCEEDED (forced an unsafe stop)
```

i.e. the clean run cruises while the real detector reports NPC detections; the phantom detection
injected at the perception stage (an avstack hook) propagates to a confirmed track and forces an
unsafe stop. (Exit code: `0` succeeded, `2` inconclusive, `1` no impact.)

Add `--plot results/impact.png` to also save a **driving-impact figure** (ego speed + brake over
time, clean vs attacked). `results/` is bind-mounted, so the PNG appears on the host:

```bash
docker compose exec avsectester avsectester run configs/carla_scenario.yaml --frames 40 --plot results/impact.png
```

The container defaults to a **shell** — open one, or run anything else, with `exec`:

```bash
docker compose exec avsectester bash                                       # a shell in the environment
```

When you're done: `docker compose down`.

## Notes

- No CUDA ops are compiled: in OpenMMLab 2.0 the ops live in **mmcv** (installed as a prebuilt
  `cu121/torch2.1.0` wheel), and mmdet/mmdet3d are pure-Python. The CUDA 12.1 base runs natively on the
  L40S (Ada, sm_89).
- Watch transitive deps (scikit-image, pandas, plyfile) — their latest releases pull numpy≥2, which
  torch 2.1 cannot run; the Dockerfile pins `numpy==1.26.4` after installing them.
- `.dockerignore` drops VCS/data/model/output trees.
- `./models` is a bind mount, so weights are shared with the host rather than baked into the image;
  `fetch_models.sh` runs at container start to (re)create the mmdet3d symlinks.
- Both services use `network_mode: host`, so the client reaches the server at `127.0.0.1:2000`; the
  default docker runtime is nvidia, so both containers get GPUs (CARLA on GPU 2, AVSecTester on 1).
  This is why the scenario config targets `gpu: 0` (the ego container's dedicated GPU). Running the
  demo directly on the host instead shares GPU 2 with CARLA, so pass `--gpu 1` there
  (`avsectester run configs/carla_scenario.yaml --gpu 1`) to run neural inference on a free device.
- The manual (conda) install is documented in [`SETUP.md`](SETUP.md).
