# Docker — reproducible GPU + CARLA end-to-end

`Dockerfile` reproduces the full stack (torch 1.13+cu117, compiled mmdet3d ops, avstack + CARLA
client) and `docker-compose.yml` wires it to a `carlasim/carla:0.9.15` server. Together they run
the **real** end-to-end path: a CARLA-trained PointPillars detector on a live CarlaLidar in a
closed-loop drive, attacked by a phantom detection.

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
docker compose exec avsectester python scripts/smoke_carla_neural.py 40
```

`scripts/smoke_carla_neural.py` runs a clean pass then a phantom-attacked pass. Expected output:

```
[clean]    frames=40 ... mean_detections=5.2 brake_frames=0
[attacked] frames=40 ... final_speed=0.0 brake_frames=39
SMOKE: PASS (phantom forced an unsafe stop)
```

i.e. the clean run cruises while the real detector reports NPC detections; the phantom detection
injected at the `perception_out` seam propagates to a track and forces an unsafe stop.

The container defaults to a **shell** — open one, or run anything else, with `exec`:

```bash
docker compose exec avsectester bash                                # a shell in the environment
docker compose exec avsectester avsectester run configs/carla_neural_experiment.yaml
docker compose exec avsectester avsectester registry
```

When you're done: `docker compose down`.

## Record screenshots (visualize the attack)

`scripts/record_carla.py` runs a clean and a phantom-attacked pass with the forward RGB camera
on, and saves, per frame, a **CARLA screenshot** and a **bird's-eye view** of the LiDAR cloud +
detector boxes (real detections green, injected phantom red):

```bash
docker compose exec avsectester python scripts/record_carla.py 40
```

Output lands under `results/carla_rec_{clean,attacked}/frames/` (`rgb_XXXX.png`, `bev_XXXX.png`,
`records.jsonl`). In the attacked BEV the phantom shows as a red `car 0.90` box floating in the
brake corridor with no LiDAR points beneath it — the injected detection that has no physical
support — while the ego brakes to a stop. `results/` is bind-mounted, so the frames appear on the
host. (The visualizer is `avsectester.viz.Recorder`, an `observer(frame, outcome)` for
`avsectester.core.run` — the metrics path is unchanged.)

## Notes

- `nvcc` 11.7 can't target Ada (sm_89) directly, so the image emits `sm_86` cubin + PTX that JITs
  on the L40S (`TORCH_CUDA_ARCH_LIST="8.0;8.6+PTX"`). On a different GPU generation, bump the base
  image to cuda 11.8+ and adjust the arch list.
- `.dockerignore` keeps `third_party/avstack-core/third_party` (the mm* sources needed to compile
  mmdet3d) and drops VCS/data/model/output trees.
- `./models` is a bind mount, so weights are shared with the host rather than baked into the image;
  `fetch_models.sh` runs at container start to (re)create the mmdet3d symlinks.
- Both services use `network_mode: host`, so the client reaches the server at `127.0.0.1:2000`; the
  default docker runtime is nvidia, so both containers get GPUs (CARLA on GPU 0, AVSecTester on 1).
- The manual (conda) install is documented in [`SETUP.md`](SETUP.md).
