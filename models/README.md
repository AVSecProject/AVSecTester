# Model checkpoints

Perception model weights for the AV stack under test. Binaries are **not** committed
(git-ignored); this tree tracks only the structure and this README. Populate it with:

```bash
./scripts/preparation/fetch_models.sh
```

## Layout

```
models/
  checkpoints/
    kitti/          # public OpenMMLab LiDAR checkpoints (auto-downloaded)
    nuscenes/       # (optional) public OpenMMLab checkpoints
  work_dirs/        # CARLA-trained LiDAR (mmdet3d) run dirs
  work_dirs_2d/     # CARLA-trained 2D camera (mmdet) run dirs
```

`scripts/preparation/fetch_models.sh` downloads the checkpoints and symlinks the two vendored roots at
these folders, which is where avstack resolves paths:

- `third_party/avstack-core/third_party/mmdetection3d/{checkpoints,work_dirs}` → LiDAR (`mm3d_root`)
- `third_party/avstack-core/third_party/mmdetection/{checkpoints,work_dirs}` → 2D (`mm2d_root`)

The LiDAR and 2D `work_dirs` are kept **separate** (`work_dirs` vs `work_dirs_2d`) on purpose:
avstack's loader probes `mm2d_root/<config>` first, so a shared tree would misroute a LiDAR
config into the 2D branch.

## avstack CARLA-trained models

All are trained on CARLA with classes **car / bicycle / truck / motorcycle**.

| Kind | `model` | `dataset` | Threshold | Notes |
|---|---|---|---|---|
| LiDAR | `pointpillars` | `carla-vehicle` | 0.3 | Ego LiDAR detector. |
| LiDAR | `pointpillars` | `carla-infrastructure` | 0.3 | Roadside/elevated-viewpoint LiDAR (for infrastructure sensors, not the ego). |
| 2D cam | `fasterrcnn` | `carla-vehicle` | 0.7 | Ego camera 2D detector. |
| 2D cam | `fasterrcnn` | `carla-infrastructure` | 0.7 | Infrastructure camera. |
| 2D cam | `fasterrcnn` | `carla-joint` | 0.7 | Joint vehicle+infrastructure. |
| 2D cam | `cascadercnn` | `carla-vehicle` | 0.5 | Ego camera, cascade head. |
| 2D cam | `cascadercnn` | `carla-infrastructure` | 0.5 | Infrastructure camera, cascade head. |

Each CARLA `work_dir` holds the config `.py`, the `.pth`, and (for the epoch-`latest` path
layouts) a `last_checkpoint` text file (absolute path to the `.pth`) that avstack reads. All
of it is produced by `scripts/preparation/fetch_models.sh`.

## External CARLA datasets

The `nuCarla` submodule supplies external dataset resources. It is not connected to the current
scenario-selection or driving pipeline, and its listed models are not loaded by the supplied demos.

- **nuCarla** — `third_party/nuCarla` (git submodule of
  [michigan-traffic-lab/nuCarla](https://github.com/michigan-traffic-lab/nuCarla)). A
  nuScenes-format, **camera-BEV** CARLA dataset (9 maps, 14 weathers, 6 classes) plus four
  pretrained BEV detectors (BEVFormer / PETR / BEVDet / FastBEV, weights on the repo's GitHub
  releases). The image data lives on
  [HuggingFace](https://huggingface.co/datasets/zhijieq/nuCarla); fetch it with the repo's
  `download.sh` (Town01–10 / Mcity / Metadata zips). Keep the downloaded data **out of git**
  — put it under a git-ignored `data/` dir (repo root `/data/` is ignored) or inside the
  submodule (its own `.gitignore` excludes `data/`). Note: the LiDAR files are dummy
  placeholders — nuCarla is camera-only.

## Usage

The default `configs/carla_scenario.yaml` selects the CARLA vehicle PointPillars checkpoint.
`ModularAVStack` automatically registers its custom `CarlaDataset` metadata before loading it.
See [SETUP.md](../docs/SETUP.md#carla-perception-checkpoints) for model loading and
[DOCKER.md](../docs/DOCKER.md) for the CARLA driving command.
