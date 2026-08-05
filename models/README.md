# Model checkpoints

Perception model weights for the AV stack under test. Binaries are **not** committed
(git-ignored); this tree tracks only the structure and this README. Populate it with:

```bash
./scripts/fetch_models.sh
```

## Layout

```
models/
  checkpoints/
    kitti/       # public OpenMMLab LiDAR checkpoints (auto-downloaded)
    nuscenes/    # (optional) public OpenMMLab checkpoints
  work_dirs/     # CARLA-trained mmengine run dirs (you provide; see below)
```

`scripts/fetch_models.sh` downloads the public checkpoints here and symlinks
`third_party/avstack-core/third_party/mmdetection3d/{checkpoints,work_dirs}` at these
folders, which is where avstack's `MMDetObjectDetector3D` resolves paths (`mm3d_root`).
So weights live here, and stock avstack finds them with no edits to vendored code.

## Auto-downloaded (public)

| Model | Dataset | Source | File |
|---|---|---|---|
| PointPillars (SECFPN) | `kitti` | OpenMMLab | `checkpoints/kitti/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_...pth` |
| PointPillars (FPN) | `carla-vehicle` | avstack-lab (Globus) | `work_dirs/pointpillars_hv_fpn_sbn-all_8xb4-2x_carla-3d-vehicle/` |

- `{"model": "pointpillars", "dataset": "carla-vehicle"}` is the **CARLA-trained** detector
  (classes car/bicycle/truck/motorcycle). Verified on live CARLA: detects an NPC 10 m ahead at
  ~0.76 confidence, correct location. **Use this for CARLA.** Match avstack's default `CarlaLidar`
  (32-beam) since the model was trained on it.
- `{"model": "pointpillars", "dataset": "kitti"}` runs on any cloud but is **near-useless on CARLA
  LiDAR** (severe domain gap: misses vehicles, many false positives). Kept only as a generic
  real-neural-path check on KITTI data.

The CARLA `work_dirs/<run>/` directory holds the config `.py`, the `.pth`, and a `last_checkpoint`
text file (absolute path to the `.pth`) that avstack reads to resolve epoch `latest`. All of it is
produced by `scripts/fetch_models.sh`. Note: avstack advertises `carla-joint`/`carla-pedestrian`
too, but only `carla-vehicle` and `carla-infrastructure` weights exist on the endpoint.

## Verify

```bash
conda activate avsec
python scripts/smoke_perception_nn.py    # loads the detector, runs one forward pass on GPU
```
