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

| Model | Dataset | File |
|---|---|---|
| PointPillars (SECFPN) | `kitti` | `checkpoints/kitti/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth` |

Selected in a spec/detector as `{"model": "pointpillars", "dataset": "kitti"}`. KITTI-trained
weights run on CARLA LiDAR with a domain gap; they prove the real neural path end to end.
For deployment-grade CARLA results, use the CARLA-trained weights below.

## CARLA-trained weights (you provide)

These are avstack-lab's own trained weights and are **not** on a public CDN. Obtain them from
avstack-lab and drop each mmengine run directory in as-is (it already contains the config,
`epoch_*.pth`, and a `last_checkpoint` file). avstack expects these exact directory names:

| Dataset id | Directory (under `models/work_dirs/`) |
|---|---|
| `carla-vehicle` | `pointpillars_hv_fpn_sbn-all_8xb4-2x_carla-3d-vehicle/` |
| `carla-joint` | `pointpillars_hv_fpn_sbn-all_8xb4-2x_carla-3d-joint/` |

Each directory must contain:

```
pointpillars_hv_fpn_sbn-all_8xb4-2x_carla-3d-vehicle/
  pointpillars_hv_fpn_sbn-all_8xb4-2x_carla-3d-vehicle.py   # config
  epoch_XX.pth                                              # weights
  last_checkpoint                                           # text: absolute path to epoch_XX.pth
```

Then select `{"model": "pointpillars", "dataset": "carla-vehicle"}` (or `carla-joint`).

## Verify

```bash
conda activate avsec
python scripts/smoke_perception_nn.py    # loads the detector, runs one forward pass on GPU
```
