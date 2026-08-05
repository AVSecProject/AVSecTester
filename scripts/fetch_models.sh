#!/usr/bin/env bash
# Fetch AV perception model checkpoints into ./models and link them where avstack expects.
#
# avstack's MM detectors resolve config/checkpoint paths relative to two vendored roots:
#   LiDAR (mmdet3d): third_party/avstack-core/third_party/mmdetection3d   (mm3d_root)
#   2D    (mmdet)  : third_party/avstack-core/third_party/mmdetection     (mm2d_root)
# The loader (base.py) probes mm2d_root/<config> FIRST, so the 2D and 3D work_dirs trees
# MUST stay separate or a LiDAR config would misroute into the 2D branch. We therefore keep:
#   mm3d_root/work_dirs   -> models/work_dirs      (LiDAR only)
#   mm2d_root/work_dirs   -> models/work_dirs_2d   (2D only)
#   both     /checkpoints -> models/checkpoints    (stock KITTI/nuScenes/COCO)
#
# All avstack-lab CARLA-trained weights live on a public Globus mirror (no auth). Only the
# checkpoints that actually exist upstream are fetched (carla-joint/pedestrian LiDAR and
# cascade-joint 2D are 404 and intentionally omitted).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"
WD3="$MODELS/work_dirs"        # LiDAR (mmdet3d) work_dirs
WD2="$MODELS/work_dirs_2d"     # 2D (mmdet) work_dirs
MM3D="$ROOT/third_party/avstack-core/third_party/mmdetection3d"
MM2D="$ROOT/third_party/avstack-core/third_party/mmdetection"

MMDET3D_BASE="https://g-b0ef78.1d0d8d.03c0.data.globus.org/models/mmdet3d/work_dirs/carla"
MMDET_BASE="https://g-b0ef78.1d0d8d.03c0.data.globus.org/models/mmdet/work_dirs/carla"

mkdir -p "$MODELS/checkpoints/kitti" "$MODELS/checkpoints/nuscenes" "$WD3" "$WD2/carla"

fetch() {  # url dest
  local url="$1" dest="$2"
  mkdir -p "$(dirname "$dest")"
  if [[ -f "$dest" ]]; then echo "have   $(basename "$dest")"; return; fi
  echo "fetch  $(basename "$dest")"
  curl -fL --retry 3 -o "$dest" "$url"
}

# Fetch a CARLA model stored as a <name>.py + <name>.pth pair into <dir>, and (when the
# avstack path template ends in 'latest.pth') write the last_checkpoint file it reads.
fetch_pair() {  # base name dir [latest]
  local base="$1" name="$2" dir="$3" latest="${4:-}"
  fetch "$base/$name.py"  "$dir/$name.py"
  fetch "$base/$name.pth" "$dir/$name.pth"
  if [[ -n "$latest" ]]; then
    readlink -f "$dir/$name.pth" > "$dir/last_checkpoint"
    echo "write  $dir/last_checkpoint"
  fi
}

link() {  # linkpath target
  local lnk="$1" tgt="$2"
  if [[ -L "$lnk" && "$(readlink "$lnk")" == "$tgt" ]]; then echo "linked $lnk"; return; fi
  if [[ -e "$lnk" && ! -L "$lnk" ]]; then
    echo "WARN: $lnk exists and is not a symlink; leaving it alone"; return
  fi
  rm -f "$lnk"; ln -s "$tgt" "$lnk"; echo "link   $lnk -> $tgt"
}

echo "== stock OpenMMLab LiDAR PointPillars (KITTI 3-class) =="
fetch "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth" \
      "$MODELS/checkpoints/kitti/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"

echo "== avstack CARLA-trained LiDAR PointPillars (mmdet3d) =="
# dataset=carla-vehicle / carla-infrastructure -> work_dirs/<name>/{<name>.py, latest.pth via last_checkpoint}
for m in vehicle infrastructure; do
  N="pointpillars_hv_fpn_sbn-all_8xb4-2x_carla-3d-$m"
  fetch_pair "$MMDET3D_BASE" "$N" "$WD3/$N" latest
done

echo "== avstack CARLA-trained 2D detectors (mmdet) =="
# faster/cascade carla-vehicle + cascade carla-infrastructure -> work_dirs/carla/<name>.{py,pth} (direct)
for N in faster_rcnn_r50_fpn_1x_carla_vehicle \
         cascade-rcnn_r50_fpn_1x_carla_vehicle \
         cascade-rcnn_r50_fpn_1x_carla_infrastructure; do
  fetch_pair "$MMDET_BASE" "$N" "$WD2/carla"
done
# faster carla-infrastructure -> work_dirs/<name>.{py,pth} (FLAT, per avstack's path map)
fetch_pair "$MMDET_BASE" "faster_rcnn_r50_fpn_1x_carla_infrastructure" "$WD2"
# faster carla-joint -> work_dirs/<name>/{<name>.py, latest.pth via last_checkpoint}
NJ="faster_rcnn_r50_fpn_1x_carla_joint"
fetch_pair "$MMDET_BASE" "$NJ" "$WD2/$NJ" latest

echo "== expose ./models where avstack resolves paths =="
if [[ -d "$MM3D" ]]; then
  link "$MM3D/checkpoints" "$MODELS/checkpoints"
  link "$MM3D/work_dirs"   "$WD3"
else echo "WARN: mm3d_root not found ($MM3D); run git submodule update --init --recursive"; fi
if [[ -d "$MM2D" ]]; then
  link "$MM2D/checkpoints" "$MODELS/checkpoints"
  link "$MM2D/work_dirs"   "$WD2"
else echo "WARN: mm2d_root not found ($MM2D); run git submodule update --init --recursive"; fi

echo "done."
