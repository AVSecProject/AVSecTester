#!/usr/bin/env bash
# Fetch AV perception model checkpoints into ./models and link them where avstack expects.
#
# avstack's MMDetObjectDetector3D resolves config/checkpoint paths relative to
#   third_party/avstack-core/third_party/mmdetection3d   ("mm3d_root")
# We keep the weights under ./models (git-ignored) and symlink mm3d_root/checkpoints and
# mm3d_root/work_dirs at them, so stock avstack finds them without editing vendored code.
#
# Public OpenMMLab checkpoints are downloaded automatically. CARLA-trained weights
# (dataset "carla-joint" / "carla-vehicle") are avstack-lab's own and are NOT public:
# drop them under ./models/work_dirs/<run_name>/ (see ./models/README.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$ROOT/models"
MM3D="$ROOT/third_party/avstack-core/third_party/mmdetection3d"

mkdir -p "$MODELS/checkpoints/kitti" "$MODELS/checkpoints/nuscenes" "$MODELS/work_dirs"

fetch() {  # url dest
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then echo "have   $(basename "$dest")"; return; fi
  echo "fetch  $(basename "$dest")"
  curl -fL --retry 3 -o "$dest" "$url"
}

link() {  # linkpath target
  local lnk="$1" tgt="$2"
  if [[ -L "$lnk" && "$(readlink "$lnk")" == "$tgt" ]]; then echo "linked $lnk"; return; fi
  if [[ -e "$lnk" && ! -L "$lnk" ]]; then
    echo "WARN: $lnk exists and is not a symlink; leaving it alone"; return
  fi
  rm -f "$lnk"; ln -s "$tgt" "$lnk"; echo "link   $lnk -> $tgt"
}

# --- public OpenMMLab checkpoints (LiDAR PointPillars, KITTI 3-class) ---
fetch "https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth" \
      "$MODELS/checkpoints/kitti/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth"

# --- avstack-lab CARLA-trained PointPillars (genuine CARLA perception, dataset=carla-vehicle) ---
CARLA_BASE="https://g-b0ef78.1d0d8d.03c0.data.globus.org/models/mmdet3d/work_dirs/carla"
CV="pointpillars_hv_fpn_sbn-all_8xb4-2x_carla-3d-vehicle"
CVDIR="$MODELS/work_dirs/$CV"
mkdir -p "$CVDIR"
fetch "$CARLA_BASE/$CV.py"  "$CVDIR/$CV.py"
fetch "$CARLA_BASE/$CV.pth" "$CVDIR/$CV.pth"
# avstack resolves epoch "latest" via a last_checkpoint file naming the .pth (abs path)
if [[ -f "$CVDIR/$CV.pth" ]]; then
  readlink -f "$CVDIR/$CV.pth" > "$CVDIR/last_checkpoint"
  echo "link   $CVDIR/last_checkpoint"
fi

# --- expose ./models where avstack resolves paths ---
if [[ -d "$MM3D" ]]; then
  link "$MM3D/checkpoints" "$MODELS/checkpoints"
  link "$MM3D/work_dirs"   "$MODELS/work_dirs"
else
  echo "WARN: mm3d_root not found ($MM3D); run 'git submodule update --init --recursive' first"
fi

echo "done."
