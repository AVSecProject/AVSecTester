#!/usr/bin/env bash
# Stand up the isolated `libcom` conda env used by patch_composite.LibcomHarmonizer.
#
# libcom's deps (mmdet 3.2 / mmpose / diffusers / chumpy) conflict with the avstack stack, so the
# learned harmonizer runs OUT-OF-PROCESS in its own env (scripts/libcom_harmonize.py). We install
# the AVSecProject fork of libcom (the upstream wheel is broken — see the fork's fix commit):
#   * added __init__.py to image_harmonization/source/src{,/lbm} (find_packages dropped the subtree)
#   * removed the deprecated hf_hub_download(progress=True) kwarg (broke the HF weight download)
#
# Only the harmonization model (PCTNet) is needed, so we install its deps and skip the heavy,
# conflicting ones (mmpose/mmdet/chumpy) via `pip install -e . --no-deps`.
set -euo pipefail

ENV=${1:-libcom}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
FORK_DIR=${2:-$REPO/third_party/libcom}   # the AVSecProject/libcom fork, vendored as a submodule

# ensure the submodule is checked out (git@github.com:AVSecProject/libcom.git, pinned in .gitmodules)
if [ ! -f "$FORK_DIR/setup.py" ] && [ ! -f "$FORK_DIR/pyproject.toml" ]; then
  git -C "$REPO" submodule update --init third_party/libcom
fi

conda create -y -n "$ENV" python=3.10
conda run -n "$ENV" pip install \
  torch torchvision opencv-python scikit-image scikit-learn pillow numpy scipy \
  tqdm pyyaml easydict timm einops albumentations lpips \
  diffusers transformers accelerate huggingface_hub safetensors omegaconf pytorch_lightning kornia
conda run -n "$ENV" pip install -e "$FORK_DIR" --no-deps

echo "Verifying harmonization import + weight download ..."
conda run -n "$ENV" python -c "from libcom import ImageHarmonizationModel; print('libcom harmonization ready')"
echo "Done. patch_composite.LibcomHarmonizer(env='$ENV') will now work."
