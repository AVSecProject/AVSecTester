# GPU stack for the end-to-end path: neural perception (CARLA-trained mmdet3d) in a closed-loop
# CARLA simulation. Reproduces the local full installation in docs/getting-started/installation.md
# on a CUDA 12.1 base (mirrors the verified avsec311 env).
#
# Usually built via docker-compose.yml (which also runs a CARLA 0.9.16 server):
#   docker compose up --build
# Standalone build:
#   docker build -t avsectester .
#
# OpenMMLab-official stack: torch 2.1.0+cu121, mmcv 2.1.0, mmdet 3.2.0, mmdet3d 1.4.0 — every version
# cap is satisfied, so no source build and no cap patching. mmcv ships the CUDA ops (prebuilt wheel).
FROM ghcr.io/astral-sh/uv:0.8.22 AS uv
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

COPY --from=uv /uv /usr/local/bin/uv

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential ninja-build \
      libgl1 libglib2.0-0 curl wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Jammy's python3.11 package is a release candidate that crashes TorchScript.
# Use a stable interpreter and keep all Python dependencies in one environment.
ENV UV_PYTHON_INSTALL_DIR=/opt/python
RUN uv python install 3.11.13 \
 && uv venv --python 3.11.13 --seed /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app
COPY . /app
# name-shim symlinks avcarla's metadata expects (avstack-core/api are our submodule names)
RUN ln -sf avstack-core third_party/lib-avstack-core && ln -sf avstack-api third_party/lib-avstack-api

# 1. torch + torchvision (cu121)
RUN python -m pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 \
      --index-url https://download.pytorch.org/whl/cu121
# 2. mmcv 2.1.0 prebuilt wheel for torch2.1.0/cu121 (no compile) + mmengine
RUN python -m pip install --no-cache-dir mmcv==2.1.0 \
      -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html "mmengine>=0.10"
# 3. mmdet 3.2.0 + mmdet3d 1.4.0 (all caps satisfied → no patch); keep numpy<2 (deps may pull numpy>=2)
RUN python -m pip install --no-cache-dir --no-deps mmdet==3.2.0 mmdet3d==1.4.0 \
 && python -m pip install --no-cache-dir "numpy==1.26.4" "scikit-image>=0.21,<0.24" "pandas>=2,<3" \
      pycocotools terminaltables plyfile trimesh tensorboard networkx
# 4. avstack packages + CARLA 0.9.16 client
RUN python -m pip install --no-cache-dir -e third_party/avstack-core -c constraints.txt \
 && python -m pip install --no-cache-dir -e third_party/avstack-api  --no-deps \
 && python -m pip install --no-cache-dir -e third_party/lib-avstack-carla --no-deps \
 && python -m pip install --no-cache-dir "carla==0.9.16" pygame ipywidgets nuscenes-devkit lyft-dataset-sdk==0.0.8 -c constraints.txt
# 5. AVSecTester (+ viz extra for the --plot driving-impact figure, augment for the corruption backend)
# Torch 2.1 and MMEngine use pkg_resources, which newer setuptools versions remove.
RUN python -m pip install --no-cache-dir -e ".[dev,viz,augment]" "setuptools==69.5.1"

# default to a shell — run attacks/recordings explicitly (see docs/getting-started/installation.md#docker / docker-compose.yml)
CMD ["bash"]
