# GPU stack for the end-to-end path: neural perception (CARLA-trained mmdet3d) in a closed-loop
# CARLA simulation. Reproduces docs/SETUP.md §2 on a CUDA 12.1 base (mirrors the verified avsec311 env).
#
# Usually built via docker-compose.yml (which also runs a CARLA 0.9.16 server):
#   docker compose up --build
# Standalone build:
#   docker build -t avsectester .
#
# OpenMMLab-official stack: torch 2.1.0+cu121, mmcv 2.1.0, mmdet 3.2.0, mmdet3d 1.4.0 — every version
# cap is satisfied, so no source build and no cap patching. mmcv ships the CUDA ops (prebuilt wheel).
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-dev python3-pip git build-essential ninja-build \
      libgl1 libglib2.0-0 curl wget ca-certificates \
 && ln -sf /usr/bin/python3.11 /usr/bin/python && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
# name-shim symlinks avcarla's metadata expects (avstack-core/api are our submodule names)
RUN ln -sf avstack-core third_party/lib-avstack-core && ln -sf avstack-api third_party/lib-avstack-api

# 1. torch + torchvision (cu121)
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 \
      --index-url https://download.pytorch.org/whl/cu121
# 2. mmcv 2.1.0 prebuilt wheel for torch2.1.0/cu121 (no compile) + mmengine
RUN pip install --no-cache-dir mmcv==2.1.0 \
      -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html "mmengine>=0.10"
# 3. mmdet 3.2.0 + mmdet3d 1.4.0 (all caps satisfied → no patch); keep numpy<2 (deps may pull numpy>=2)
RUN pip install --no-cache-dir --no-deps mmdet==3.2.0 mmdet3d==1.4.0 \
 && pip install --no-cache-dir "numpy==1.26.4" "scikit-image>=0.21,<0.24" "pandas>=2,<3" \
      pycocotools terminaltables plyfile trimesh tensorboard networkx
# 4. avstack packages + CARLA 0.9.16 client
RUN pip install --no-cache-dir -e third_party/avstack-core -c constraints.txt \
 && pip install --no-cache-dir -e third_party/avstack-api  --no-deps \
 && pip install --no-cache-dir -e third_party/lib-avstack-carla --no-deps \
 && pip install --no-cache-dir "carla==0.9.16" pygame ipywidgets nuscenes-devkit -c constraints.txt
# 5. AVSecTester (+ viz extra for the --plot driving-impact figure, augment for the corruption backend)
RUN pip install --no-cache-dir -e ".[dev,viz,augment]"

# default to a shell — run attacks/recordings explicitly (see docs/DOCKER.md / docker-compose.yml)
CMD ["bash"]
