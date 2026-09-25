# GPU stack for the end-to-end path: neural perception (CARLA-trained mmdet3d) in a closed-loop
# CARLA simulation. Reproduces docs/SETUP.md §2, upgraded to CUDA 12.8 / torch 2.7.1 so it runs on
# Blackwell (sm_120) GPUs — torch 1.13.1+cu117 ships no sm_120 kernels.
#
# Usually built via docker-compose.yml (which also runs a CARLA 0.9.15 server):
#   docker compose up --build
# Standalone build (needs the nested mm* submodules initialized on the host — see docs/DOCKER.md):
#   docker build -t avsectester .
#
# mmdet 3.0 / mmdet3d 1.1 assert mmcv<2.1.0, and no mmcv 2.0.x wheel exists for torch 2.7/cu128,
# so mmcv 2.0.1 is built from source (step 2). mmdet3d 1.1 itself has no CUDA extensions.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip git build-essential ninja-build \
      libgl1 libglib2.0-0 curl wget ca-certificates \
 && ln -sf /usr/bin/python3.10 /usr/bin/python && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
# name-shim symlinks avcarla's metadata expects (avstack-core/api are our submodule names)
RUN ln -sf avstack-core third_party/lib-avstack-core && ln -sf avstack-api third_party/lib-avstack-api

# 1. torch + torchvision (cu128 — first CUDA line with Blackwell sm_120 support)
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
      --index-url https://download.pytorch.org/whl/cu128
# 2. mmengine + mmcv 2.0.1 from source, compiled for sm_120. Compat patch:
#    mmcv 2.0.1's setup.py forces -std=c++14, but torch>=2.1 headers require C++17.
RUN pip install --no-cache-dir "mmengine>=0.7.3,<0.8" "numpy==1.24.4" -c constraints.txt \
 && git clone --depth 1 --branch v2.0.1 https://github.com/open-mmlab/mmcv.git /opt/mmcv \
 && sed -i 's/c++14/c++17/g' /opt/mmcv/setup.py \
 && cd /opt/mmcv && CUDA_HOME=/usr/local/cuda FORCE_CUDA=1 MMCV_WITH_OPS=1 \
      TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=16 \
      pip install --no-cache-dir --no-build-isolation . -c /app/constraints.txt \
 && rm -rf /opt/mmcv
# 3. mm-detectors editable (pure Python — mmdet3d 1.1 takes all CUDA ops from mmcv)
RUN pip install --no-cache-dir -e third_party/avstack-core/third_party/mmdetection    -c constraints.txt \
 && pip install --no-cache-dir -e third_party/avstack-core/third_party/mmsegmentation -c constraints.txt \
 && pip install --no-cache-dir -e third_party/avstack-core/third_party/mmdetection3d -c constraints.txt
# 4. avstack packages + CARLA 0.9.15 client
RUN pip install --no-cache-dir -e third_party/avstack-core -c constraints.txt \
 && pip install --no-cache-dir -e third_party/avstack-api  -c constraints.txt \
 && pip install --no-cache-dir -e third_party/lib-avstack-carla --no-deps \
 && pip install --no-cache-dir "carla==0.9.15" pygame -c constraints.txt
# 5. AVSecTester (+ viz extra for the --plot driving-impact figure)
RUN pip install --no-cache-dir -e ".[dev,viz]"

# default to a shell — run attacks/recordings explicitly (see docs/DOCKER.md / docker-compose.yml)
CMD ["bash"]
