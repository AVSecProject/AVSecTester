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
# AlpaSim's uv settings use a relative exclude-newer, which uv 0.8 cannot parse.
FROM ghcr.io/astral-sh/uv:0.9.30 AS uv-nurec

# NuRec + Alpamayo image. AlpaSim's driver requires Python 3.12, so it cannot share the Python 3.11
# environment below. This target installs only the AlpaSim packages AVSecTester imports
# (alpasim_driver with its model dependencies, and alpasim_grpc) plus the framework dependencies.
# AVSecTester itself is not installed: mount the checkout at /app and run from it.
#   docker build --target nurec -t avsectester:nurec .
# It is defined before the default stage, so `docker build .` and docker compose still build the
# CARLA image. The same base and apt layer are shared with that image.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS nurec

COPY --from=uv-nurec /uv /usr/local/bin/uv

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential ninja-build \
      libgl1 libglib2.0-0 curl wget ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# utils_rs, required by the driver through alpasim_utils[geometry], is a Rust extension built here.
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH="/opt/cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain 1.90.0

ARG ALPASIM_COMMIT=affc2eab209fa43bdfa2f26c0f8d437922d78a68
RUN git clone https://github.com/NVlabs/alpasim /opt/alpasim \
 && git -C /opt/alpasim checkout --quiet "${ALPASIM_COMMIT}"
WORKDIR /opt/alpasim
ENV UV_PYTHON_INSTALL_DIR=/opt/python UV_LINK_MODE=copy UV_NO_CACHE=1
RUN uv sync --python 3.12 --package alpasim_driver --no-dev \
 && uv run --no-sync compile-protos
ENV PATH="/opt/alpasim/.venv/bin:${PATH}"

# Install the same framework dependencies as the Python 3.11 image, including the CLI and
# report exporters. AlpaSim controls NumPy: keep its installed version rather than applying
# the CARLA/OpenMMLab-specific NumPy<2 constraint from AVSecTester's Python 3.11 environment.
COPY pyproject.toml /tmp/avsectester-pyproject.toml
RUN python -c 'import re, tomllib, numpy; from pathlib import Path; project = tomllib.load(open("/tmp/avsectester-pyproject.toml", "rb"))["project"]; deps = project["dependencies"] + project["optional-dependencies"]["nurec"]; Path("/tmp/avsectester-requirements.txt").write_text("\n".join(d for d in deps if re.match(r"[A-Za-z0-9_.-]+", d).group(0).lower() != "numpy") + "\n"); Path("/tmp/alpasim-numpy.txt").write_text("numpy==" + numpy.__version__ + "\n")' \
 && uv pip install --python /opt/alpasim/.venv/bin/python \
      -r /tmp/avsectester-requirements.txt -c /tmp/alpasim-numpy.txt \
 && python -c "import typer, yaml, matplotlib, PIL, pyarrow.parquet, alpasim_grpc; from embreex.rtcore_scene import EmbreeScene; from embreex.mesh_construction import TriangleMesh; from alpasim_driver.models.alpamayo1_5_model import Alpamayo15Model" \
 && rm /tmp/avsectester-pyproject.toml /tmp/avsectester-requirements.txt /tmp/alpasim-numpy.txt

WORKDIR /app
ENV PYTHONPATH=/app
CMD ["bash"]

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
      pycocotools terminaltables plyfile tensorboard networkx
# 4. avstack packages + CARLA 0.9.16 client
RUN python -m pip install --no-cache-dir -e third_party/avstack-core -c constraints.txt \
 && python -m pip install --no-cache-dir -e third_party/avstack-api  --no-deps \
 && python -m pip install --no-cache-dir -e third_party/lib-avstack-carla --no-deps \
 && python -m pip install --no-cache-dir "carla==0.9.16" pygame ipywidgets nuscenes-devkit lyft-dataset-sdk==0.0.8 -c constraints.txt
# 5. AVSecTester (includes the pinned Parquet dependency from pyproject.toml;
# plots are part of reports; augment adds Albumentations and nurec adds mesh ray casting).
# Torch 2.1 and MMEngine use pkg_resources, which newer setuptools versions remove.
RUN python -m pip install --no-cache-dir -e ".[dev,augment,nurec]" "setuptools==69.5.1" \
 && python -c "import pyarrow.parquet; from embreex.rtcore_scene import EmbreeScene; from embreex.mesh_construction import TriangleMesh"

# default to a shell — run attacks/recordings explicitly (see docs/getting-started/installation.md#docker / docker-compose.yml)
CMD ["bash"]
