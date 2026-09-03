# Minimal, reproducible image for the CARLA-free end-to-end example.
#
# The mock environment/system uses avstack's passthrough perception + tracker, which are pure
# Python — so this image is torch-free / CUDA-free and builds in minutes on any machine.
# (The neural-perception + CARLA path needs the full GPU stack; see docs/SETUP.md.)
#
#   docker build -t avsectester:mock .
#   docker run --rm avsectester:mock                      # runs the mock attack example
#   docker run --rm avsectester:mock pytest -q            # runs the offline test suite
FROM python:3.10-slim

# opencv + scientific-stack runtime libs; build-essential/git for editable source installs
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# avstack-core (BASE deps only — no torch/mmcv/mmdet3d) provides the AV modules the mock uses;
# then AVSecTester itself. pytest is included so the image can also run the offline suite.
RUN pip install --no-cache-dir -e third_party/avstack-core \
 && pip install --no-cache-dir -e ".[dev]"

# default command: the end-to-end mock attack example (clean vs attacked vs attacked+defended)
CMD ["avsectester", "run", "configs/mock_experiment.yaml"]
