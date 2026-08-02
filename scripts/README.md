# Smoke tests

Manual, environment-dependent checks for the **full avstack + CARLA stack** (require the
`avsec` conda env with the `[avstack]` extras installed — see `docs/SETUP.md`). These are
*not* part of the default `pytest` run, which stays core-only and hardware-free.

```bash
conda activate avsec

# 1. Perception: compiled mmcv/mmdet3d CUDA ops run on this GPU (needs a CUDA GPU)
python scripts/smoke_perception.py

# 2. Closed-loop CARLA: connect, sync-mode ticks, ego + LiDAR sensor (needs the CARLA server)
docker run -d --name carla-avsec --gpus 'device=0' --net=host \
  carlasim/carla:0.9.15 ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -quality-level=Epic
python scripts/smoke_carla.py
```

Both printed `PASS` on the reference box (4× L40S, CARLA 0.9.15, torch 1.13.1+cu117).
