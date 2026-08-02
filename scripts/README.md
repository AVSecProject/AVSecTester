# Smoke tests & demo

Manual, environment-dependent scripts for the **full avstack + CARLA stack** (require the
`avsec` conda env with the `[avstack]` extras installed — see `docs/SETUP.md`). These are
*not* part of the default `pytest` run, which stays core-only and hardware-free.

Start the CARLA server once (headless, GPU 0):

```bash
docker run -d --name carla-avsec --gpus 'device=0' --net=host \
  carlasim/carla:0.9.15 ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -quality-level=Epic
```

```bash
conda activate avsec

# 1. Perception: compiled mmcv/mmdet3d CUDA ops run on this GPU (needs a CUDA GPU)
python scripts/smoke_perception.py

# 2. Closed-loop sensor path: connect, sync-mode ticks, ego + LiDAR sensor
python scripts/smoke_carla.py
```

## Closed-loop demo — avstack drives a CARLA ego

`scripts/demo_avstack_carla.py` runs avstack's full decision stack in the loop:

```
CARLA ground-truth -> Passthrough3DObjectDetector -> BasicBoxTracker3D
                   -> forward-waypoint plan -> VehiclePIDController -> carla.VehicleControl
```

The ego is driven entirely by avstack (NPC traffic uses autopilot). Perception runs in
ground-truth mode, so **no model checkpoints are needed** — and that detector is exactly
the seam where an attacked/real detector gets swapped in later.

```bash
python scripts/demo_avstack_carla.py --frames 200 --npcs 30
```

Reference-box result (4× L40S, CARLA 0.9.15): 30 detections → 30 confirmed tracks held for
200 sync frames, ego accelerates 0→~5.3 m/s under PID control → `DEMO: PASS`.

> Two upstream avstack quirks worked around in the demo (documented inline): `GoStraightPlanner`
> swaps its `Pose(position, attitude)` args and raises (we inline the waypoint push); and
> `get_obj_type_from_actor` only maps 2/4-wheeled actors (we restrict NPCs to 4-wheel cars).

All three scripts printed `PASS` on the reference box (torch 1.13.1+cu117).
