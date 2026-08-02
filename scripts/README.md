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
                   -> route follower (CARLA lane waypoints) -> VehiclePIDController
                   -> carla.VehicleControl
```

The ego is driven **entirely by avstack's own controller — no CARLA autopilot**. To prove the
custom control has real authority (throttle *and* steering), the ego follows the actual road:
each step targets a CARLA lane waypoint ahead, so the avstack lateral PID must steer the car
through curves and junction turns. Perception runs in ground-truth mode, so **no model
checkpoints are needed** — that detector is exactly the seam where an attacked/real detector
gets swapped in later. NPC traffic uses autopilot.

```bash
python scripts/demo_avstack_carla.py --frames 300 --npcs 15
```

Reference-box result (4× L40S, CARLA 0.9.15): ego follows the lane through a ~90° junction turn
over 300 sync frames — **cross-track ≤2.4 m**, ~112° cumulative heading change, peak steer 0.43,
steady 5.2 m/s under PID, with 15 detections → 15 confirmed tracks → `DRIVE: PASS`. The steer
trace peaks mid-turn (−0.33) and settles to ~0 on the straight, i.e. the controller is genuinely
regulating the vehicle, not coasting on defaults.

> Three upstream avstack/avcarla quirks worked around (documented inline): `GoStraightPlanner`
> swaps its `Pose(position, attitude)` args and raises; `get_obj_type_from_actor` only maps
> 2/4-wheeled actors (we restrict NPCs to 4-wheel cars); and `actor.get_location()` returns the
> origin until the first `world.tick()` after spawn (we settle one tick before anchoring the route).

All three scripts printed `PASS` on the reference box (torch 1.13.1+cu117).
