# Scripts

Environment-dependent helpers for the **full avstack + CARLA stack** (need the `avsec` conda env
with the `[avstack]` extras, plus a running CARLA server — see `docs/SETUP.md` / `docs/DOCKER.md`).
These are *not* part of `pytest`, which stays hardware-free (`tests/`).

- **`fetch_models.sh`** — pull the CARLA-trained PointPillars weights into `./models` and link them
  into the mmdet3d root. Run once before any neural run.
- **`run_demo.py`** — the end-to-end demo/smoke: build `configs/carla_scenario.yaml`, run it clean
  then phantom-attacked in real CARLA, and assert the attack forced an unsafe stop.

```bash
# start a CARLA server (headless, GPU 0)
docker run -d --name carla-avsec --gpus 'device=0' --net=host \
  carlasim/carla:0.9.15 ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -quality-level=Low

conda activate avsec
./scripts/fetch_models.sh
python scripts/run_demo.py 40
```

Expected:

```
[clean]    mean_detections=5.1 final_speed=2.56 brake_frames=0
[attacked] mean_detections=6.0 final_speed=0.09 brake_frames=14
=> ATTACK SUCCEEDED (forced an unsafe stop)
SMOKE: PASS (phantom forced an unsafe stop)
```

The same run is available through the CLI: `avsectester run configs/carla_scenario.yaml --frames 40`.
