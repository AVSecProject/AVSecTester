# Scripts

Environment-dependent helpers for the **full avstack + CARLA stack** (need the `avsec` conda env
with the `[avstack]` extras, plus a running CARLA server — see `docs/SETUP.md` / `docs/DOCKER.md`).
These are *not* part of `pytest`, which stays hardware-free (`tests/`).

- **`fetch_models.sh`** — pull the CARLA-trained PointPillars weights into `./models` and link them
  into the mmdet3d root. Run once before any neural run.
- **`run_demo.py`** — the end-to-end demo/smoke: build `configs/carla_scenario.yaml`, run it clean
  then phantom-attacked in real CARLA, and assert the attack forced an unsafe stop.
  `python scripts/run_demo.py [frames] [--gpu N]`.

```bash
# start a CARLA server (headless, GPU 0)
docker run -d --name carla-avsec --gpus 'device=0' --net=host \
  carlasim/carla:0.9.15 ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -quality-level=Low

conda activate avsec
./scripts/fetch_models.sh
python scripts/run_demo.py 40 --gpu 1     # CARLA holds GPU 0, so run neural inference on GPU 1
```

`--gpu` overrides the perception CUDA device. The scenario config targets GPU 0 (correct in Docker,
where the ego container gets a dedicated GPU); on a single host CARLA already renders on GPU 0, so
pass `--gpu 1` (or another free device) to avoid contention. The CLI takes the same flag:
`avsectester run configs/carla_scenario.yaml --frames 40 --gpu 1`.

Add `--plot results/impact.png` to also save a **driving-impact figure** (ego speed + brake command
over time, clean vs attacked) — needs the `viz` extra (`pip install -e ".[viz]"`):

```bash
python scripts/run_demo.py 40 --gpu 1 --plot results/impact.png
```

Expected:

```
[clean]    peak_speed=5.19 final_speed=5.17 brake_frames=0
[attacked] final_speed=0.00 brake_frames=38
=> ATTACK SUCCEEDED (forced an unsafe stop)
SMOKE: PASS (phantom forced an unsafe stop)
```

The same run is available through the CLI: `avsectester run configs/carla_scenario.yaml --frames 40`.
