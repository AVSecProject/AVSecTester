# Scripts

- **`fetch_models.sh`** — pull the CARLA-trained PointPillars weights into `./models` and link them
  into the mmdet3d root. Run once before any neural run.

The demo itself is **not** a script — it's the CLI, `avsectester run` (see `avsectester/cli.py`).
Everything below needs the `avsec` conda env with the `[avstack]` extras and a running CARLA server
(see `docs/SETUP.md` / `docs/DOCKER.md`); it's not part of `pytest`, which stays hardware-free.

```bash
# start a CARLA server (headless, GPU 0)
docker run -d --name carla-avsec --gpus 'device=0' --net=host \
  carlasim/carla:0.9.15 ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -quality-level=Low

conda activate avsec
./scripts/fetch_models.sh
avsectester run configs/carla_scenario.yaml --frames 40 --gpu 1 --plot results/impact.png
```

Expected:

```
[clean]    mean_detections=6.8 peak_speed=5.19 final_speed=5.17 brake_frames=0
[attacked] mean_detections=8.1 final_speed=0.00 brake_frames=38
clean:    peak_speed= 5.19  final_speed= 5.17  brake_frames=0
attacked: final_speed= 0.00  brake_frames=38
=> ATTACK SUCCEEDED (forced an unsafe stop)
[plot]     saved results/impact.png
```

- `--frames` — steps per run (enough for the clean ego to reach cruising speed; 40 is good).
- `--gpu` — perception CUDA device. The config targets GPU 0 (right in Docker, where the ego gets a
  dedicated GPU); on a single host CARLA already renders on GPU 0, so pass `--gpu 1`.
- `--plot` — save the clean-vs-attacked driving-impact figure (needs the `viz` extra:
  `pip install -e ".[viz]"`).

Exit code encodes the verdict: `0` succeeded, `2` inconclusive (clean never drove), `1` no impact.
