# Scripts

- **`fetch_models.sh`** — pull the CARLA-trained PointPillars weights into `./models` and link them
  into the mmdet3d root. Run once before any neural CARLA run.
- **`alpamayo_nurec_demo.py`** — the end-to-end **NuRec + Alpamayo** demo: drives the real
  Alpamayo-1.5-10B policy on photoreal NuRec imagery via `run(NuRecBackend, AlpamayoAVStack, frames)`.
  Runs in the **AlpaSim driver env** (Python 3.12); `--stub` swaps in black frames (no renderer),
  `--save-frames` dumps each frame under `./tmp/alpamayo_nurec/`. See §4 of `docs/SETUP.md`.

  ```bash
  cd /workspace/nvme/qzzhang/alpasim
  HF_HOME=/workspace/hdd/models/huggingface PYTHONPATH=<AVSecTester> \
    uv run python <AVSecTester>/scripts/alpamayo_nurec_demo.py 8 --save-frames --gpu 1
  ```

The **CARLA modular demo** is not a script — it's the CLI, `avsectester run` (see `avsectester/cli.py`).
It needs the `avsec` conda env with the `[avstack]` extras and a running CARLA server
(see `docs/SETUP.md` / `docs/DOCKER.md`). Neither demo is part of `pytest`, which stays hardware-free.

```bash
# start a CARLA server (headless, GPU 2)
docker run -d --name carla-avsec --gpus 'device=2' --net=host \
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
[output]   wrote driving-impact figure to file: /.../AVSecTester/results/impact.png
```

- `--frames` — steps per run (enough for the clean ego to reach cruising speed; 40 is good).
- `--gpu` — perception CUDA device. The config targets GPU 0 (right in Docker, where the ego gets a
  dedicated GPU); on a single host CARLA already renders on GPU 2, so pass `--gpu 1`.
- `--plot` — save the clean-vs-attacked driving-impact figure (needs the `viz` extra:
  `pip install -e ".[viz]"`).

Exit code encodes the verdict: `0` succeeded, `2` inconclusive (clean never drove), `1` no impact.
