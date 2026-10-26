# Repository tools

Run experiments from the repository root with the supplied configurations:

```bash
avsectester run configs/carla_phantom.yaml
avsectester run configs/carla_patch.yaml
avsectester run configs/carla_patch_optimized.yaml
avsectester run configs/carla_object_insertion.yaml
avsectester run configs/nurec_speed_feedback.yaml
avsectester run configs/nurec_object_insertion.yaml
avsectester run configs/nurec_patch.yaml
avsectester run configs/nurec_stop_sign_patch_alpamayo_2601.yaml
avsectester run configs/nurec_stop_sign_patch_alpamayo_2604.yaml
avsectester run configs/nuscenes_object_insertion.yaml
```

Before running, set the server, device, model and dataset locations in the selected configuration.
Paths are relative to that configuration file. CARLA uses a dedicated synchronous server and the
Modular or camera driving policy. NuRec uses a running neural renderer and Alpamayo in its driver
environment. nuScenes uses recorded images; commands do not alter recorded trajectories. Its
configuration uses the supported Faster R-CNN/nuScenes detector pair and requires the checkpoint
downloaded by scripts/preparation/fetch_models.sh.

The NuRec renderer must start with --enable-editing-actors.
Every render includes all supported recorded actor tracks, including unchanged ones; see the
[renderer setup](../docs/getting-started/installation.md#nurec-renderer-and-scenes).

The command prepares metadata and artifacts, executes clean and attacked treatments, and writes
RunLog records and evaluation under runs.output. The configuration keeps components at the top
level and treatments, seeds, environment, recording and failure settings under runs. Existing
`runs/` directories are rejected. Reuse saved preparation with `--prepared DIRECTORY`; recompute
metrics after the simulator closes with `avsectester evaluate EXPERIMENT.yaml`. Experiment and
optimization settings belong in the YAML files. Fixed and optimized patches use separate
configurations. The stop-sign patch configurations include nine independently prepared attacks;
the clean treatment repeats in every attack group.

## Preparation

- [fetch_models.sh](preparation/fetch_models.sh) downloads the perception models used by the
  supplied driving configurations.
- [export_carla_maps.py](preparation/export_carla_maps.py) connects once to CARLA, loads the
  requested towns and writes local OpenDRIVE files for CarlaDataset.map_directory. Use
  `python -m scripts.preparation.export_carla_maps --town Town03 --output data/carla_maps`.
- [extract_person_cutouts.py](preparation/extract_person_cutouts.py) extracts reusable person
  assets using SAM and nuScenes annotations.

See [installation](../docs/getting-started/installation.md) for environments and model locations.

## Validation and visualization

Run Python tools as modules, for example:

```bash
python -m scripts.validation.patch_hide_probe --help
python -m scripts.validation.validate_carla_visibility --help
python -m scripts.validation.validate_nurec_depth --help
python -m scripts.visualization.visualize_scene_labels --help
python -m scripts.visualization.visualize_augmentations --help
python -m scripts.visualization.plot_stop_line_approaches configs/nurec_stop_sign_patch_alpamayo_2601.yaml
```

- [patch_hide_probe.py](validation/patch_hide_probe.py) compares patch sizes and harmonization
  on a CARLA SensorView using the same insertion action used by attacks.
- [validate_carla_visibility.py](validation/validate_carla_visibility.py) measures visibility
  with simulator-owned RGB/depth captures and full-silhouette reference masks on a dedicated CARLA
  world. It does not construct sensor actors, queues or depth decoders.
- [validate_nurec_depth.py](validation/validate_nurec_depth.py) checks NuRec mesh depth against
  the recorded rig and original clipgt lane-rail and road-boundary points. Ground checks cast down
  from 3 m above every recorded rig origin and require all hits and signed heights within ±0.1 m.
  Road-height acceptance uses original points within --road-radius-m of the complete recorded
  trajectory (10 m by default). Distance is horizontal distance to its continuous polyline,
  using every recorded rig sample, independently of --frames. The selected points must have
  at least 90% ground intersections and median absolute height error below 0.1 m. Median signed
  error, median and 90th-percentile absolute errors, hit fraction and missing-hit count are
  reported for this subset; missing points are never treated as zero error. All original road
  points also have separate point/hit counts in 0–10, 10–25, 25–50, 50–100 and 100+ m bands
  from that trajectory. These bands include distant points excluded from road-height acceptance.
  These checks run offline using mesh_ground.ply and the archive's native camera calibration.
  Signed height means ground z minus the sampled point's z, computed as 3 m minus hit distance.
  Relative optical-z errors in 0–10 m, 10–25 m and 25–50 m Euclidean camera-range bins are also
  reported, but do not determine pass/fail. Grazing views amplify centimetre-scale height offsets,
  while guardrails and vehicles in the mesh can occlude projected road points. Only an explicit
  --endpoint enables depth-edge overlays on NRE RGB images; otherwise overlays are marked skipped.
  If the archive has no native inverse lens polynomial, only these informational bins are marked
  unavailable with a reason. The two vertical checks still run, without fitting a camera model.
- [visualize_scene_labels.py](visualization/visualize_scene_labels.py) projects recorded
  FrameData actors onto their camera images, using timestamps to match video frames.
- [plot_stop_line_approaches.py](visualization/plot_stop_line_approaches.py) reads saved experiment
  records and plots speed against signed distance to the selected stop line, one figure per scene.
  Its default destination is stop_line_plots/ under runs.output; --output selects another directory.
- [visualize_augmentations.py](visualization/visualize_augmentations.py) applies sensor transforms
  to one image and writes a labelled comparison figure.

These tools write files only when invoked. They do not run as part of configuration preflight.

For offline NuRec ground and road validation, run from the checkout using an interpreter with the
nurec dependencies:

```bash
python -m scripts.validation.validate_nurec_depth \
  --usdz /path/to/scene.usdz --output /tmp/nurec-depth-validation \
  --camera-id camera_front_wide_120fov --frames 5 --road-radius-m 10
```

It writes depth_validation.json, including every rig sample's signed height. In road_height,
point_count and hit_count describe only the radius-selected acceptance subset;
total_source_points and all_point_distance_bands describe all original road points.
--road-radius-m must be positive and finite. Add
--endpoint 127.0.0.1:50051 to render depth_edges_NNNNN.png using an already running NRE service.
Without that option, no neural renderer is created or queried. Exit code 0 means both vertical
alignment checks passed; 1 means validation failed, and 2 indicates invalid command arguments.

The stop-sign patch configurations combine detector PGD and Alpamayo rollout NES in one
preparation process. Start that process with CUBLAS_WORKSPACE_CONFIG=:4096:8 set so
CuBLAS operations support the deterministic mode enabled by PGD. For example:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 avsectester run configs/nurec_stop_sign_patch_alpamayo_2601.yaml
```
